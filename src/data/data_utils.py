"""
Data Processing & Loading Utilities
-----------------------------------
This module handles the end-to-end data pipeline:
1. Retrieval: Downloading Wikipedia (Pre-training) or Instruction sets (Fine-tuning).
2. Processing: Tokenization and Sliding Window chunking.
3. Batching: Custom collation with padding and target masking for stable GPU training.
"""

import os
import json
import requests
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

# Grabbing the Wiki-2 dataset for pretraining:
def fetch_wikitext_train():
    """Fetches high-quality Wikipedia text for architectural alignment."""
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    print("Downloading WikiText-2 (Train)...")
    return requests.get(url).text

def fetch_wikitext_val():
    """Fetches validation text to track Perplexity during training."""
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt"
    print("Downloading WikiText-2 (Valid)...")
    return requests.get(url).text


# Standard GPT Pre-training: Chunks a massive book into small, overlapping snippets.
# Each target is just the input shifted by one word (predict the next token).
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Turn the whole text into numbers first
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Text is too short for the window size!"

        # Sliding window: we grab a chunk, then shift by 'stride' to get the next one.
        # This gives the model many different 'perspectives' on the same text.
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1] # The next-word 'answer'
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# Instruction Fine-tuning: Packages specific Q&A pairs into a chat-like format.
# Instead of a sliding window, each entry is a complete "Prompt -> Response" unit.
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []

        for entry in data:
            # We wrap the user's request and the model's answer into one big string.
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            
            # Save the whole conversation as one encoded block
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


# Following the course methodology for Instruction Fine-Tuning: 
# This template turns raw dictionary data into a structured prompt that the 
# model recognizes as a command rather than just a sequence of random text.
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

# As required for the fine-tuning phase: This 'tailor' function handles the 
# varied lengths of human instructions by padding them into uniform batches 
# and masking the 'filler' tokens so the model only learns from the actual text.
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    This is our 'tailor' function. It takes a batch of different-length conversations 
    and pads them so they fit into a uniform square tensor that the GPU can process.
    """
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor