"""
GPT-2 Chat Fine-Tuning Module
-----------------------------
This script performs instruction-based fine-tuning on a GPT-2 (355M) (by default) architecture. 
It utilizes a "weight transplant" technique to load baseline GPT-2 weights into 
a custom GQA/SWA model (by default configurated like standard classic GPT-2).

Key Features:
- Automated Hardware Setup: Detects and optimizes for CUDA, MPS, or CPU.
- Weight Transplant: Seamlessly migrates standard baseline weights to custom layers.
- Instruction Training: Processes structured JSON data for chat-specific fine-tuning.
- Comprehensive Checkpointing: Saves both model weights and architectural blueprints (DNA).
"""


import time
import torch
import tiktoken
from functools import partial
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

# Local imports from your project structure
from configs.base_config import BASE_CONFIG, model_configs, mapping
from src.models.model import MyGPT_GQA_SWA, load_standard_baseline
from src.data.data_utils import (
    download_and_load_file, 
    InstructionDataset, 
    custom_collate_fn, 
    format_input
)
from src.engine import train_model_simple

# ==========================================
# GLOBAL CONFIGURATION & HYPERPARAMETERS
# ==========================================
CHOSEN_MODEL = "gpt2-medium (355M)"
DATA_URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
DATA_FILE_PATH = "instruction-data.json"

# File path for saving the final model
SAVE_PATH = "gpt2_medium_chat_finetuned.pth"
# Path for intermediate best-model saves inside the engine
BEST_VAL_PATH = "current_best_chat_model.pth"

EVAL_WITH_TXT_GENERATION_SAMPLE=True            # Toggle: Generate text snippets during training to see progress
SAVE_BEST_AFTER_EVAL=True                       # Toggle: Automatically save files when a better model is found

# Training Hyperparameters
NUM_EPOCHS = 2
BATCH_SIZE = 2  # Adjusted for memory safety on medium model
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.1
EVAL_FREQ = 20
EVAL_ITER = 20

# Data Split Ratios
TRAIN_RATIO = 0.85
TEST_RATIO = 0.10
# Validation is implicitly the remainder (0.05)

# Hardware Settings
RANDOM_SEED = 123
NUM_WORKERS = 0

def main():
    # -------------------------------------------------------------------
    # 1. HARDWARE & DEVICE SETUP
    # -------------------------------------------------------------------
    torch.manual_seed(RANDOM_SEED)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Use PyTorch 2.9 or newer for stable mps results
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"🚀 Training on Device: {device}")

    # -------------------------------------------------------------------
    # 2. CONFIGURATION & ARCHITECTURE
    # -------------------------------------------------------------------
    chat_cfg = BASE_CONFIG.copy()
    chat_cfg.update(model_configs[CHOSEN_MODEL])
    
    # Setting parameters for Chat Fine-tuning
    chat_cfg.update({
        "use_dual_stream": False,
        "use_adaptive_moe": False,
        "use_parallel_att": False,
        "kv_groups": chat_cfg["n_heads"], # Force MHA (16 heads) for 1:1 Baseline
        "drop_rate": 0.1                 # Adding dropout to prevent overfitting on instructions
    })

    print(f"🏗️ Initializing {CHOSEN_MODEL} architecture...")
    model = MyGPT_GQA_SWA(chat_cfg).to(device)
    tokenizer = tiktoken.get_encoding("gpt2")

    # -------------------------------------------------------------------
    # 3. WEIGHT TRANSPLANT (Baseline GPT-2 weights into Custom Architecture)
    # -------------------------------------------------------------------
    model_type = mapping[CHOSEN_MODEL] # Gets "gpt2-medium"
    print(f"📥 Downloading raw {model_type} weights from Hugging Face...")

    # Downloads ONLY the .bin file (approx 1.4GB for medium)
    state_dict_path = hf_hub_download(repo_id=model_type, filename="pytorch_model.bin")
    sd_hf = torch.load(state_dict_path, map_location="cpu", weights_only=True)

    # Clean keys: GPT-2 weights often have a 'transformer.' prefix that we strip
    sd_hf = {k.replace("transformer.", ""): v for k, v in sd_hf.items()}

    # This moves numbers from sd_hf into your model's parameters
    load_standard_baseline(model, sd_hf)
    
    # Verify Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Weight transplant successful! Parameters: {total_params:,}")

    # -------------------------------------------------------------------
    # 4. DATA PREPARATION (Instruction Following)
    # -------------------------------------------------------------------
    print("📊 Preparing Instruction Dataset...")
    
    data = download_and_load_file(DATA_FILE_PATH, DATA_URL)

    # Splitting data (85% Train, 10% Test, 5% Val)
    train_portion = int(len(data) * TRAIN_RATIO)
    test_portion = int(len(data) * TEST_RATIO)
    
    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    # Collate function setup
    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )

    # DataLoaders
    train_loader = DataLoader(
        InstructionDataset(train_data, tokenizer), 
        batch_size=BATCH_SIZE, 
        collate_fn=customized_collate_fn,
        shuffle=True, 
        drop_last=True,
        num_workers=NUM_WORKERS
    )
    
    val_loader = DataLoader(
        InstructionDataset(val_data, tokenizer), 
        batch_size=BATCH_SIZE, 
        collate_fn=customized_collate_fn,
        shuffle=False, 
        drop_last=False,
        num_workers=NUM_WORKERS
    )

    # -------------------------------------------------------------------
    # 5. TRAINING / FINE-TUNING ENGINE
    # -------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,            # Standard for 355M params
        weight_decay=WEIGHT_DECAY,   # High decay helps generalize to new tasks
        betas=(0.9, 0.95),           # GPT-2 standard betas
        eps=1e-8
    )

    print("🔥 Starting Chat Fine-Tuning Loop...")
    start_time = time.time()

    # ALIGNED CALL: Passing cfg and checkpoint_path to match updated engine logic
    train_losses, val_losses, track_tokens_seen = train_model_simple(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        device=device,
        num_epochs=NUM_EPOCHS, 
        eval_freq=EVAL_FREQ, 
        eval_iter=EVAL_ITER,
        start_context=format_input(val_data[0]), 
        tokenizer=tokenizer,
        cfg=chat_cfg,                  # Pass blueprint for internal best-model saves
        checkpoint_path=BEST_VAL_PATH, # Path for internal safety saves
        do_live_sample=EVAL_WITH_TXT_GENERATION_SAMPLE,            
        do_save_best=SAVE_BEST_AFTER_EVAL
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"✨ Training completed in {execution_time_minutes:.2f} minutes.")

    # -------------------------------------------------------------------
    # 6. SAVE COMPREHENSIVE CHECKPOINT (For portability and future testing)
    # -------------------------------------------------------------------
    # We save the config so anyone loading the model can reconstruct the 
    # specific architecture without having to guess the parameters.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": chat_cfg,
        "tokenizer_info": "gpt2",
        "train_losses": train_losses,
        "val_losses": val_losses,
        "tokens_seen": track_tokens_seen
    }
    
    torch.save(checkpoint, SAVE_PATH)
    
    print(f"💾 Comprehensive checkpoint saved as: '{SAVE_PATH}'")
    print("🚀 Model is ready for inference!")

if __name__ == "__main__":
    main()