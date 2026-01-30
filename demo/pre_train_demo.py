import torch
import tiktoken
import time
import os
import sys
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from configs.base_config import BASE_CONFIG, model_configs
from src.models.model import MyGPT_GQA_SWA
from src.data.data_utils import create_dataloader_v1
from src.engine import train_model_simple, plot_losses


# ==========================================
# GLOBAL CONFIGURATION & HYPERPARAMETERS
# ==========================================
CHOSEN_MODEL_SIZE = "gpt2-small (124M)"
MODEL_TYPE = "gpt2"

# Training Hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 2
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 0.1
EVAL_FREQ = 5
EVAL_ITER = 5
CONTEXT_LENGTH = 256

# Generation Settings
START_CONTEXT = "The future of AI is"

FILE_PATH = "the-verdict.txt"
URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

def main():
    # -------------------------------------------------------------------
    # 1. HARDWARE & DEVICE SETUP
    # -------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # -------------------------------------------------------------------
    # 2. CONFIGURATION & ARCHITECTURE
    # -------------------------------------------------------------------
    cfg = BASE_CONFIG.copy()
    cfg.update(model_configs[CHOSEN_MODEL_SIZE])
    cfg["kv_groups"] = model_configs[CHOSEN_MODEL_SIZE]["n_heads"]
    cfg["context_length"] = CONTEXT_LENGTH
    
    print(f"🏗️ Initializing {CHOSEN_MODEL_SIZE} architecture...")
    model = MyGPT_GQA_SWA(cfg).to(device)
    
    # -------------------------------------------------------------------
    # 3. DATA PREPARATION 
    # -------------------------------------------------------------------
    # text_data = download_and_load_file(FILE_PATH, URL)
    if not os.path.exists(FILE_PATH):
        response = requests.get(URL, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(FILE_PATH, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(FILE_PATH, "r", encoding="utf-8") as file:
            text_data = file.read()
    
    train_ratio = 0.7
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=cfg["context_length"],
        stride=cfg["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=cfg["context_length"],
        stride=cfg["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    
    # -------------------------------------------------------------------
    # 4. OPTIMIZER & ENGINE SETUP
    # -------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    
    print(f"🔥 Starting Training on: {device}")
    start_time = time.time()
    
    train_losses, val_losses, track_tokens_seen = train_model_simple(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        device=device,
        num_epochs=NUM_EPOCHS, 
        eval_freq=EVAL_FREQ, 
        eval_iter=EVAL_ITER,
        start_context=START_CONTEXT, 
        tokenizer=tokenizer,
        cfg=cfg,                  # Pass blueprint for internal best-model saves
        do_live_sample=False,            
        do_save_best=False
    )
    
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    
    print("-" * 30)
    print(f"✅ Training completed in {execution_time_minutes:.2f} minutes.")
    
    epochs_tensor = torch.linspace(0, NUM_EPOCHS, len(train_losses))
    plot_losses(epochs_tensor, track_tokens_seen, train_losses, val_losses)
    

if __name__ == "__main__":
    main()