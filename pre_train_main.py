"""
GPT-2 Pre-training & Up-training Template
-----------------------------------------
This module serves as a flexible baseline for training or "up-training" a 
custom GQA/SWA GPT-2 model. It is designed as a highly configurable template 
that can be adapted for raw pre-training on new datasets or specialized 
continued training from pre-existing weights.

Key Versatility:
- Modular Configuration: Easily adjust model size, attention groups (KV_GROUPS), 
  and context window through the Global Configuration section.
- Transfer Learning Ready: Features a dedicated 'Weight Loading' section 
  to manually toggle between random initialization and surgical "weight 
  transplants" from standard GPT-2.
- Data Agnostic: Uses WikiText-2 as a default baseline, but can be 
  swapped for any text-based corpus.
- Blueprint Checkpointing: Saves comprehensive state dictionaries that 
  bundle the architecture's 'DNA' (config) with its weights for zero-guess loading.
"""

import torch
import tiktoken
import time
from configs.base_config import BASE_CONFIG, model_configs
from src.models.model import MyGPT_GQA_SWA
from src.data.data_utils import fetch_wikitext_train, fetch_wikitext_val, create_dataloader_v1
from src.engine import train_model_simple

# =========================================================================
# ⚠️ USER INSTRUCTION: WEIGHT INITIALIZATION REQUIRED
# =========================================================================
# This script initializes the architecture with RANDOM weights by default.
# If you want to use pre-trained weights (Up-training), you MUST manually 
# add the loading call in SECTION 3:
#
# A) TO UP-TRAIN FROM CLASSICAL GPT-2:
#    Use: load_gpt2_weights_raw(model, model_type="gpt2")
#    (Ensures the model starts with standard OpenAI knowledge)
#
# B) TO UP-TRAIN FROM CUSTOM ARCHITECTURE (GQA/SWA/MoE):
#    Use: load_gpt2_weights_raw(model, model_type="gpt2")
#    (The specialized surgeon logic slices weights to fit custom layers)
# =========================================================================

# ==========================================
# GLOBAL CONFIGURATION & HYPERPARAMETERS
# ==========================================
CHOSEN_MODEL_SIZE = "gpt2-small (124M)"
MODEL_TYPE = "gpt2"

# Training Hyperparameters
NUM_EPOCHS = 1
BATCH_SIZE = 2
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.1
EVAL_FREQ = 50
EVAL_ITER = 5

CONTEXT_LENGTH = 256
KV_GROUPS = 4

# Generation Settings
START_CONTEXT = "The future of AI is"

# Path to save the model blueprint and weights
SAVE_PATH = "gpt2_pretrain_baseline.pth"        # Path to save the 'Safety' version when training is 100% done
BEST_VAL_PATH = "best_baseline_model.pth"       # Path to save the 'Val Loss Champion' (lowest error model)
EVAL_WITH_TXT_GENERATION_SAMPLE=True            # Toggle: Generate text snippets during training to see progress
SAVE_BEST_AFTER_EVAL=True                       # Toggle: Automatically save files when a better model is found

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
    cfg["kv_groups"] = KV_GROUPS
    cfg["context_length"] = CONTEXT_LENGTH
    
    print(f"🏗️ Initializing {CHOSEN_MODEL_SIZE} architecture...")
    model = MyGPT_GQA_SWA(cfg).to(device)
    
    # -------------------------------------------------------------------
    # 3. WEIGHT LOADING (MANUAL CHOICE REQUIRED)
    # -------------------------------------------------------------------
    print("🌱 Weights initialized: RANDOM (Modify Section 3 to load)")

    # -------------------------------------------------------------------
    # 4. DATA PREPARATION (WikiText-2 for Baseline)
    # -------------------------------------------------------------------
    print("📥 Fetching training and validation text...")
    train_txt = fetch_wikitext_train()
    val_txt = fetch_wikitext_val()
    
    train_loader = create_dataloader_v1(
        train_txt, batch_size=BATCH_SIZE, max_length=cfg["context_length"]
    )
    val_loader = create_dataloader_v1(
        val_txt, batch_size=BATCH_SIZE, max_length=cfg["context_length"]
    )
    
    # -------------------------------------------------------------------
    # 5. OPTIMIZER & ENGINE SETUP
    # -------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    
    print(f"🔥 Starting Training on: {device}")
    start_time = time.time()
    
    # ALIGNED CALL: Passing cfg and BEST_VAL_PATH to the engine for checkpointing
    # This allows the engine to save a 'Self-Describing' model during training.
    train_losses, val_losses, tokens_seen = train_model_simple(
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
        cfg=cfg,                       
        checkpoint_path=BEST_VAL_PATH,
        final_save_path=SAVE_PATH,   
        do_live_sample=EVAL_WITH_TXT_GENERATION_SAMPLE,            
        do_save_best=SAVE_BEST_AFTER_EVAL
    )
    
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60

    # -------------------------------------------------------------------
    # 6. FINAL COMPREHENSIVE SAVE
    # -------------------------------------------------------------------
    # This bundle includes the architecture blueprint (config) and metrics
    # so the model can be reconstructed easily for later inference.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "tokenizer_info": "gpt2",
        "train_losses": train_losses,
        "val_losses": val_losses,
        "tokens_seen": tokens_seen
    }
    torch.save(checkpoint, SAVE_PATH)
    
    print("-" * 30)
    print(f"✅ Training completed in {execution_time_minutes:.2f} minutes.")
    print(f"💾 Final checkpoint saved as: '{SAVE_PATH}'")
    print("🚀 Model is ready for baseline testing or chat fine-tuning!")

if __name__ == "__main__":
    main()