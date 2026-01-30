import torch
import tiktoken
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from configs.base_config import BASE_CONFIG, model_configs
from src.models.model import MyGPT_GQA_SWA
from src.data.data_utils import download_and_load_file, format_input
from src.engine import text_to_token_ids, token_ids_to_text, generate

# MODEL_PATH = r".\test_ready_bin_model\gpt2_medium_chat_finetuned.pth"
MODEL_PATH = r"./demo/chat_mode_for_demo.pth"
FILE_PATH = "instruction-data.json"
URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"

def main():
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    data = download_and_load_file(FILE_PATH, URL)
    test_data = data[int(len(data) * 0.85):int(len(data) * 0.95)] # Adjusted slice for clarity
    
    # Configure architecture
    CHOSEN_MODEL = "gpt2-medium (355M)"
    chat_cfg = BASE_CONFIG.copy()
    chat_cfg.update(model_configs[CHOSEN_MODEL])
    chat_cfg.update({
        "use_dual_stream": False,
        "use_adaptive_moe": False,
        "use_parallel_att": False,
        "kv_groups": chat_cfg["n_heads"], 
        "drop_rate": 0.0 # Setting dropout to 0 for inference is safer
    })

    # Step 1: Initialize the "Body" (Architecture)
    model = MyGPT_GQA_SWA(chat_cfg)
    
    # 1. Load the raw data
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    # 2. Dig into the 'package' if needed
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # 3. Strip any 'decoration' prefixes (_orig_mod. or module.)
    clean_sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}

    # 4. Load it!
    msg = model.load_state_dict(clean_sd, strict=False)
    print(f"📦 Load Report: {msg}")
    model.to(device)
    model.eval()
    
    print(f"Loaded model to {device}. Starting generation...\n")

    with torch.no_grad(): # Saves memory by not tracking gradients
        for entry in test_data[:10]:
            input_text = format_input(entry)
            
            # Generate response
            input_ids = text_to_token_ids(input_text, tokenizer).to(device)
            token_ids = generate(
                model=model,
                idx=input_ids,
                max_new_tokens=256, # Reduced for faster testing
                context_size=chat_cfg["context_length"],
                eos_id=tokenizer.eot_token
            )
            
            # Cleanup and print
            full_text = token_ids_to_text(token_ids, tokenizer)
            response_text = full_text[len(input_text):].replace("### Response:", "").strip()

            print(f"PROMPT:\n{input_text}")
            print(f"\nGPT RESPONSE:\n>> {response_text}")
            print(f"\nCORRECT RESPONSE:\n>> {entry['output']}")
            print("-" * 125)

if __name__ == "__main__":
    main()