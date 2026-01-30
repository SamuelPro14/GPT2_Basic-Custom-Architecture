"""
Model Configuration Hub
-----------------------
This module defines the architectural blueprints for the transformer.
It includes the baseline GPT-2 scales and the hyperparameter toggles for 
custom features like GQA, Sliding Window Attention, and Mixture of Experts.
"""

BASE_CONFIG = {
    # Text Processing
    "vocab_size": 50257,        # Total 'words' the model knows (Standard GPT-2)
    "context_length": 1024,     # Maximum tokens the model can "remember" at once
    
    # Layer Stability
    "qkv_bias": True,           # Adds learnable bias to the Attention projections
    "drop_rate": 0.0,           # Regularization: % of neurons to disable during training
    "use_RMSNorm": False,       # Toggle: Standard LayerNorm vs faster Llama-style Norm
    
    # Research Architecture Switches
    "use_dual_stream": False,   # Toggle: Separates Global context from Local expertise
    "use_adaptive_moe": False,  # Toggle: Enables the Hierarchical Mixture of Experts
    "use_parallel_att": False,  # Toggle: Runs Attention and FFN at the same time (PaLM style)
    
    # GQA (Grouped Query Attention) 
    # 
    "kv_groups": 16,            # Number of KV heads (saves VRAM compared to standard MHA)
    
    # MoE (Mixture of Experts) Settings
    # 
    "e_num": 1,                 # Total number of available expert brains
    "moe_groups": 1,            # How we cluster experts into 'knowledge neighborhoods'
    "moe_group_top_p": 0.7,     # Adaptive threshold for selecting active groups
    "moe_top_p": 0.9,           # Adaptive threshold for selecting experts within groups
    "moe_max_groups": 1,        # Limit on how many groups a token can visit
    "moe_max_k": 1,             # Limit on how many experts per group a token can use
    
    # SWA (Sliding Window Attention)
    # 
    "window_size": 1024,        # Cache size for inference (ring buffer limit)
    "swa_size": 1024,           # Look-back limit during training (saves computation)
}

# Standard OpenAI Scaling blue-prints
model_configs = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Linking our readable names to the official Hugging Face Hub identifiers
mapping = {
    "gpt2-small (124M)":  "gpt2", 
    "gpt2-medium (355M)": "gpt2-medium",
    "gpt2-large (774M)":  "gpt2-large", 
    "gpt2-xl (1558M)":    "gpt2-xl"
}