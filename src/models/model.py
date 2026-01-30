"""
Core Model Architecture & Weight Transplantation
------------------------------------------------
This module assembles the custom Transformer architecture, integrating advanced 
features like Grouped Query Attention (GQA), Sliding Window Attention (SWA), 
and Dual-Stream processing. 

It also contains the 'Surgical Loaders'—specialized functions that map 
standard GPT-2 weights into our custom architecture, even when the 
shapes and layers (like SwiGLU and MoE) don't match perfectly.
"""


import math
import torch
import torch.nn as nn
import gc
from huggingface_hub import hf_hub_download
from src.models.layers import NormLayer, RMSNorm, FFN
from src.models.attention import GQA_SWA_Flash
from src.models.moe_atlas import HDynMoF


class TransformerBlock_GQA_SWA(nn.Module):
    """
    The TransformerBlock is the fundamental computational unit of the model. 
    It is designed to be highly flexible, supporting three distinct 'wiring' modes:
    
    1. Classic Mode: Sequential processing where attention is followed by the FFN.
    2. Parallel Mode: Attention and FFN run simultaneously for faster computation.
    3. Dual-Stream Mode: A custom experimental architecture that separates global 
       context (Attention) from local expertise (MoE) into two parallel paths.
    """
    def __init__(self, cfg):
        super().__init__()

        self.use_parallel_att = cfg.get("use_parallel_att", False)
        self.use_RMSNorm      = cfg.get("use_RMSNorm", False)
        self.use_adaptive_moe = cfg.get("use_adaptive_moe", False)

        # NEW: dual global/local stream mode (push global/local all the way)
        # When True, we keep separate global (attn) and local (MoE) streams
        # through all layers and only combine at the very end of the model.
        self.use_dual_stream  = cfg.get("use_dual_stream", False)

        self.att = GQA_SWA_Flash(
            emb_dim=cfg["emb_dim"],
            model_dim=cfg["emb_dim"],
            max_context_len=cfg["context_length"],
            drop_rate=cfg["drop_rate"],
            heads_num=cfg["n_heads"],
            kv_groups=cfg["kv_groups"],
            swa_size=cfg["swa_size"],
            qkv_bias=cfg["qkv_bias"],
        )

        # --- MoE depend on mode ---
        MoE_FFN = HDynMoF if self.use_adaptive_moe else FFN
        self.ff = MoE_FFN(cfg)
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

        # --- Norms depend on mode ---
        Normalization = RMSNorm if self.use_RMSNorm else NormLayer

        if self.use_dual_stream:
            # Dual-stream: separate norms for global and local paths
            #  - global stream goes through attention only
            #  - local stream goes through MoE/FFN only
            self.ln_global = Normalization(cfg["emb_dim"])
            self.ln_local  = Normalization(cfg["emb_dim"])
            # We keep a √2 scale handy if we ever want to scale inside the block
            self.res_scale = 1.0 / math.sqrt(2.0)
            self.enhance_global_ctx = nn.GELU()

        elif self.use_parallel_att:
            # Single norm for both att and ff paths (parallel residual)
            self.ln = Normalization(cfg["emb_dim"])
            self.res_scale = 1.0 / math.sqrt(2.0)
        else:
            # Classic pre-norm: separate ln1/ln2
            self.ln1 = Normalization(cfg["emb_dim"])
            self.ln2 = Normalization(cfg["emb_dim"])

    def forward(self, x, use_cache: bool = False, use_swa: bool = False):
        if self.use_dual_stream:
            # Dual-stream mode:
            # x is a tuple: (x_global, x_local)
            x_global, x_local = x

            # Global stream: normalized then passed through attention (global / prefix semantics)
            y_global = self.ln_global(x_global)
            att_out  = self.att(y_global, use_cache=use_cache, use_swa=use_swa)

            # Local stream: normalized then passed through MoE/FFN (local token-wise expertise)
            y_local = self.ln_local(x_local)
            ff_out  = self.ff(y_local)

            # Residual updates are kept separate for the two streams
            enhanced_att = self.enhance_global_ctx(att_out)
            x_global = x_global + self.drop_shortcut(enhanced_att)
            x_local  = x_local  + self.drop_shortcut(ff_out)

            return (x_global, x_local)

        if self.use_parallel_att:
            # Parallel residual: x + (att(ln(x)) + ff(ln(x))) / √2
            y = self.ln(x)
            att_out = self.att(y, use_cache=use_cache, use_swa=use_swa)
            ff_out  = self.ff(y)
            z = att_out + ff_out
            z = self.res_scale * z
            x = x + self.drop_shortcut(z)
            return x
        else:
            # Classic: x + Att(ln1(x)) then x + FF(ln2(x))
            y = self.ln1(x)
            x = x + self.drop_shortcut(self.att(y, use_cache=use_cache, use_swa=use_swa))

            y2 = self.ln2(x)
            x = x + self.drop_shortcut(self.ff(y2))

            return x
        
        
class MyGPT_GQA_SWA(nn.Module):
    """
    Top-level Transformer orchestrator implementing a decoupled Dual-Stream 
    architecture with Grouped-Query Attention (GQA) and Sliding Window Attention (SWA).

    Key Features:
    - Dual-Stream Logic: Separates global context from local processing.
    - Inference Optimization: Integrated KV-cache management for O(1) decoding.
    - MODERN Components: Support for RMSNorm, SwiGLU, and Flash-based attention.
    """
    def __init__(self, cfg):
        super().__init__()

        self.use_RMSNorm = cfg.get("use_RMSNorm", False)
        # NEW: propagate dual-stream flag to model level
        self.use_dual_stream = cfg.get("use_dual_stream", False)

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Use ModuleList so we can pass use_cache through each block
        self.trm_blocks = nn.ModuleList(
            [TransformerBlock_GQA_SWA(cfg) for _ in range(cfg["n_layers"])]
        )

        # --- Norms depend on mode ---
        Normalization = RMSNorm if self.use_RMSNorm else NormLayer
        self.final_norm = Normalization(cfg["emb_dim"])

        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # Track current position in cached sequence for inference
        # (how many tokens have been seen so far in cache mode)
        self.cache_pos = 0

        # For final combination of global/local streams (when use_dual_stream=True)
        self._dual_res_scale = 1.0 / math.sqrt(2.0)

    def reset_cache(self):
        """Resets the KV cache in all attention modules within the transformer blocks."""
        for block in self.trm_blocks:
            block.att.reset_cache()
        # Also reset our positional counter
        self.cache_pos = 0

    def forward(self, x, use_cache: bool = False, use_swa: bool = False):
        batch_size, seq_len = x.shape
        device = x.device

        if use_cache:
            # In cache mode, we may be doing:
            #  - prefill: first call after reset_cache (seq_len = T_prompt)
            #  - incremental decode: subsequent calls (seq_len = 1)
            # We must ensure we don't exceed the max context length.
            assert self.cache_pos + seq_len <= self.pos_emb.num_embeddings, (
                f"Total length {self.cache_pos + seq_len} exceeds context length "
                f"{self.pos_emb.num_embeddings}"
            )

            # Positions continue from where the cache left off
            pos_start = self.cache_pos
            pos_ids = torch.arange(pos_start, pos_start + seq_len, device=device)
            self.cache_pos += seq_len
        else:
            # Training / no cache: positions always start at 0
            assert seq_len <= self.pos_emb.num_embeddings, \
                f"Sequence length {seq_len} exceeds context length {self.pos_emb.num_embeddings}"
            pos_ids = torch.arange(seq_len, device=device)

        # Token + positional embeddings
        x = self.tok_emb(x) + self.pos_emb(pos_ids)
        x = self.drop_emb(x)

        if self.use_dual_stream:
            # NEW: dual global/local streams
            #  - x_global: updated only by attention (global prefix semantics)
            #  - x_local:  updated only by MoE/FFN (local token-wise expertise)
            x_global = x
            x_local  = x

            state = (x_global, x_local)
            # Pass use_cache through every TransformerBlockGQA
            for block in self.trm_blocks:
                state = block(state, use_cache=use_cache, use_swa=use_swa)

            x_global, x_local = state

            # Only now combine global + local once, before final norm + classifier
            x = (x_global + x_local) * self._dual_res_scale
        else:
            # Pass use_cache through every TransformerBlockGQA
            for block in self.trm_blocks:
                x = block(x, use_cache=use_cache, use_swa=use_swa)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
 
    
def load_standard_baseline(model, sd_hf):
    """
    Surgical loader 100% aligned with NormLayer (gamma/bias)
    and MyGPT_GQA_SWA architecture.
    """
    print("🩹 Initializing Baseline weight mapping...")

    with torch.no_grad():
        # 1. GLOBAL EMBEDDINGS
        model.tok_emb.weight.copy_(sd_hf['wte.weight'])
        model.pos_emb.weight.copy_(sd_hf['wpe.weight'])

        # 2. TRANSFORMER BLOCKS
        for i, block in enumerate(model.trm_blocks):
            prefix = f'h.{i}.'

            # --- A. Normalization (Using .gamma and .bias from your NormLayer) ---
            block.ln1.gamma.copy_(sd_hf[f'{prefix}ln_1.weight'])
            block.ln1.bias.copy_(sd_hf[f'{prefix}ln_1.bias'])
            block.ln2.gamma.copy_(sd_hf[f'{prefix}ln_2.weight'])
            block.ln2.bias.copy_(sd_hf[f'{prefix}ln_2.bias'])

            # --- B. Attention ---
            qkv_w = sd_hf[f'{prefix}attn.c_attn.weight'].t()
            qkv_b = sd_hf[f'{prefix}attn.c_attn.bias']
            qw, kw, vw = qkv_w.chunk(3, dim=0)
            qb, kb, vb = qkv_b.chunk(3, dim=0)

            block.att.W_q.weight.copy_(qw); block.att.W_q.bias.copy_(qb)
            block.att.W_k.weight.copy_(kw); block.att.W_k.bias.copy_(kb)
            block.att.W_v.weight.copy_(vw); block.att.W_v.bias.copy_(vb)

            block.att.final_proj.weight.copy_(sd_hf[f'{prefix}attn.c_proj.weight'].t())
            block.att.final_proj.bias.copy_(sd_hf[f'{prefix}attn.c_proj.bias'])

            # --- C. Feed-Forward ---
            block.ff.layers[0].weight.copy_(sd_hf[f'{prefix}mlp.c_fc.weight'].t())
            block.ff.layers[0].bias.copy_(sd_hf[f'{prefix}mlp.c_fc.bias'])
            block.ff.layers[2].weight.copy_(sd_hf[f'{prefix}mlp.c_proj.weight'].t())
            block.ff.layers[2].bias.copy_(sd_hf[f'{prefix}mlp.c_proj.bias'])

        # 3. FINAL HEAD (Fixed to use .gamma)
        model.final_norm.gamma.copy_(sd_hf['ln_f.weight'])
        model.final_norm.bias.copy_(sd_hf['ln_f.bias'])
        model.out_head.weight.copy_(sd_hf['wte.weight'])

    print("✅ Baseline weights loaded successfully.")


def load_gpt2_weights_raw(model, model_type="gpt2"):
    """
    Loads GPT-2 weights directly from the raw bin file.
    No GPT2LMHeadModel initialization = No Triton/AO warnings.
    """
    print(f"📥 Downloading raw {model_type} weight file...")

    # Downloads ONLY the weight file, not the whole model class
    state_dict_path = hf_hub_download(repo_id=model_type, filename="pytorch_model.bin")
    sd_hf = torch.load(state_dict_path, map_location="cpu", weights_only=True)

    # GPT-2 weights in the bin file often have a 'transformer.' prefix
    # We strip it to make matching easier
    sd_hf = {k.replace("transformer.", ""): v for k, v in sd_hf.items()}

    print(f"🏗️ Transplanting weights into custom architecture...")

    with torch.no_grad():
        # 1. Embeddings
        model.tok_emb.weight.copy_(sd_hf['wte.weight'])
        model.pos_emb.weight.copy_(sd_hf['wpe.weight'])

        # 2. Sequential Blocks
        for i, block in enumerate(model.trm_blocks):
            prefix = f'h.{i}.'

            # --- A. Attention Weights (Conv1D to Linear) ---
            # Raw GPT-2 weights are stored as [input, output], we need [output, input]
            qkv_w = sd_hf[f'{prefix}attn.c_attn.weight'].t()
            qkv_b = sd_hf[f'{prefix}attn.c_attn.bias']

            qw, kw, vw = qkv_w.chunk(3, dim=0)
            qb, kb, vb = qkv_b.chunk(3, dim=0)

            # GQA Head Sub-sampling logic
            step = model.tok_emb.embedding_dim // (block.att.head_dim * block.att.kv_groups)

            block.att.W_q.weight.copy_(qw)
            block.att.W_k.weight.copy_(kw[::step])
            block.att.W_v.weight.copy_(vw[::step])
            block.att.final_proj.weight.copy_(sd_hf[f'{prefix}attn.c_proj.weight'].t())

            block.att.W_q.bias.copy_(qb)
            block.att.W_k.bias.copy_(kb[::step])
            block.att.W_v.bias.copy_(vb[::step])
            block.att.final_proj.bias.copy_(sd_hf[f'{prefix}attn.c_proj.bias'])

            # --- B. MLP to MoE SwiGLU Upcycling ---
            mlp_fc_w = sd_hf[f'{prefix}mlp.c_fc.weight'].t()
            mlp_fc_b = sd_hf[f'{prefix}mlp.c_fc.bias']
            mlp_proj_w = sd_hf[f'{prefix}mlp.c_proj.weight'].t()
            mlp_proj_b = sd_hf[f'{prefix}mlp.c_proj.bias']

            for group in block.ff.experts:
                for expert in group:
                    # Map pre-trained knowledge to SwiGLU gates
                    expert.swiGLU.W1.weight.copy_(mlp_fc_w[:expert.swiGLU.W1.out_features])
                    expert.swiGLU.W1.bias.copy_(mlp_fc_b[:expert.swiGLU.W1.out_features])
                    expert.swiGLU.W2.weight.copy_(mlp_fc_w[:expert.swiGLU.W2.out_features])
                    expert.swiGLU.W2.bias.copy_(mlp_fc_b[:expert.swiGLU.W2.out_features])
                    expert.W_out.weight.copy_(mlp_proj_w)
                    expert.W_out.bias.copy_(mlp_proj_b)

        # 3. Final Norm and LM Head
        model.final_norm.weight.copy_(sd_hf['ln_f.weight'])
        if hasattr(model.final_norm, 'bias') and model.final_norm.bias is not None:
            model.final_norm.bias.copy_(sd_hf['ln_f.bias'])

        # NOTE: GPT-2 often ties weights between wte and lm_head
        # If your out_head is separate, copy it from wte.weight or lm_head.weight
        lm_head_key = 'lm_head.weight' if 'lm_head.weight' in sd_hf else 'wte.weight'
        model.out_head.weight.copy_(sd_hf[lm_head_key])

    # Final Memory Cleanup
    del sd_hf
    gc.collect()
    print("✅ Weights successfully transplanted without Transformers Model class!")