"""
Advanced Attention Mechanism: GQA + SWA + Flash
----------------------------------------------
This module implements a high-performance attention layer that combines:
1. Grouped Query Attention (GQA): Reduces KV-cache memory footprint.
2. Sliding Window Attention (SWA): Limits attention span for linear scaling.
3. Flash Attention: Optimized CUDA kernels for accelerated computation.

Operational Modes:
- Training: Full sequence processing with optional SWA masking.
- Inference: Optimized 'Ring Buffer' KV-caching for low-latency decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA_SWA_Flash(nn.Module):
    """
    Grouped Query Attention (GQA) + Sliding Window Attention (SWA).
    
    The goal here is speed. GQA saves memory by sharing keys/values across 
    multiple query heads, and SWA saves computation by making sure tokens 
    don't look too far back into the past.
    """
    def __init__(self,
                 emb_dim,
                 model_dim,
                 max_context_len,
                 drop_rate,
                 heads_num,
                 kv_groups=None,
                 window_size=None,
                 swa_size=None,
                 qkv_bias=False):
        super().__init__()
        self.model_dim = model_dim
        self.heads_num = heads_num                     # total query heads H

        # Safety: ensure model_d is divisible by number of heads
        assert model_dim % heads_num == 0, "model_dim must be divisible by heads_num"

        self.head_dim = model_dim // heads_num           # dim per head
        self.kv_groups = kv_groups or 1                # number of KV groups G
        assert self.heads_num % self.kv_groups == 0, \
            "heads_num must be divisible by kv_groups"
        self.q_in_group = self.heads_num // self.kv_groups  # H/G heads per KV group

        # Linear projections for Query, Key, Value
        self.W_q = nn.Linear(emb_dim, model_dim, bias=qkv_bias)
        # In GQA, K/V have kv_groups groups of head_dim each: G * d_head
        self.W_k = nn.Linear(emb_dim, self.head_dim * self.kv_groups, bias=qkv_bias)
        self.W_v = nn.Linear(emb_dim, self.head_dim * self.kv_groups, bias=qkv_bias)

        self.window_size = window_size or max_context_len   # for KV cache / inference
        self.swa_size = swa_size or max_context_len         # for SWA during training
        self.drop_rate = drop_rate

        # --- KV cache (inference only) ---
        # Will hold: (b, kv_groups, window_size, head_dim) in GQA
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

        # Pointer into the ring buffer
        self.cache_ptr = 0
        self.cache_len = 0   # actual number of valid tokens in cache

        self.final_proj = nn.Linear(model_dim, model_dim)

    def reset_cache(self):
        """Call this before starting a new sequence."""
        self.cache_k = None
        self.cache_v = None
        self.cache_ptr = 0
        self.cache_len = 0

    def sliding_window_mask(self, seq_len, device):
        """
        Build SWA mask of shape (T, T):
        True  = masked
        False = allowed
        Token i can attend j if:
          j <= i and (i - j) < swa_size
        """
        i = torch.arange(seq_len, device=device).unsqueeze(1)  # (T, 1)
        j = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, T)
        # Mask positions where j > i (future) or too far in the past (i - j >= swa_size)
        return (j > i) | (i - j >= self.swa_size)

    def forward(self, x, use_cache: bool = False, use_swa: bool = False):
        """
        x:
          - training / no-cache: (b, T, emb_d)
          - inference / cache:
                * first call after reset_cache (prefill): (b, T_prompt, emb_d)
                * subsequent incremental calls:        (b, 1, emb_d)

        Returns:
            (b, T, model_d)
        """
        b_size, seq_len, _ = x.shape

        # ---- Q, K, V projections ----
        Q = self.W_q(x)      # (b, T, H * d_head)
        K_new = self.W_k(x)  # (b, T, G * d_head)
        V_new = self.W_v(x)  # (b, T, G * d_head)

        # ---- reshape Q into (b, H, T, d_head) ----
        Q = Q.view(b_size, seq_len, self.heads_num, self.head_dim).transpose(1, 2)
        # Q: (b, H, T_q, d_head)

        # ---- reshape K,V into (b, G, T, d_head) ----
        K_new = K_new.view(b_size, seq_len, self.kv_groups, self.head_dim).transpose(1, 2)
        V_new = V_new.view(b_size, seq_len, self.kv_groups, self.head_dim).transpose(1, 2)
        # K_new, V_new: (b, G, T_k_new, d_head)

        if use_cache:
            # ================= RING BUFFER KV-CACHE MODE =================

            # Sliding window attention (use_swa) is a training-only feature, not used in cache mode
            assert not use_swa, "use_swa is not supported when use_cache=True"

            # Initialize cache buffers for this batch if needed
            if self.cache_k is None or self.cache_k.size(0) != b_size:
                # Fixed-size ring buffers for KV groups
                # (b, G, window_size, d_head)
                self.cache_k = torch.zeros(
                    b_size, self.kv_groups, self.window_size, self.head_dim,
                    device=x.device, dtype=K_new.dtype
                )
                self.cache_v = torch.zeros_like(self.cache_k)
                self.cache_ptr = 0
                self.cache_len = 0

            if self.cache_len == 0:
                # ---------- PREFILL: FIRST CACHED CALL (FULL PROMPT) ----------
                # We process the entire prompt in one go and build the initial cache.
                # For simplicity and correctness, require prompt length <= window_size.
                assert seq_len <= self.window_size, \
                    "In prefill (first cache) call, seq_len must be <= window_size"

                # Store full prompt K/V into the beginning of the ring buffer
                # (no wrap-around since seq_len <= window_size and cache_ptr == 0)
                insert_len = seq_len
                end = insert_len
                # Use K_new, V_new directly: (b, G, T, d)
                self.cache_k[:, :, :end, :] = K_new
                self.cache_v[:, :, :end, :] = V_new

                self.cache_ptr = end % self.window_size
                self.cache_len = insert_len

                # For attention in prefill, just use K_new/V_new directly
                K = K_new  # (b, G, T_k, d_head)
                V = V_new  # (b, G, T_k, d_head)

                # Q keeps shape (b, H, T_q, d_head), with T_q == seq_len
                T_q = seq_len
                T_k = seq_len  # same as cache_len here

                # ---- expand K/V per query head within each group ----
                # Q: (b, H, T_q, d), we view it as (b, G, H/G, T_q, d)
                Qg = Q.reshape(b_size, self.kv_groups, self.q_in_group, T_q, self.head_dim)
                # K, V: (b, G, T_k, d) -> (b, G, 1, T_k, d) then broadcast over q_in_group
                Kg = K.unsqueeze(2)  # (b, G, 1, T_k, d)
                Vg = V.unsqueeze(2)  # (b, G, 1, T_k, d)

                # Broadcast to (b, G, q_in_group, T_k, d)
                Kg = Kg.expand(b_size, self.kv_groups, self.q_in_group, T_k, self.head_dim)
                Vg = Vg.expand(b_size, self.kv_groups, self.q_in_group, T_k, self.head_dim)

                # Merge groups and heads back: (b, H, T_q, d), (b, H, T_k, d)
                Q_flat = Qg.contiguous().view(b_size, self.heads_num, T_q, self.head_dim)
                K_flat = Kg.contiguous().view(b_size, self.heads_num, T_k, self.head_dim)
                V_flat = Vg.contiguous().view(b_size, self.heads_num, T_k, self.head_dim)

                # Prefill: full causal attention over the prompt
                att = F.scaled_dot_product_attention(
                    Q_flat,      # (b, H, T_q, d)
                    K_flat,      # (b, H, T_k, d)
                    V_flat,      # (b, H, T_k, d)
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True
                )

            else:
                # ---------- INCREMENTAL DECODE: SUBSEQUENT CACHED CALLS ----------
                # After prefill, we assume one-token-at-a-time decoding for strict causality.
                assert seq_len == 1, \
                    "After prefill, cache mode expects seq_len == 1 for incremental decoding"

                # Store new token(s) into ring buffer along time dim=2
                insert_len = min(seq_len, self.window_size)
                end = self.cache_ptr + insert_len
                # Use only the last insert_len timesteps from K_new/V_new
                K_slice = K_new[:, :, -insert_len:, :]  # (b, G, insert_len, d)
                V_slice = V_new[:, :, -insert_len:, :]

                if end <= self.window_size:
                    # Straight write
                    self.cache_k[:, :, self.cache_ptr:end, :] = K_slice
                    self.cache_v[:, :, self.cache_ptr:end, :] = V_slice
                else:
                    # Wrap-around case: split write
                    first = self.window_size - self.cache_ptr
                    self.cache_k[:, :, self.cache_ptr:, :] = K_slice[:, :, :first, :]
                    self.cache_k[:, :, :end - self.window_size, :] = K_slice[:, :, first:, :]

                    self.cache_v[:, :, self.cache_ptr:, :] = V_slice[:, :, :first, :]
                    self.cache_v[:, :, :end - self.window_size, :] = V_slice[:, :, first:, :]

                self.cache_ptr = (self.cache_ptr + insert_len) % self.window_size
                self.cache_len = min(self.cache_len + insert_len, self.window_size)

                # Reconstruct ordered K/V: (b, G, T_k, d_head)
                if self.cache_len < self.window_size:
                    K = self.cache_k[:, :, :self.cache_len, :]
                    V = self.cache_v[:, :, :self.cache_len, :]
                else:
                    K = torch.cat(
                        (self.cache_k[:, :, self.cache_ptr:, :],
                         self.cache_k[:, :, :self.cache_ptr, :]),
                        dim=2
                    )
                    V = torch.cat(
                        (self.cache_v[:, :, self.cache_ptr:, :],
                         self.cache_v[:, :, :self.cache_ptr, :]),
                        dim=2
                    )
                # K, V: (b, G, T_k, d_head)

                T_q = seq_len              # 1
                T_k = K.size(2)            # cache_len

                # ---- expand K/V per query head within each group ----
                # Q: (b, H, T_q, d), we view it as (b, G, H/G, T_q, d)
                # Use reshape here to be safe with non-contiguous Q
                Qg = Q.reshape(b_size, self.kv_groups, self.q_in_group, T_q, self.head_dim)
                # K, V: (b, G, T_k, d) -> (b, G, 1, T_k, d) then broadcast over q_in_group
                Kg = K.unsqueeze(2)  # (b, G, 1, T_k, d)
                Vg = V.unsqueeze(2)  # (b, G, 1, T_k, d)

                # Broadcast to (b, G, q_in_group, T_k, d)
                Kg = Kg.expand(b_size, self.kv_groups, self.q_in_group, T_k, self.head_dim)
                Vg = Vg.expand(b_size, self.kv_groups, self.q_in_group, T_k, self.head_dim)

                # Merge groups and heads back: (b, H, T_q, d), (b, H, T_k, d)
                Q_flat = Qg.contiguous().view(b_size, self.heads_num, T_q, self.head_dim)
                K_flat = Kg.contiguous().view(b_size, self.heads_num, T_k, self.head_dim)
                V_flat = Vg.contiguous().view(b_size, self.heads_num, T_k, self.head_dim)

                # FlashAttention path (no explicit causal mask; cache only holds past + current token)
                att = F.scaled_dot_product_attention(
                    Q_flat,      # (b, H, T_q, d)
                    K_flat,      # (b, H, T_k, d)
                    V_flat,      # (b, H, T_k, d)
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False
                )

        else:
            # ================= NORMAL TRAINING MODE =================
            # Full sequence attention with causal masking handled internally

            # Here we don't need to actually store cache; just use K_new, V_new.
            # K_new, V_new: (b, G, T, d)
            # Expand them per query head similarly as above:

            # Use reshape here as well to avoid view-on-transposed issues
            Qg = Q.reshape(b_size, self.kv_groups, self.q_in_group, seq_len, self.head_dim)
            Kg = K_new.unsqueeze(2)  # (b, G, 1, T, d)
            Vg = V_new.unsqueeze(2)  # (b, G, 1, T, d)

            Kg = Kg.expand(b_size, self.kv_groups, self.q_in_group, seq_len, self.head_dim)
            Vg = Vg.expand(b_size, self.kv_groups, self.q_in_group, seq_len, self.head_dim)

            Q_flat = Qg.contiguous().view(b_size, self.heads_num, seq_len, self.head_dim)
            K_flat = Kg.contiguous().view(b_size, self.heads_num, seq_len, self.head_dim)
            V_flat = Vg.contiguous().view(b_size, self.heads_num, seq_len, self.head_dim)

            attn_mask = self.sliding_window_mask(seq_len, device=Q_flat.device) if use_swa else None

            att = F.scaled_dot_product_attention(
                Q_flat,      # (b, H, T, d)
                K_flat,      # (b, H, T, d)
                V_flat,      # (b, H, T, d)
                attn_mask=attn_mask,
                dropout_p=self.drop_rate if self.training else 0.0,
                is_causal=False if use_swa else True
            )

        # Merge heads: (b, H, T, d) → (b, T, H*d) = (b, T, model_d)
        out = att.transpose(1, 2).reshape(b_size, seq_len, self.model_dim)
        return self.final_proj(out)