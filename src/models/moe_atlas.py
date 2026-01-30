"""
Hierarchical Dynamic Mixture of Experts (HDynMoF)
-----------------------------------------------
This module implements an adaptive, sparse MoE layer. 
Unlike standard Top-K MoE, this version uses two-stage Top-p routing:
1. Group Routing: Selects relevant clusters of experts.
2. Expert Routing: Selects the specific specialists within those clusters.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import SwiGLU_FFN


class HDynMoF(nn.Module):
    """
    Adaptive hierarchical sparse MoE with:
      - Group-level top-p routing (adaptive #groups per token)
      - Expert-level top-p routing within each active group (adaptive #experts)
      - True sparse execution: each expert sees only its routed tokens.

    cfg keys:
      emb_dim              : int, embedding dim
      e_num                : int, total experts across all groups
      moe_groups           : int, number of groups
      moe_group_top_p      : float, group-level top-p threshold per token
      moe_max_groups       : int, max groups per token
      moe_top_p            : float, expert-level top-p threshold per token
      moe_max_k            : int, max experts per group per token
    """

    def __init__(self, cfg: dict):
        super().__init__()

        emb_dim = int(cfg["emb_dim"])

        # ---- Core MoE config ----
        self.e_num = int(cfg.get("e_num", 16))
        self.num_groups = int(cfg.get("moe_groups", 4))

        # Expert routing config
        self.top_p = float(cfg.get("moe_top_p", 0.9))
        self.max_k_per_grp = int(cfg.get("moe_max_k", 2))

        # Group routing config
        self.group_top_p = float(cfg.get("moe_group_top_p", 0.9))
        self.max_groups_per_token = int(cfg.get("moe_max_groups", self.num_groups))

        # ---- Sanity checks ----
        assert self.e_num >= 1, "e_num must be >= 1"
        assert self.num_groups >= 1, "moe_groups must be >= 1"
        assert 0.0 < self.top_p <= 1.0, "moe_top_p must be in (0, 1]"
        assert 0.0 < self.group_top_p <= 1.0, "moe_group_top_p must be in (0, 1]"
        assert self.e_num % self.num_groups == 0, "e_num must be divisible by moe_groups"

        self.exp_per_group = self.e_num // self.num_groups

        assert 1 <= self.max_k_per_grp <= self.exp_per_group, \
            f"moe_max_k must be in [1, {self.exp_per_group}], got {self.max_k_per_grp}"

        assert 1 <= self.max_groups_per_token <= self.num_groups, \
            f"moe_max_groups must be in [1, {self.num_groups}], got {self.max_groups_per_token}"

        # ---- Experts ----
        # experts[g][e] = expert e in group g
        self.experts = nn.ModuleList([
            nn.ModuleList([
                SwiGLU_FFN(emb_dim, 4 * emb_dim)
                for _ in range(self.exp_per_group)
            ])
            for _ in range(self.num_groups)
        ])

        # ---- Gating over experts (per group) ----
        self.gates = nn.ModuleList([
            nn.Linear(emb_dim, self.exp_per_group)
            for _ in range(self.num_groups)
        ])

        # ---- Group router ----
        self.group_router = nn.Linear(emb_dim, self.num_groups)

        # Variance scaling across groups
        self.group_scale = 1.0 / math.sqrt(self.num_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        returns: (B, T, D)
        """
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype

        # Flatten tokens: N = B * T
        x_flat = x.reshape(-1, D)              # (N, D)
        N = x_flat.size(0)

        # Output accumulator
        out_flat = x_flat.new_zeros((N, D))    # (N, D)

        # ============================================================
        # 1) Group-level routing (vectorized)
        # ============================================================
        # (B, T, G)
        group_logits = self.group_router(x)
        group_probs = F.softmax(group_logits, dim=-1)

        # sort along groups dim
        group_sorted_probs, group_sorted_idx = group_probs.sort(
            dim=-1, descending=True
        )  # (B, T, G)

        G = self.num_groups
        Kg = min(self.max_groups_per_token, G)

        # top-Kg candidates per token
        group_probs_k = group_sorted_probs[..., :Kg]        # (B, T, Kg)
        group_idx_k = group_sorted_idx[..., :Kg]            # (B, T, Kg)

        # cumulative prob for top-p
        group_cum = group_probs_k.cumsum(dim=-1)            # (B, T, Kg)
        group_active_mask = (group_cum <= self.group_top_p) # (B, T, Kg)
        group_active_mask[..., 0] = True                    # always keep best group

        # flatten to (N, Kg)
        group_probs_k_flat = group_probs_k.reshape(N, Kg)           # (N, Kg)
        group_idx_k_flat = group_idx_k.reshape(N, Kg)               # (N, Kg)
        group_active_flat = group_active_mask.reshape(N, Kg)        # (N, Kg)

        # masked + renormalize
        group_masked_probs = group_probs_k_flat * group_active_flat.to(dtype)
        group_sum_active = group_masked_probs.sum(dim=-1, keepdim=True) + 1e-9
        group_renorm_probs = group_masked_probs / group_sum_active             # (N, Kg)

        # ============================================================
        # 2) Group-wise expert routing (sparse, but GPU-friendly)
        # ============================================================
        # We keep the outer loop over groups (usually small: 4–8).
        for g in range(self.num_groups):
            gate_g = self.gates[g]
            experts_g = self.experts[g]

            # tokens & slots where group g is chosen
            # mask_group: (N, Kg)
            mask_group = (group_idx_k_flat == g) & group_active_flat
            if not mask_group.any():
                continue

            # indices of active (token, slot) for this group
            active_pos = mask_group.nonzero(as_tuple=False)    # (M_g, 2)
            if active_pos.numel() == 0:
                continue

            token_idx_g = active_pos[:, 0]          # (M_g,)
            group_slot_idx_g = active_pos[:, 1]     # (M_g,)

            # tokens that use this group (with repetition if same token picked
            # group g via multiple slots — rare, but we handle it)
            x_g_flat = x_flat[token_idx_g]          # (M_g, D)
            group_weight_g = group_renorm_probs[token_idx_g, group_slot_idx_g]  # (M_g,)

            M_g = x_g_flat.size(0)
            if M_g == 0:
                continue

            # --------------------------------------------------------
            # Expert-level routing for group g (vectorized over tokens)
            # --------------------------------------------------------
            # (M_g, E_g)
            gate_logits_g = gate_g(x_g_flat)
            probs_g = F.softmax(gate_logits_g, dim=-1)

            # sort experts by prob
            sorted_probs, sorted_idx = probs_g.sort(dim=-1, descending=True)  # (M_g, E_g)

            K = min(self.max_k_per_grp, self.exp_per_group)
            sorted_probs_k = sorted_probs[:, :K]      # (M_g, K)
            sorted_idx_k = sorted_idx[:, :K]          # (M_g, K)

            # top-p within group
            cum_probs = sorted_probs_k.cumsum(dim=-1)         # (M_g, K)
            active_mask = (cum_probs <= self.top_p)           # (M_g, K)
            active_mask[:, 0] = True                          # always keep best

            # renormalize only over active experts
            masked_probs = sorted_probs_k * active_mask.to(dtype)     # (M_g, K)
            sum_active = masked_probs.sum(dim=-1, keepdim=True) + 1e-9
            renorm_probs = masked_probs / sum_active                  # (M_g, K)

            # --------------------------------------------------------
            # Combine group + expert routing into a single mapping:
            #   for each active (token_local, expert_local)
            #   we know: global_token_idx, expert_local_idx, weight
            # --------------------------------------------------------
            active_exp_pos = active_mask.nonzero(as_tuple=False)          # (M_ge, 2)
            if active_exp_pos.numel() == 0:
                continue

            token_local_all = active_exp_pos[:, 0]    # (M_ge,)
            slot_all = active_exp_pos[:, 1]           # (M_ge,)

            # Map local token indices back to global [0, N)
            global_token_all = token_idx_g[token_local_all]               # (M_ge,)

            # Which local expert each (token, slot) chose
            expert_local_all = sorted_idx_k[token_local_all, slot_all]    # (M_ge,)

            # expert-level probs p_{g,e}(x)
            p_all = renorm_probs[token_local_all, slot_all]               # (M_ge,)
            # group-level probs q_g(x)
            q_all = group_weight_g[token_local_all]                       # (M_ge,)

            # Total mixture weights
            total_weight_all = (q_all * p_all).to(dtype)                  # (M_ge,)

            # --------------------------------------------------------
            # Sparse expert execution:
            #   we still loop over experts in this group,
            #   but everything else is precomputed and vectorized.
            # --------------------------------------------------------
            # To avoid repeated comparisons, we can pre-sort by expert id
            # and then process experts by slices.
            # (Optional but nice: reduces number of boolean masks.)
            sort_by_expert = torch.argsort(expert_local_all)
            expert_local_sorted = expert_local_all[sort_by_expert]
            global_token_sorted = global_token_all[sort_by_expert]
            weight_sorted = total_weight_all[sort_by_expert]

            # Find boundaries where expert id changes
            # unique_experts: (U,)
            # counts: (U,)
            unique_experts, counts = torch.unique_consecutive(
                expert_local_sorted, return_counts=True
            )

            # prefix sums for slicing
            offsets = counts.cumsum(dim=0)
            starts = torch.cat([
                offsets.new_zeros((1,)),
                offsets[:-1]
            ], dim=0)  # (U,)

            # iterate only over actually used experts (unique_experts)
            for idx_u, e_local in enumerate(unique_experts.tolist()):
                start = int(starts[idx_u].item())
                end = int(offsets[idx_u].item())
                if end <= start:
                    continue

                # slice for this expert
                token_slice = global_token_sorted[start:end]   # (M_e,)
                w_slice = weight_sorted[start:end]             # (M_e,)

                x_e = x_flat[token_slice]                      # (M_e, D)
                y_e = experts_g[e_local](x_e)                  # (M_e, D)

                w_e = (w_slice * self.group_scale).unsqueeze(-1)  # (M_e, 1)
                out_flat[token_slice] += w_e * y_e

        # back to (B, T, D)
        out = out_flat.reshape(B, T, D)
        return out