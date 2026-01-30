"""
Transformer Component Utilities
-------------------------------
This module provides optimized normalization and activation layers used to 
stabilize and enhance the model's representative power. 

Key Components:
- NormLayer: Standard Layer Normalization for baseline architectural stability.
- RMSNorm: A lightweight, Llama-style alternative that scales based on root mean square.
- SwiGLU: A gated activation function used in modern SOTA models to improve learning capacity.
- SwiGLU_FFN: The "Knowledge Center" that uses gated logic to process information and expand the model's thinking capacity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormLayer(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        # These are learnable scales and shifts so the model can 
        # undo the normalization if it actually needs a different distribution.
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.bias = nn.Parameter(torch.zeros(emb_dim))
        self.eps = eps

    def forward(self, x):
        # Traditional LayerNorm: we zero-center the data and scale it by variance.
        # This keeps gradients from exploding or dying during deep training.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * norm_x + self.bias


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(emb_dim))
        self.eps = eps

    def forward(self, x):
        # Llama-style normalization. It's faster because we skip the mean 
        # centering and only focus on the Root Mean Square.
        # It's basically saying: "just keep the scale consistent."
        norm = x.norm(2, dim=-1, keepdim=True)  # L2 norm
        rms = norm * (1.0 / x.size(-1))**0.5
        return x / (rms + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int):
        """
        SwiGLU is basically a 'gated' linear unit. 
        One side (W2) acts as a gate that decides what information 
        from the other side (W1) gets to pass through.
        """
        super().__init__()
        self.W1 = nn.Linear(emb_dim, hidden_dim)
        self.W2 = nn.Linear(emb_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We split the input into two paths. 
        # v goes through SiLU (Swish) to create a non-linear mask.
        u = self.W1(x)
        v = self.W2(x)
        # Element-wise multiplication: u is the signal, SiLU(v) is the gate.
        return u * F.silu(v)
  
    
class FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
    
    
class SwiGLU_FFN(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int | None = None):
        """
        The Feed-Forward Network (FFN) is the 'knowledge' center of the model. 
        We use SwiGLU here because it's more stable and expressive than standard ReLU,
        which is why modern models like LLaMA and PaLM use it.
        """
        super().__init__()
        
        # If we don't specify a hidden dimension, we follow the 'magic' 8/3 rule.
        if hidden_dim is None:
            hidden_dim = int(8 * emb_dim / 3)

        # SwiGLU handles the non-linear transformation
        self.swiGLU = SwiGLU(emb_dim, hidden_dim)
        
        # W_out maps the processed data back to our model's original embedding size
        self.W_out = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First, expand and activate the features using SwiGLU
        x = self.swiGLU(x)
        
        # Then, project them back down to the model dimension
        return self.W_out(x)