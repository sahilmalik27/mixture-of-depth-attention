"""MoDAAttention module — drop-in attention replacement with depth fusion."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .config import MoDAConfig
from .kernels import moda_attention_naive
from .kernels import HAS_TRITON

if HAS_TRITON:
    from .kernels import moda_attention_triton


class MoDAAttention(nn.Module):
    """Mixture-of-Depths Attention.

    Drop-in replacement for standard multi-head attention that additionally
    attends to depth KV (representations from preceding layers at the same
    token position), fused into a single softmax with sequence attention.

    Args:
        config: MoDAConfig instance.
    """

    def __init__(self, config: MoDAConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_layers = config.num_layers
        self.scale = config.head_dim ** -0.5

        # Query projection: d_model -> num_heads * head_dim
        self.q_proj = nn.Linear(config.d_model, config.num_heads * config.head_dim, bias=False)
        # KV projections: d_model -> num_kv_heads * head_dim
        self.k_proj = nn.Linear(config.d_model, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.num_kv_heads * config.head_dim, bias=False)
        # Output projection
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.d_model, bias=False)

        # Depth KV write projections: d_model -> num_kv_heads * head_dim
        self.k_depth_proj = nn.Linear(config.d_model, config.num_kv_heads * config.head_dim, bias=False)
        self.v_depth_proj = nn.Linear(config.d_model, config.num_kv_heads * config.head_dim, bias=False)

        self.use_triton = config.use_triton and HAS_TRITON
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: Tensor,
        K_depth: Optional[Tensor] = None,
        V_depth: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [B, T, D].
            K_depth: Depth key cache [B, H_k, T*L', d] from preceding layers.
                If None, only sequence attention is performed.
            V_depth: Depth value cache [B, H_k, T*L', d].

        Returns:
            Tuple of:
                - output: [B, T, D] attention output.
                - k_write: [B, H_k, T, d] depth keys to write to cache.
                - v_write: [B, H_k, T, d] depth values to write to cache.
        """
        B, T, D = x.shape
        H_q = self.num_heads
        H_k = self.num_kv_heads
        d = self.head_dim

        # Project Q, K, V
        Q = self.q_proj(x).view(B, T, H_q, d).transpose(1, 2)  # [B, H_q, T, d]
        K = self.k_proj(x).view(B, T, H_k, d).transpose(1, 2)  # [B, H_k, T, d]
        V = self.v_proj(x).view(B, T, H_k, d).transpose(1, 2)  # [B, H_k, T, d]

        # Depth KV write projections
        k_write = self.k_depth_proj(x).view(B, T, H_k, d).transpose(1, 2)  # [B, H_k, T, d]
        v_write = self.v_depth_proj(x).view(B, T, H_k, d).transpose(1, 2)  # [B, H_k, T, d]

        if K_depth is not None and V_depth is not None and K_depth.shape[2] > 0:
            # MoDA: fused sequence + depth attention
            attn_fn = moda_attention_triton if self.use_triton else moda_attention_naive
            out = attn_fn(
                Q, K, V, K_depth, V_depth,
                num_layers=self.num_layers,
                scale=self.scale,
                chunk_size=self.config.chunk_size,
            )
        else:
            # Standard causal attention (no depth cache available yet)
            out = self._standard_causal_attention(Q, K, V)

        out = out.transpose(1, 2).contiguous().view(B, T, H_q * d)
        out = self.o_proj(out)
        out = self.dropout(out)

        return out, k_write, v_write

    def _standard_causal_attention(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        """Standard causal multi-head attention (no depth). Uses Flash Attention via SDPA."""
        H_q = Q.shape[1]
        H_k = K.shape[1]
        G = H_q // H_k

        if G > 1:
            K = K.repeat_interleave(G, dim=1)
            V = V.repeat_interleave(G, dim=1)

        return torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, is_causal=True, scale=self.scale,
        )
