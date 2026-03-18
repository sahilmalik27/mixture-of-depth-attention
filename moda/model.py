"""MoDA transformer block and full model."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .config import MoDAConfig
from .attention import MoDAAttention
from .cache import DepthKVCache


class MoDATransformerBlock(nn.Module):
    """Single transformer block with MoDA attention.

    Contains MoDAAttention + FFN, with depth KV write from both.
    Supports pre-norm and post-norm configurations.
    """

    def __init__(self, config: MoDAConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn = MoDAAttention(config)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.d_model, bias=False),
        )

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        # FFN depth KV projections (FFN also writes to depth cache)
        self.ffn_k_depth_proj = nn.Linear(
            config.d_model, config.num_kv_heads * config.head_dim, bias=False
        )
        self.ffn_v_depth_proj = nn.Linear(
            config.d_model, config.num_kv_heads * config.head_dim, bias=False
        )

    def forward(
        self,
        x: Tensor,
        depth_cache: Optional[DepthKVCache] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: [B, T, D] input.
            depth_cache: Depth KV cache. Updated in-place.

        Returns:
            [B, T, D] output.
        """
        B, T, D = x.shape
        H_k = self.config.num_kv_heads
        d = self.config.head_dim

        # Read depth cache for this layer
        K_depth, V_depth = None, None
        if depth_cache is not None:
            K_depth, V_depth = depth_cache.read(seq_len=T)

        if self.config.post_norm:
            # Post-norm: attention -> add -> norm -> ffn -> add -> norm
            attn_out, k_write_attn, v_write_attn = self.attn(x, K_depth, V_depth)
            x = self.norm1(x + attn_out)

            # Write attention depth KV
            if depth_cache is not None:
                depth_cache.write(self.layer_idx * 2, k_write_attn, v_write_attn, seq_len=T)

            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)

            # Write FFN depth KV
            if depth_cache is not None:
                k_write_ffn = self.ffn_k_depth_proj(x).view(B, T, H_k, d).transpose(1, 2)
                v_write_ffn = self.ffn_v_depth_proj(x).view(B, T, H_k, d).transpose(1, 2)
                depth_cache.write(self.layer_idx * 2 + 1, k_write_ffn, v_write_ffn, seq_len=T)
        else:
            # Pre-norm: norm -> attention -> add -> norm -> ffn -> add
            normed = self.norm1(x)
            attn_out, k_write_attn, v_write_attn = self.attn(normed, K_depth, V_depth)
            x = x + attn_out

            if depth_cache is not None:
                depth_cache.write(self.layer_idx * 2, k_write_attn, v_write_attn, seq_len=T)

            normed = self.norm2(x)
            ffn_out = self.ffn(normed)
            x = x + ffn_out

            if depth_cache is not None:
                k_write_ffn = self.ffn_k_depth_proj(x).view(B, T, H_k, d).transpose(1, 2)
                v_write_ffn = self.ffn_v_depth_proj(x).view(B, T, H_k, d).transpose(1, 2)
                depth_cache.write(self.layer_idx * 2 + 1, k_write_ffn, v_write_ffn, seq_len=T)

        return x


class MoDAModel(nn.Module):
    """Full MoDA transformer model.

    Stacks MoDATransformerBlocks and manages the depth KV cache.
    """

    def __init__(self, config: MoDAConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList(
            [MoDATransformerBlock(config, layer_idx=i) for i in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        use_depth_cache: bool = True,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input token embeddings [B, T, D] or token ids [B, T].
            use_depth_cache: Whether to use depth KV cache.

        Returns:
            [B, T, D] output representations.
        """
        if x.dtype in (torch.long, torch.int):
            x = self.embed(x)

        B, T, D = x.shape

        depth_cache = None
        if use_depth_cache:
            # L in depth cache = num_layers * 2 (attn + FFN each write)
            depth_cache = DepthKVCache(
                batch_size=B,
                max_seq_len=T,
                num_layers=self.config.num_layers * 2,
                num_kv_heads=self.config.num_kv_heads,
                head_dim=self.config.head_dim,
                device=x.device,
                dtype=x.dtype,
            )

        for layer in self.layers:
            x = layer(x, depth_cache=depth_cache)

        x = self.final_norm(x)
        x = self.head(x)
        return x
