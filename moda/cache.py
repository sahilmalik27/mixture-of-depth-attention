"""Depth KV cache for MoDA."""

import torch
from torch import Tensor
from typing import Optional


class DepthKVCache:
    """Manages the depth KV cache across transformer layers.

    The depth cache stores K, V projections from each layer so that subsequent
    layers can attend to representations from all preceding layers at the same
    token position.

    Layout: For a sequence of T tokens and L layers, depth entries are stored
    contiguously per token: token t's depth states occupy indices [t*L, (t+1)*L).
    Within that range, index t*L + l holds layer l's contribution.

    Both attention layers and FFN layers write to this cache.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Total depth entries: max_seq_len * num_layers
        total_depth = max_seq_len * num_layers
        self.k_cache = torch.zeros(
            batch_size, num_kv_heads, total_depth, head_dim,
            device=device, dtype=dtype,
        )
        self.v_cache = torch.zeros(
            batch_size, num_kv_heads, total_depth, head_dim,
            device=device, dtype=dtype,
        )
        # Track how many layers have written for each position
        self._layer_count = 0
        self._seq_len = 0

    def write(
        self,
        layer_idx: int,
        k_depth: Tensor,
        v_depth: Tensor,
        seq_len: Optional[int] = None,
    ):
        """Write depth KV for a given layer.

        Args:
            layer_idx: Which layer is writing (0 to num_layers-1).
            k_depth: [B, H_k, T, d] — depth keys from this layer.
            v_depth: [B, H_k, T, d] — depth values from this layer.
            seq_len: Sequence length (defaults to k_depth.shape[2]).
        """
        T = seq_len or k_depth.shape[2]
        self._seq_len = max(self._seq_len, T)
        L = self.num_layers

        # For each token t, write at index t * L + layer_idx
        indices = torch.arange(T, device=k_depth.device) * L + layer_idx
        # indices: [T] -> expand for scatter: [1, 1, T, 1]
        idx = indices.view(1, 1, T, 1).expand_as(k_depth[:, :, :T])
        self.k_cache.scatter_(2, idx, k_depth[:, :, :T])
        self.v_cache.scatter_(2, idx, v_depth[:, :, :T])

    def read(self, seq_len: Optional[int] = None) -> tuple[Tensor, Tensor]:
        """Read depth KV cache for current sequence.

        Args:
            seq_len: Number of tokens to read depth cache for.

        Returns:
            (K_depth, V_depth) each of shape [B, H_k, T*L, d]
        """
        T = seq_len or self._seq_len
        L = self.num_layers
        total = T * L
        return self.k_cache[:, :, :total], self.v_cache[:, :, :total]

    def reset(self):
        """Clear the cache."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self._layer_count = 0
        self._seq_len = 0
