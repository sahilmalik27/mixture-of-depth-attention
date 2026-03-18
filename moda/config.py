"""MoDA configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MoDAConfig:
    """Configuration for MoDA (Mixture-of-Depths Attention).

    Attributes:
        d_model: Model dimension (D).
        num_heads: Number of query attention heads (H_q).
        num_kv_heads: Number of KV attention heads (H_k). Defaults to num_heads.
        head_dim: Per-head dimension (d). Defaults to d_model // num_heads.
        num_layers: Number of transformer layers (L).
        chunk_size: Query chunk size (C) for chunk-aware depth optimization.
            Controls depth utilization: each chunk of C queries only loads C*L
            depth KV entries instead of T*L. Defaults to 64.
        ffn_dim: FFN intermediate dimension. Defaults to 4 * d_model.
        dropout: Attention dropout probability.
        max_seq_len: Maximum sequence length.
        use_triton: Whether to use Triton kernels when available.
        post_norm: Use post-norm (True) vs pre-norm (False). Paper prefers post-norm.
    """

    d_model: int = 768
    num_heads: int = 12
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    num_layers: int = 12
    chunk_size: int = 64
    ffn_dim: Optional[int] = None
    dropout: float = 0.0
    max_seq_len: int = 2048
    use_triton: bool = False
    post_norm: bool = True

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.head_dim is None:
            self.head_dim = self.d_model // self.num_heads
        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.d_model

        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) must be divisible by "
            f"num_kv_heads ({self.num_kv_heads})"
        )

    @property
    def gqa_groups(self) -> int:
        """Number of query heads per KV head (G)."""
        return self.num_heads // self.num_kv_heads

    @property
    def kv_dim(self) -> int:
        """Total KV dimension: num_kv_heads * head_dim."""
        return self.num_kv_heads * self.head_dim
