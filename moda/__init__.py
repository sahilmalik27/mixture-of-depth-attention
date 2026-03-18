"""MoDA: Mixture-of-Depths Attention.

A PyTorch library implementing MoDA from arXiv:2603.15619.
Fuses sequence attention and depth attention into a single softmax.
"""

from .config import MoDAConfig
from .cache import DepthKVCache
from .attention import MoDAAttention
from .model import MoDATransformerBlock, MoDAModel
from .kernels import moda_attention_naive

__all__ = [
    "MoDAConfig",
    "DepthKVCache",
    "MoDAAttention",
    "MoDATransformerBlock",
    "MoDAModel",
    "moda_attention_naive",
]

try:
    from .kernels import moda_attention_triton
    __all__.append("moda_attention_triton")
except ImportError:
    pass
