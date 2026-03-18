"""MoDA kernel implementations."""

from .moda_naive import moda_attention_naive

__all__ = ["moda_attention_naive"]

try:
    from .moda_triton import moda_attention_triton
    __all__.append("moda_attention_triton")
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
