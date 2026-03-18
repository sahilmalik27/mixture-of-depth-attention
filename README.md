# MoDA — Mixture-of-Depths Attention

A pip-installable PyTorch library implementing **Mixture-of-Depths Attention** from [arXiv:2603.15619](https://arxiv.org/abs/2603.15619).

MoDA lets each attention head attend to both **sequence KV** (standard causal attention) and **depth KV** (representations from all preceding layers at the same token position), fused into a **single softmax** so they naturally compete for attention mass.

## Install

```bash
pip install -e .

# With Triton kernel support (CUDA required):
pip install -e ".[triton]"
```

## Quick Start

```python
import torch
from moda import MoDAConfig, MoDAModel

config = MoDAConfig(
    d_model=512,
    num_heads=8,
    num_kv_heads=4,     # GQA: 2 query heads per KV head
    num_layers=6,
    chunk_size=64,       # chunk-aware depth optimization
)

model = MoDAModel(config)
x = torch.randn(2, 128, config.d_model)
out = model(x)  # [2, 128, 512]
```

### Using MoDAAttention Standalone

```python
from moda import MoDAConfig, MoDAAttention

config = MoDAConfig(d_model=256, num_heads=4, num_kv_heads=2, num_layers=8)
attn = MoDAAttention(config)

x = torch.randn(1, 32, 256)
K_depth = torch.randn(1, 2, 32 * 4, 32)  # depth KV from 4 preceding layers
V_depth = torch.randn(1, 2, 32 * 4, 32)

output, k_write, v_write = attn(x, K_depth, V_depth)
```

### Using the Kernel Directly

```python
from moda.kernels import moda_attention_naive

Q = torch.randn(1, 4, 32, 64)       # [B, H_q, T, d]
K = torch.randn(1, 2, 32, 64)       # [B, H_k, T, d]
V = torch.randn(1, 2, 32, 64)
K_depth = torch.randn(1, 2, 32*8, 64)  # [B, H_k, T*L, d]
V_depth = torch.randn(1, 2, 32*8, 64)

out = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=8)
```

## Architecture

- **`MoDAConfig`** — Dataclass holding all hyperparameters
- **`DepthKVCache`** — Manages contiguous depth KV storage across layers
- **`MoDAAttention`** — Drop-in attention module with fused seq + depth softmax
- **`MoDATransformerBlock`** — Attention + FFN, both writing to depth cache
- **`MoDAModel`** — Full transformer stack with depth cache management

### Kernels

- `moda_attention_naive` — Reference PyTorch implementation (correct, not fast)
- `moda_attention_triton` — Fused Triton kernel with online softmax (requires CUDA + Triton)

## Key Design Choices

- **Combined softmax**: Sequence and depth attention scores are concatenated and softmaxed together, so they compete for attention mass
- **Chunk-aware optimization**: Queries split into chunks of size C; each chunk accesses only C×L depth KV entries
- **GQA support**: G query heads share each KV head; `repeat_interleave` handles expansion
- **Post-norm default**: Paper finds post-norm outperforms pre-norm with MoDA
- **FFN writes to depth cache**: Both attention and FFN layers produce depth KV entries

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0

## Citation

```bibtex
@article{zhu2026moda,
  title={Mixture-of-Depths Attention},
  author={Zhu, Yongqi and others},
  journal={arXiv preprint arXiv:2603.15619},
  year={2026}
}
```
