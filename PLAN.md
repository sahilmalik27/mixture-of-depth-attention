# MoDA Library — Build Plan

## Source Paper
**Mixture-of-Depths Attention** (arXiv:2603.15619)
ByteDance Seed + HUST — Zhu et al., March 2026

## What We're Building
A standalone pip-installable Python library that provides MoDA as a drop-in attention replacement for any PyTorch transformer.

## Core Idea
MoDA lets each attention head attend to BOTH:
1. **Sequence KV** — standard causal attention within current layer
2. **Depth KV** — KV pairs from ALL preceding layers at the same token position

These are fused into a single softmax (combined softmax), so sequence and depth compete for attention mass naturally.

## Architecture

### Module: `MoDAAttention`
Drop-in replacement for `nn.MultiheadAttention` or any standard attention layer.

**Inputs:**
- `Q ∈ R^{T_q × (H_k × d)}` — queries from current layer
- `K, V ∈ R^{T_kv × (H_k × d)}` — sequence keys/values (current layer)
- `K_depth, V_depth ∈ R^{(T_kv × L) × (H_k × d)}` — depth KV cache (all preceding layers)
- `G` — GQA group number

**Output:**
- `O ∈ R^{T_q × (H_k × d)}` — attention output (normalized over combined softmax)

### Module: `MoDATransformerBlock`
Wraps MoDAAttention + FFN + depth KV write logic.
- After attention: projects output to depth K, V via lightweight projections `W_K^W, W_V^W`
- Appends to depth KV cache for subsequent layers
- FFN layers also write to depth cache via KV projection

### Module: `MoDAModel`
Full transformer stack with MoDA. Manages depth KV cache across layers.

## Algorithm (from paper Algorithm 1)

For each query block b_q:
1. Load Q[b_q] to SRAM
2. Initialize online softmax states: m=-∞, acc=0, o=0
3. Compute base_time for each query: t_base(i_q) = floor(i_q / G)
4. **Sequence phase (causal):** For each sequence key block:
   - If fully before t_base_start: full block attention (no masking)
   - If overlapping t_base range: apply grouped causal mask (floor(i_q/G) >= i_k)
   - Update online softmax: m', acc', o'
5. **Depth phase:** For each depth block in [t_base_start*L, t_base_end*L):
   - Apply mask: 1[floor(i_q/G) == floor(j_d/L)]
   - Update online softmax with depth scores
6. Normalize: o = o / acc
7. Store output

## Key Optimizations

### Chunk-Aware Depth KV Layout
- Queries divided into chunks of size C
- Each chunk accesses only C×L depth KV (not T×L)
- Depth utilization: 1/C instead of 1/T

### Group-Aware Indexing (GQA)
- G adjacent query rows share same base_time
- Reduces effective depth span to (C×L)/G per chunk
- Reuses depth KV blocks across GQA group

## File Structure

```
p2p-moda/
├── moda/
│   ├── __init__.py          # Public API
│   ├── attention.py         # MoDAAttention module (PyTorch)
│   ├── model.py             # MoDATransformerBlock, MoDAModel
│   ├── cache.py             # DepthKVCache management
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── moda_triton.py   # Triton kernel (fused forward)
│   │   └── moda_naive.py    # Reference PyTorch impl (for testing)
│   └── config.py            # MoDAConfig dataclass
├── tests/
│   ├── test_attention.py    # Unit tests for MoDAAttention
│   ├── test_cache.py        # Depth KV cache tests
│   ├── test_kernel.py       # Triton vs naive correctness
│   ├── test_model.py        # Full model forward/backward
│   └── test_gqa.py          # GQA compatibility tests
├── benchmarks/
│   └── bench_attention.py   # Speed comparison vs FlashAttention-2
├── pyproject.toml
├── README.md
├── LICENSE                  # Apache 2.0
└── PLAN.md
```

## Hyperparameters from Paper
- **Depth utilization**: 12.5% typical (L=64 layers / G=8 groups)
- **Chunk size C**: Tunable, controls depth utilization (smaller C = higher utilization but more overhead)
- **Post-norm preferred**: MoDA + post-norm > MoDA + pre-norm
- **Extra FLOPs**: ~3.7% overhead vs standard attention
- **Depth KV projections**: `W_K^W, W_V^W ∈ R^{D × D/G}` — lightweight per-layer

## Constraints
- Must work with PyTorch >= 2.5
- Triton >= 3.0 for kernel
- Must support GQA (grouped query attention)
- Must handle causal masking correctly
- Backward pass needed for training
- No private info in repo (PUBLIC)

## Testing Strategy
1. Naive PyTorch reference implementation (slow but correct)
2. Triton kernel must match naive within tolerance
3. Test shapes: various T, L, G, d, H_k, H_q combinations
4. Test gradient flow (backward pass)
5. Test causal masking correctness
6. Benchmark against FlashAttention-2
