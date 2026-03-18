# Building MoDA: A Drop-In Attention Replacement That Lets Transformers See Their Own Depth

*March 2026*

Standard transformers have a blind spot: each layer can only see the current sequence, never what earlier layers computed. Information from shallow layers gets diluted through dozens of residual connections before reaching the final output. The deeper the model, the worse this gets.

**Mixture-of-Depths Attention (MoDA)** fixes this by letting each attention head attend to both the current sequence *and* representations from all preceding layers — fused into a single softmax. We built an open-source implementation from the paper ([arXiv:2603.15619](https://arxiv.org/abs/2603.15619)) and are releasing it as a drop-in PyTorch library.

## The Problem: Signal Degradation in Deep Networks

In a standard transformer, layer 24 has no direct access to what layer 1 computed. It only sees the residual stream, where early features have been mixed, overwritten, and diluted through 23 layers of updates. Skip connections help, but they're a blunt instrument — they carry everything forward equally, with no mechanism for selective retrieval.

This matters most for tasks that require combining low-level features (syntax, token identity) with high-level reasoning (semantics, planning). The model has to hope that important early signals survive the residual gauntlet.

## The Fix: Attend Across Depth

MoDA introduces a second attention pathway. At each layer, the model attends to:

1. **Sequence KV** — the standard causal attention over the current layer's tokens
2. **Depth KV** — key-value pairs from every preceding layer, stored per-token

Both are combined in a single softmax, so sequence and depth attention naturally compete for attention mass. The model learns when to look at the current layer's context versus when to reach back to an earlier layer's representation.

```
Standard Attention:    Q × K_seq^T → softmax → V_seq
MoDA Attention:        Q × [K_seq; K_depth]^T → softmax → [V_seq; V_depth]
```

The depth KV cache stores representations contiguously per token: for token `t` with `L` layers, depth entries occupy indices `[t*L, (t+1)*L)`. This layout enables efficient chunk-aware access patterns.

## Why This Isn't Just "More Attention"

The overhead is surprisingly small: **~3.7% extra FLOPs** over standard attention. Three design choices make this possible:

### 1. Chunk-Aware Access

Queries are split into chunks of size `C`. Each chunk only accesses `C × L` depth KV entries, not the full `T × L`. This reduces depth utilization from `O(T)` to `O(C)`, making the overhead constant regardless of sequence length.

### 2. GQA Integration

With Grouped Query Attention, `G` adjacent query heads share the same KV head. MoDA exploits this: `G` adjacent queries share the same base time position, reducing the effective depth span to `(C × L) / G` per chunk.

### 3. Fused Online Softmax

The sequence and depth attention phases share online softmax state `(m, acc, o)`. There's no separate softmax for depth — it's one unified operation, which keeps memory and compute tight.

## What We Built

The library provides five levels of abstraction:

```
MoDAModel          — Full transformer with depth cache management
MoDATransformerBlock — Single block (attention + FFN + depth write)
MoDAAttention      — Drop-in attention module
moda_attention_triton — Fused Triton kernel
moda_attention_naive  — Reference PyTorch implementation
```

### The Naive Kernel

We started with a pure PyTorch reference implementation. It's slow but readable and correct — every line maps directly to the paper's Algorithm 1:

```python
# Combined softmax over sequence + depth
seq_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
depth_scores = torch.matmul(Q, K_depth.transpose(-2, -1)) / scale

# Apply causal mask (sequence) and position mask (depth)
seq_scores.masked_fill_(causal_mask, float('-inf'))
depth_scores.masked_fill_(depth_mask, float('-inf'))

# Single softmax across both
all_scores = torch.cat([seq_scores, depth_scores], dim=-1)
weights = F.softmax(all_scores, dim=-1)

# Split and apply
seq_out = weights[..., :T] @ V
depth_out = weights[..., T:] @ V_depth
output = seq_out + depth_out
```

This also implements the chunk-aware variant, where queries are processed in chunks of size `C` and each chunk only accesses its corresponding `C × L` depth entries.

### The Triton Kernel

The fused Triton kernel processes query blocks with online softmax, updating running max/sum/output accumulators across both sequence and depth phases:

```python
# Sequence phase: iterate over key blocks with causal masking
for j in range(num_seq_blocks):
    # Load K block, compute scores, update online softmax
    ...

# Depth phase: iterate over matching depth blocks
for j in range(depth_start, depth_end):
    # Load depth K block, apply position mask, update same softmax state
    ...

# Final normalization
output = output / accumulator
```

Single kernel launch, no intermediate materialization, O(1) extra memory.

## Results from the Paper

On a 1.5B parameter OLMo2 model trained on 100B tokens:

| Metric | Standard Attention | MoDA | Improvement |
|--------|-------------------|------|-------------|
| Avg downstream accuracy (10 tasks) | Baseline | +2.11% | Significant |
| Avg perplexity (10 benchmarks) | Baseline | -0.2 | Better |
| Speed vs FlashAttention-2 (64K seq) | 100% | 97.3% | Negligible cost |
| Extra FLOPs | 0% | 3.7% | Minimal |

The improvements are consistent across tasks: PIQA, HellaSwag, ARC, MMLU, and others all see gains. The model is accessing useful information from earlier layers that was previously inaccessible.

## How to Use It

### Drop-in model replacement

```python
from moda import MoDAConfig, MoDAModel

config = MoDAConfig(
    d_model=2048,
    num_heads=16,
    num_kv_heads=4,
    num_layers=24,
    chunk_size=64,
    vocab_size=32000,
)

model = MoDAModel(config)
token_ids = torch.randint(0, 32000, (1, 512))
logits = model(token_ids)  # [1, 512, 32000]
```

### Standalone attention module

```python
from moda import MoDAConfig, MoDAAttention, DepthKVCache

config = MoDAConfig(d_model=512, num_heads=8, num_kv_heads=4, num_layers=12)
attn = MoDAAttention(config)
cache = DepthKVCache(config, batch_size=1, max_seq_len=256)

# In your forward loop, at each layer:
output, k_write, v_write = attn(
    x,
    K_depth=cache.get_depth_kv(layer_idx, seq_len)[0],
    V_depth=cache.get_depth_kv(layer_idx, seq_len)[1],
)
cache.write(layer_idx, k_write, v_write)
```

## Design Decisions

**Post-norm over pre-norm.** The paper finds that MoDA benefits from post-norm (LayerNorm after attention/FFN) rather than the more common pre-norm. This makes sense: post-norm normalizes the combined output of sequence + depth attention, giving the model a consistent scale to work with.

**FFN layers write to depth cache too.** Not just attention — the FFN output at each layer also gets projected into depth K, V and stored. This means later layers can attend to both the attention patterns *and* the feature transformations from earlier layers.

**Chunked depth access is critical.** Without chunking, depth attention scales as O(T × L), which dominates sequence attention for deep models. Chunking makes it O(C × L), independent of sequence length.

## What's Next

The paper shows results on language modeling, with vision experiments coming. The architecture is general — anywhere you stack attention layers, MoDA can replace standard attention. We're exploring integration with existing training pipelines and may run independent validation experiments.

The library is Apache 2.0 licensed and available at [github.com/sahilmalik27/mixture-of-depth-attention](https://github.com/sahilmalik27/mixture-of-depth-attention).

---

*Built from [arXiv:2603.15619](https://arxiv.org/abs/2603.15619) by Zhu et al. (ByteDance Seed + HUST).*
