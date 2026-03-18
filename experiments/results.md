# A/B Experiment: MoDA vs Standard Attention

## Setup

Two identical 125M parameter GPT models trained on the same data, same hyperparameters — one with standard causal attention (baseline), one with MoDA (depth attention from all preceding layers).

| Config | Value |
|--------|-------|
| Parameters | ~125M (12 layers, dim 768, 12 heads, 4 KV heads) |
| Architecture | Llama-style (GQA, SwiGLU, RoPE, RMSNorm) |
| Vocab size | 8,192 (BPE) |
| Sequence length | 2,048 |
| Batch size | 32 effective (8 × 4 grad accum) |
| Learning rate | 6e-4 (cosine decay, 200 step warmup) |
| Precision | bf16 mixed |
| Data | ClimbMix-400B (web + book + code) |
| Hardware | Single NVIDIA GB10 GPU |
| MoDA config | chunk_size=64, post_norm=True |

## Results

### Training Progress

**Baseline** ran for 1,196 steps (~4 hours, 5,440 tok/s).
**MoDA** ran for 110 steps (~3.3 hours, 598 tok/s).

### Head-to-Head at Step 100

| Metric | Baseline | MoDA | Δ |
|--------|----------|------|---|
| Val BPB | 8.598 | **8.579** | **-0.018** |
| Val Loss | 5.960 | 5.947 | -0.013 |

MoDA shows a marginal quality advantage per step — the model benefits from attending to earlier layer representations even at step 100.

### Best Achieved (full run)

| Metric | Baseline | MoDA |
|--------|----------|------|
| Steps | 1,196 | 110 |
| Best val BPB | **5.593** | 8.579 |
| Final train BPB | 5.672 | 8.383 |
| Throughput | **5,440 tok/s** | 598 tok/s |

### Validation BPB Curves

**Baseline:**
| Step | Val BPB |
|------|---------|
| 100 | 8.598 |
| 200 | 7.972 |
| 300 | 7.402 |
| 400 | 6.972 |
| 500 | 6.815 |
| 600 | 6.555 |
| 700 | 6.367 |
| 800 | 6.200 |
| 900 | 6.027 |
| 1000 | 5.883 |
| 1100 | 5.761 |
| 1196 | 5.593 |

**MoDA:**
| Step | Val BPB |
|------|---------|
| 50 | 9.783 |
| 100 | 8.579 |

## Analysis

### Per-Step Quality
MoDA shows a slight quality advantage at equal step counts (-0.018 BPB at step 100). This is consistent with the paper's claim that depth attention preserves signal from earlier layers. However, the signal is small at this early stage — the paper's +2.11% downstream improvement was measured at 100B tokens on a 1.5B model, far beyond our 110-step run.

### Throughput Gap
The critical finding: **MoDA with the naive PyTorch kernel runs 9.1x slower** than standard attention. The paper claims only 3.7% FLOPs overhead, but this assumes a fused Triton kernel with:

1. **Online softmax** — our naive impl materializes the full combined score matrix
2. **Chunk-aware memory access** — naive impl doesn't exploit locality
3. **No intermediate tensors** — naive impl creates multiple temporary tensors per layer

The throughput gap means that at equal wall-clock time, baseline trains ~10x more steps and reaches much lower BPB.

### Wall-Clock Efficiency
At 4 hours:
- Baseline: step 1,196, val BPB **5.593**
- MoDA: step ~110, val BPB **8.579**

Baseline wins decisively on wall-clock efficiency with the naive kernel.

### What This Means

1. **MoDA's quality claim has weak supporting evidence** in our experiment — a -0.018 BPB improvement at step 100 is directionally positive but not statistically significant.

2. **The fused Triton kernel is critical** for MoDA to be practical. Without it, the depth KV cache overhead dominates training time.

3. **Longer runs needed** to see if MoDA's per-step advantage compounds. The paper trained for 100B tokens; we trained for ~7M tokens with MoDA.

## Reproducing

```bash
# Install dependencies
pip install moda-attention torch

# Run baseline (4 hours)
python experiments/train_ab.py --mode baseline --max-hours 4

# Run MoDA (4 hours)
python experiments/train_ab.py --mode moda --max-hours 4

# Generate comparison
python experiments/compare.py
```

## Next Steps

- Benchmark the fused Triton kernel to close the throughput gap
- Run longer MoDA training (1000+ steps) for meaningful BPB comparison
- Profile memory: depth KV cache scales as O(T × L × d) per token
- Test with larger models where depth signal degradation is more pronounced
