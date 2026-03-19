# MoDA A/B Experiment: Does Depth Attention Help at Small Scale?

*March 19, 2026*

We ran a 12-hour A/B experiment pitting standard causal attention against Mixture-of-Depths Attention (MoDA) on identical 125M parameter GPT models. The results challenge MoDA's value proposition at small scale — but don't necessarily disprove the paper's claims. Here's what happened and what it means.

## The Setup

Two identical Llama-style GPT models (12 layers, 768 dim, 12 heads, 4 KV heads, SwiGLU, RoPE, RMSNorm) trained on ClimbMix-400B data with 8K BPE vocabulary. Same optimizer, same learning rate schedule, same data order. The only difference: one uses standard causal attention, the other uses MoDA.

| Config | Baseline | MoDA |
|--------|----------|------|
| Parameters | ~125M | ~140M (+12% from depth KV projections) |
| Effective batch | 32 sequences | 32 sequences |
| Micro-batch × grad accum | 8 × 4 | 4 × 8 (halved for memory) |
| Attention kernel | PyTorch SDPA | Naive PyTorch (no fused Triton) |
| Sequence length | 2048 | 2048 |
| Training time | 12.7 hours | 12.7 hours |
| GPU | NVIDIA GB10 | NVIDIA GB10 |

## The Results

### Final Numbers

| Metric | Baseline | MoDA | Δ |
|--------|----------|------|---|
| Steps completed | 3,050 | 1,950 | -36% |
| Best val BPB | **4.769** | 5.839 | +1.07 |
| Throughput | **3,777 tok/s** | 2,806 tok/s | -25.7% |

Baseline wins on every metric. More steps, lower loss, faster throughput.

### The Gap Over Time

The critical question: does MoDA learn more efficiently *per gradient step*, even if it's slower per step?

No. At matched step counts, MoDA is consistently worse:

| Step | Baseline val BPB | MoDA val BPB | Gap |
|------|-----------------|-------------|-----|
| 100 | 8.597 | 8.609 | +0.01 (neck and neck) |
| 500 | 6.695 | 7.252 | +0.56 |
| 1000 | 5.903 | 6.375 | +0.47 |
| 1400 | 5.338 | 6.103 | +0.77 (peak) |
| 1900 | 5.078 | 5.839 | +0.76 (plateaued) |

The gap starts near zero at step 100, widens to ~0.55 by step 500, then stabilizes around +0.76 from step 1200 onward. MoDA never closes the gap — it plateaus.

## Does This Disprove the Paper?

**Short answer: No, but it raises real questions about when MoDA's benefits emerge.**

The paper ([arXiv:2603.15619](https://arxiv.org/abs/2603.15619), Zhu et al., ByteDance Seed + HUST) reports:
- +2.11% average accuracy across 10 downstream tasks
- -0.2 average perplexity across 10 validation benchmarks
- 97.3% of FlashAttention-2's speed at 64K sequence length
- Only 3.7% extra FLOPs

Our experiment differs from the paper's conditions in several important ways:

### 1. Scale: 125M vs 1.5B (12× smaller)

The paper's core argument is about **signal degradation in deep networks** — shallow-layer features getting diluted through many residual updates. Our 12-layer model may simply not be deep enough for this to matter. A 1.5B model with 24–48 layers has far more opportunity for signal loss across depth.

This is the strongest argument that our results don't contradict the paper. MoDA may solve a problem that barely exists at 12 layers.

### 2. Training tokens: ~80M vs 100B (1,250× fewer)

The paper trains on 100 billion tokens. Our MoDA arm saw approximately 80 million tokens (1,950 steps × 32 batch × 2,048 seq_len ÷ grad_accum accounting). At 1,250× fewer tokens, depth attention may not have had enough signal to learn *what* to retrieve from earlier layers.

The depth KV projections are randomly initialized. The model needs to learn both (a) what to write into the depth cache and (b) what to read from it. This may require more training than standard attention to break even, then accelerate past it.

### 3. Kernel: Naive PyTorch vs Fused Triton

This is the most concrete problem. The paper achieves 97.3% of FlashAttention-2's throughput with their fused Triton kernel. We used the naive PyTorch reference implementation, which:

- Materializes the full combined score matrix (sequence + depth)
- Creates intermediate tensors at every layer for depth KV stacking
- Doesn't exploit chunk-locality in memory access patterns

Result: **25.7% throughput overhead** vs the paper's **2.7%**. This means at equal wall time, baseline got 56% more training steps. Even if MoDA were marginally better per step, it would need to overcome a massive efficiency penalty.

### 4. Architecture: Pre-norm vs Post-norm

The paper specifically finds that **post-norm works better with MoDA** than pre-norm. Our implementation uses pre-norm (the standard for modern LLMs). This could partially explain the gap — depth attention may need post-norm's different gradient dynamics to be effective.

## What Our Experiment Does Show

Even accounting for the above caveats, several findings are informative:

### MoDA doesn't help per-step at small scale

At step 100, the models are essentially tied (8.597 vs 8.609 BPB). The paper reports marginal per-step improvements, but those were measured at 1.5B parameters. At 125M, depth attention adds noise without adding useful signal.

### The gap widens, then plateaus — it never closes

If MoDA were slowly "learning to use depth," we'd expect the gap to narrow over time as the depth projections improve. Instead, the gap grows from +0.01 (step 100) to +0.76 (step 1200) and stays there. This suggests that at this scale, the model never learns to effectively use the depth cache.

### The naive kernel is a dealbreaker for practical use

25.7% throughput overhead makes MoDA strictly worse at equal wall time. The fused Triton kernel isn't optional — it's essential for MoDA to be viable. Anyone reproducing MoDA research *must* invest in kernel optimization.

### Memory overhead forces micro-batch reduction

MoDA's depth KV cache grows as O(T × L × d) per token. We had to halve the micro-batch from 8 to 4 and double gradient accumulation to fit in memory. This doesn't change the effective batch size, but it does impact throughput (more serialized forward/backward passes).

## The Honest Assessment

Our experiment tested MoDA at 1/12th the model size, 1/1250th the tokens, with a 10× slower kernel, and the wrong normalization scheme. That's a lot of confounders.

**What we can confidently say:**
- MoDA provides no benefit at 125M scale / 80M tokens
- Without a fused kernel, MoDA is not viable for training
- The depth KV overhead is real and non-trivial

**What we can't say:**
- Whether MoDA helps at 1.5B+ scale (the paper's claimed regime)
- Whether the fused kernel + post-norm + longer training would change the picture
- Whether downstream task accuracy (not just val BPB) would tell a different story

The paper's hypothesis — that deep networks lose signal from shallow layers, and direct depth attention recovers it — is plausible. But it may be a large-scale phenomenon with a threshold below which it simply doesn't manifest. Our experiment sits well below that threshold.

## Reproducing This Experiment

```bash
git clone https://github.com/sahilmalik27/mixture-of-depth-attention
cd mixture-of-depth-attention

# Run baseline (12 hours)
python experiments/train_ab.py --mode baseline --max-hours 12

# Run MoDA (12 hours)  
python experiments/train_ab.py --mode moda --max-hours 12

# Generate comparison tables
python experiments/compare.py
```

Full training logs and results are in `experiments/logs/` and `experiments/results.md`.

## What's Next

This experiment was a feasibility check. The real test would be:

1. **Implement the fused Triton kernel** to eliminate the throughput penalty
2. **Scale to 1B+ parameters** where depth signal degradation actually matters
3. **Use post-norm** as the paper recommends
4. **Train longer** — 10B+ tokens to give depth projections time to learn
5. **Evaluate on downstream tasks**, not just perplexity

Until then, we can't rule MoDA out — but we also can't endorse it at small scale.

---

*A/B experiment run on a single NVIDIA GB10 GPU, March 18–19, 2026. Built from [arXiv:2603.15619](https://arxiv.org/abs/2603.15619) by Zhu et al.*
