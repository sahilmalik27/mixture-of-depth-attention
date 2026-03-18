#!/usr/bin/env python3
"""Benchmark: Standard attention vs MoDA naive vs MoDA Triton.

Measures wall-clock time per forward pass and reports tok/s.
Uses the same dimensions as the A/B experiment (125M GPT config).

Usage:
    TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas python benchmarks/bench_attention.py
"""

import os
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moda.kernels.moda_naive import moda_attention_naive
from moda.kernels.moda_triton import moda_attention_triton


def benchmark_fn(fn, warmup=10, repeat=50):
    """Benchmark a function, return median time in seconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times.sort()
    return times[len(times) // 2]


def bench_standard_attention(Q, K, V, G):
    """Standard causal attention via SDPA, including GQA expansion."""
    K_exp = K if G == 1 else K.repeat_interleave(G, dim=1)
    V_exp = V if G == 1 else V.repeat_interleave(G, dim=1)
    return F.scaled_dot_product_attention(Q, K_exp, V_exp, is_causal=True)


def bench_moda_naive(Q, K, V, K_depth, V_depth, L):
    return moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L)


def bench_moda_triton(Q, K, V, K_depth, V_depth, L):
    return moda_attention_triton(Q, K, V, K_depth, V_depth, num_layers=L)


def main():
    device = "cuda"
    dtype = torch.float32

    B = 4
    H_q = 12
    H_k = 4
    d = 64
    L = 12
    G = H_q // H_k

    seq_lengths = [128, 256, 512, 1024, 2048]

    print(f"{'':=<90}")
    print(f"MoDA Attention Benchmark")
    print(f"Config: B={B}, H_q={H_q}, H_k={H_k}, d={d}, L={L}, GQA groups={G}, dtype={dtype}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Standard attention includes GQA expansion (fair comparison)")
    print(f"{'':=<90}")
    print()
    print(f"{'T':>6}  {'Standard':>12}  {'MoDA Naive':>12}  {'MoDA Triton':>12}  "
          f"{'Naive/Std':>10}  {'Triton/Std':>10}  {'Triton':>8}")
    print(f"{'':>6}  {'(tok/s)':>12}  {'(tok/s)':>12}  {'(tok/s)':>12}  "
          f"{'(ratio)':>10}  {'(ratio)':>10}  {'speedup':>8}")
    print(f"{'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}  "
          f"{'-'*10}  {'-'*10}  {'-'*8}")

    for T in seq_lengths:
        total_tokens = B * T

        Q = torch.randn(B, H_q, T, d, device=device, dtype=dtype)
        K = torch.randn(B, H_k, T, d, device=device, dtype=dtype)
        V = torch.randn(B, H_k, T, d, device=device, dtype=dtype)
        K_depth = torch.randn(B, H_k, T * L, d, device=device, dtype=dtype)
        V_depth = torch.randn(B, H_k, T * L, d, device=device, dtype=dtype)

        # Standard attention (includes GQA expansion)
        t_std = benchmark_fn(lambda: bench_standard_attention(Q, K, V, G))
        tps_std = total_tokens / t_std

        # MoDA naive
        t_naive = benchmark_fn(lambda: bench_moda_naive(Q, K, V, K_depth, V_depth, L))
        tps_naive = total_tokens / t_naive

        # MoDA Triton
        t_triton = benchmark_fn(lambda: bench_moda_triton(Q, K, V, K_depth, V_depth, L))
        tps_triton = total_tokens / t_triton

        ratio_naive = t_naive / t_std
        ratio_triton = t_triton / t_std
        speedup = t_naive / t_triton

        print(f"{T:>6}  {tps_std:>12,.0f}  {tps_naive:>12,.0f}  {tps_triton:>12,.0f}  "
              f"{ratio_naive:>10.2f}x  {ratio_triton:>10.2f}x  {speedup:>7.1f}x")

        # Verify correctness
        with torch.no_grad():
            out_naive = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L)
            out_triton = moda_attention_triton(Q, K, V, K_depth, V_depth, num_layers=L)
            diff = (out_naive - out_triton).abs().max().item()
            if diff > 1e-3:
                print(f"  WARNING: max diff = {diff:.6f} (> 1e-3)")

        del Q, K, V, K_depth, V_depth
        torch.cuda.empty_cache()

    print()
    print("Naive/Std & Triton/Std = time relative to standard (lower is better)")
    print("Triton speedup = how much faster Triton is vs naive MoDA")


if __name__ == "__main__":
    main()
