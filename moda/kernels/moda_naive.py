"""Naive (reference) PyTorch implementation of MoDA attention.

This is a straightforward, correct implementation used as ground truth
for testing the optimized Triton kernel. Not optimized for speed.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def moda_attention_naive(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    K_depth: Tensor,
    V_depth: Tensor,
    num_layers: int,
    scale: Optional[float] = None,
    chunk_size: Optional[int] = None,
) -> Tensor:
    """MoDA attention: fused sequence + depth attention with combined softmax.

    Implements Algorithm 1 from the paper. Sequence (causal) attention and
    depth attention scores are concatenated and passed through a single softmax,
    so they naturally compete for attention mass.

    Args:
        Q: Query tensor [B, H_q, T, d].
        K: Sequence key tensor [B, H_k, T, d].
        V: Sequence value tensor [B, H_k, T, d].
        K_depth: Depth key cache [B, H_k, T*L, d]. For token t, entries
            [t*L, (t+1)*L) hold depth keys from layers 0..L-1.
        V_depth: Depth value cache [B, H_k, T*L, d]. Same layout as K_depth.
        num_layers: Number of layers (L) in depth cache.
        scale: Attention scale factor. Defaults to 1/sqrt(d).
        chunk_size: Query chunk size (C) for chunk-aware optimization.
            If None, processes all queries at once (no chunking).

    Returns:
        Output tensor [B, H_q, T, d].
    """
    B, H_q, T, d = Q.shape
    _, H_k, T_kv, _ = K.shape
    G = H_q // H_k  # GQA group size

    if scale is None:
        scale = d ** -0.5

    # Expand KV heads to match query heads for GQA
    if G > 1:
        K = K.repeat_interleave(G, dim=1)       # [B, H_q, T_kv, d]
        V = V.repeat_interleave(G, dim=1)       # [B, H_q, T_kv, d]
        K_depth = K_depth.repeat_interleave(G, dim=1)
        V_depth = V_depth.repeat_interleave(G, dim=1)

    if chunk_size is not None:
        return _moda_chunked(Q, K, V, K_depth, V_depth, num_layers, scale, chunk_size)

    # --- Sequence attention scores ---
    # [B, H_q, T, T_kv]
    scores_seq = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # Causal mask: query at position i can attend to key at position j if i >= j
    causal_mask = torch.triu(
        torch.ones(T, T_kv, device=Q.device, dtype=torch.bool), diagonal=1
    )
    scores_seq = scores_seq.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # --- Depth attention scores ---
    # [B, H_q, T, T_kv * L]
    scores_depth = torch.matmul(Q, K_depth.transpose(-2, -1)) * scale

    # Depth mask: query at position i attends to depth entries [i*L, (i+1)*L)
    L = num_layers
    q_pos = torch.arange(T, device=Q.device)          # [T]
    d_pos = torch.arange(T_kv * L, device=Q.device)   # [T_kv * L]
    # Token position for each depth entry: floor(j / L)
    d_token = d_pos // L                               # [T_kv * L]
    # Mask: allow only where query position == depth token position
    depth_mask = q_pos.unsqueeze(1) != d_token.unsqueeze(0)  # [T, T_kv*L]
    scores_depth = scores_depth.masked_fill(depth_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # --- Combined softmax ---
    # Concatenate scores: [B, H_q, T, T_kv + T_kv*L]
    scores_combined = torch.cat([scores_seq, scores_depth], dim=-1)
    weights = F.softmax(scores_combined, dim=-1)

    # Split weights back
    w_seq = weights[:, :, :, :T_kv]           # [B, H_q, T, T_kv]
    w_depth = weights[:, :, :, T_kv:]         # [B, H_q, T, T_kv*L]

    # Weighted sum
    out = torch.matmul(w_seq, V) + torch.matmul(w_depth, V_depth)

    return out


def _moda_chunked(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    K_depth: Tensor,
    V_depth: Tensor,
    num_layers: int,
    scale: float,
    chunk_size: int,
) -> Tensor:
    """Chunk-aware MoDA attention.

    Splits queries into chunks of size C. Each chunk only accesses the
    relevant portion of the depth KV cache, reducing memory from O(T*L)
    to O(C*L) per chunk.
    """
    B, H_q, T, d = Q.shape
    _, _, T_kv, _ = K.shape
    L = num_layers
    C = chunk_size

    output = torch.zeros_like(Q)

    for chunk_start in range(0, T, C):
        chunk_end = min(chunk_start + C, T)
        Q_chunk = Q[:, :, chunk_start:chunk_end]  # [B, H_q, C', d]
        C_actual = chunk_end - chunk_start

        # Sequence attention: this chunk queries ALL sequence keys (with causal mask)
        scores_seq = torch.matmul(Q_chunk, K.transpose(-2, -1)) * scale  # [B, H_q, C', T_kv]
        # Causal: query at position (chunk_start + i) can attend to key j if chunk_start + i >= j
        q_positions = torch.arange(chunk_start, chunk_end, device=Q.device)
        k_positions = torch.arange(T_kv, device=Q.device)
        causal_mask = q_positions.unsqueeze(1) < k_positions.unsqueeze(0)  # [C', T_kv]
        scores_seq = scores_seq.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Depth attention: only load depth KV for positions in this chunk
        depth_start = chunk_start * L
        depth_end = chunk_end * L
        K_depth_chunk = K_depth[:, :, depth_start:depth_end]  # [B, H_q, C'*L, d]
        V_depth_chunk = V_depth[:, :, depth_start:depth_end]

        scores_depth = torch.matmul(Q_chunk, K_depth_chunk.transpose(-2, -1)) * scale

        # Depth mask within chunk
        q_pos_local = torch.arange(C_actual, device=Q.device)  # [C']
        d_pos_local = torch.arange(C_actual * L, device=Q.device)  # [C'*L]
        d_token_local = d_pos_local // L  # token index within chunk
        depth_mask = q_pos_local.unsqueeze(1) != d_token_local.unsqueeze(0)
        scores_depth = scores_depth.masked_fill(depth_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Combined softmax
        scores_combined = torch.cat([scores_seq, scores_depth], dim=-1)
        weights = F.softmax(scores_combined, dim=-1)

        w_seq = weights[:, :, :, :T_kv]
        w_depth = weights[:, :, :, T_kv:]

        output[:, :, chunk_start:chunk_end] = (
            torch.matmul(w_seq, V) + torch.matmul(w_depth, V_depth_chunk)
        )

    return output
