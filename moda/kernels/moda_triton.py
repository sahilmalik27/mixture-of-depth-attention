"""Triton-accelerated MoDA attention (fused forward pass).

Strategy: hybrid SDPA + fused depth Triton kernel.
- Sequence phase: delegate to PyTorch SDPA (uses Flash/memory-efficient attention)
  to get O_seq and LSE_seq.
- Depth phase + combine: a single Triton kernel that computes depth attention
  inline (per-layer iteration, no materialized score matrix) and combines with
  sequence output using LSE-weighted fusion.

Key optimizations:
- Sequence phase runs at native SDPA speed
- No GQA expansion for depth (handled in-kernel)
- No intermediate tensors for depth scores/weights/output
- Single Triton kernel for depth + combine (~L element-wise ops per query)
"""

import torch
from torch import Tensor
from typing import Optional

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _depth_fuse_kernel(
        O_seq_ptr, LSE_seq_ptr,
        Q_ptr, K_depth_ptr, V_depth_ptr,
        O_ptr,
        # O_seq strides [N, T, D] (flattened from [B, H_q, T, D] — may be non-contiguous)
        stride_os_n, stride_os_t, stride_os_d,
        # LSE_seq strides [N, T]
        stride_ls_n, stride_ls_t,
        # Q strides [B, H_q, T, D]
        stride_q_b, stride_q_h, stride_q_t, stride_q_d,
        # K_depth strides [B, H_k, T*L, D]
        stride_kd_b, stride_kd_h, stride_kd_t, stride_kd_d,
        # V_depth strides [B, H_k, T*L, D]
        stride_vd_b, stride_vd_h, stride_vd_t, stride_vd_d,
        # O strides [B, H_q, T, D]
        stride_o_b, stride_o_h, stride_o_t, stride_o_d,
        T: tl.constexpr,
        D: tl.constexpr,
        D_PAD: tl.constexpr,
        L: tl.constexpr,
        G: tl.constexpr,
        scale,
        BLOCK_T: tl.constexpr,
    ):
        """Fused depth attention + sequence/depth combination.

        For each query position, computes:
        1. Depth attention via online softmax over L depth entries
        2. LSE-weighted combination of O_seq (from SDPA) and O_depth

        Grid: (cdiv(T, BLOCK_T), B, H_q)
        """
        pid_t = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_hq = tl.program_id(2)
        pid_hkv = pid_hq // G

        # Token offsets in this block
        t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        d_offs = tl.arange(0, D_PAD)
        t_mask = t_offs < T
        d_mask = d_offs < D
        td_mask = t_mask[:, None] & d_mask[None, :]

        # Flat index for O_seq/LSE_seq: n = pid_b * H_q + pid_hq
        n_idx = pid_b * G * (stride_q_h // stride_q_h) + pid_hq  # simplified: use strides
        # Actually compute from O_seq which is [B, H_q, T, D] possibly non-contiguous
        # Use B*H_q indexing into flattened N dimension
        # O_seq was reshaped as O_seq[b, hq, t, d]
        # stride_os_n is stride for the flattened N = B*H_q dimension

        # Load O_seq: [BLOCK_T, D_PAD]
        os_base = O_seq_ptr + pid_b * stride_os_n * tl.cdiv(stride_q_b, stride_q_h) + pid_hq * stride_os_n
        # Simpler: use Q's B and H strides since O_seq has same logical layout
        # Actually O_seq comes from SDPA and may have different strides.
        # Let's use dedicated strides passed as [B, H_q, T, D]
        # Redefine: stride_os_n = stride for batch dim, stride_os_t for head dim actually...
        # This is getting complicated. Let me use 4D strides for O_seq too.

        # Load LSE_seq: [BLOCK_T]
        lse_ptrs = LSE_seq_ptr + pid_b * stride_ls_n + pid_hq * stride_ls_t + t_offs
        lse_seq = tl.load(lse_ptrs, mask=t_mask, other=float("-inf")).to(tl.float32)

        # Load Q: [BLOCK_T, D_PAD]
        q_ptrs = (Q_ptr
                  + pid_b * stride_q_b + pid_hq * stride_q_h
                  + t_offs[:, None] * stride_q_t + d_offs[None, :] * stride_q_d)
        q = tl.load(q_ptrs, mask=td_mask, other=0.0).to(tl.float32)

        # Load O_seq: [BLOCK_T, D_PAD]
        os_ptrs = (O_seq_ptr
                   + pid_b * stride_os_n + pid_hq * stride_os_t
                   + t_offs[:, None] * stride_os_d + d_offs[None, :])
        # Wait, this doesn't work with arbitrary strides. Let me use 4D strides for O_seq.
        # I'll restructure to pass 4D strides for O_seq and 3D strides for LSE_seq.
        os_ptrs_v2 = (O_seq_ptr
                      + pid_b * stride_os_n + pid_hq * stride_os_t
                      + t_offs[:, None] * stride_os_d + d_offs[None, :] * 1)
        # Hmm, this needs proper stride computation. Let me just pass all strides explicitly.
        pass

    # The above kernel has stride issues. Let me use a cleaner approach.

    @triton.jit
    def _depth_combine_kernel(
        O_seq_ptr, LSE_seq_ptr,
        Q_ptr, K_depth_ptr, V_depth_ptr,
        O_ptr,
        # O_seq strides: [B, H_q, T, D] (4D)
        stride_os_b, stride_os_h, stride_os_t, stride_os_d,
        # LSE_seq strides: [B, H_q, T] (3D)
        stride_ls_b, stride_ls_h, stride_ls_t,
        # Q strides: [B, H_q, T, D]
        stride_q_b, stride_q_h, stride_q_t, stride_q_d,
        # K_depth strides: [B, H_k, T*L, D]
        stride_kd_b, stride_kd_h, stride_kd_t, stride_kd_d,
        # V_depth strides: [B, H_k, T*L, D]
        stride_vd_b, stride_vd_h, stride_vd_t, stride_vd_d,
        # O strides: [B, H_q, T, D]
        stride_o_b, stride_o_h, stride_o_t, stride_o_d,
        T: tl.constexpr,
        D: tl.constexpr,
        D_PAD: tl.constexpr,
        L: tl.constexpr,
        G: tl.constexpr,
        scale,
        BLOCK_T: tl.constexpr,
    ):
        """Fused depth attention + LSE-weighted combination with sequence output.

        Grid: (cdiv(T, BLOCK_T), B, H_q)
        """
        pid_t = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_hq = tl.program_id(2)
        pid_hkv = pid_hq // G

        t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        d_offs = tl.arange(0, D_PAD)
        t_mask = t_offs < T
        d_mask = d_offs < D
        td_mask = t_mask[:, None] & d_mask[None, :]

        # Load Q: [BLOCK_T, D_PAD]
        q_base = Q_ptr + pid_b * stride_q_b + pid_hq * stride_q_h
        q_ptrs = q_base + t_offs[:, None] * stride_q_t + d_offs[None, :] * stride_q_d
        q = tl.load(q_ptrs, mask=td_mask, other=0.0).to(tl.float32)

        # Depth attention via online softmax over L layers
        m = tl.full([BLOCK_T], value=float("-inf"), dtype=tl.float32)
        acc = tl.zeros([BLOCK_T], dtype=tl.float32)
        o_depth = tl.zeros([BLOCK_T, D_PAD], dtype=tl.float32)

        kd_base = K_depth_ptr + pid_b * stride_kd_b + pid_hkv * stride_kd_h
        vd_base = V_depth_ptr + pid_b * stride_vd_b + pid_hkv * stride_vd_h

        for l in range(L):
            depth_idx = tl.where(t_mask, t_offs * L + l, 0)

            # Load K_depth entry: [BLOCK_T, D_PAD]
            kd_ptrs = kd_base + depth_idx[:, None] * stride_kd_t + d_offs[None, :] * stride_kd_d
            kd = tl.load(kd_ptrs, mask=td_mask, other=0.0).to(tl.float32)

            # Per-query dot product
            score = tl.sum(q * kd, axis=1) * scale
            score = tl.where(t_mask, score, float("-inf"))

            # Load V_depth entry: [BLOCK_T, D_PAD]
            vd_ptrs = vd_base + depth_idx[:, None] * stride_vd_t + d_offs[None, :] * stride_vd_d
            vd = tl.load(vd_ptrs, mask=td_mask, other=0.0).to(tl.float32)

            # Online softmax update
            m_new = tl.maximum(m, score)
            alpha = tl.exp(m - m_new)
            p = tl.exp(score - m_new)

            o_depth = o_depth * alpha[:, None] + p[:, None] * vd
            acc = acc * alpha + p
            m = m_new

        # Normalize depth output and compute LSE
        o_depth = o_depth / acc[:, None]
        lse_depth = m + tl.log(acc)

        # Load sequence output and LSE
        os_base = O_seq_ptr + pid_b * stride_os_b + pid_hq * stride_os_h
        os_ptrs = os_base + t_offs[:, None] * stride_os_t + d_offs[None, :] * stride_os_d
        o_seq = tl.load(os_ptrs, mask=td_mask, other=0.0).to(tl.float32)

        ls_base = LSE_seq_ptr + pid_b * stride_ls_b + pid_hq * stride_ls_h
        ls_ptrs = ls_base + t_offs * stride_ls_t
        lse_seq = tl.load(ls_ptrs, mask=t_mask, other=float("-inf")).to(tl.float32)

        # LSE-weighted combination
        M = tl.maximum(lse_seq, lse_depth)
        w_s = tl.exp(lse_seq - M)
        w_d = tl.exp(lse_depth - M)
        denom = w_s + w_d

        o_out = (w_s[:, None] * o_seq + w_d[:, None] * o_depth) / denom[:, None]

        # Store
        o_base = O_ptr + pid_b * stride_o_b + pid_hq * stride_o_h
        o_ptrs = o_base + t_offs[:, None] * stride_o_t + d_offs[None, :] * stride_o_d
        tl.store(o_ptrs, o_out, mask=td_mask)


def moda_attention_triton(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    K_depth: Tensor,
    V_depth: Tensor,
    num_layers: int,
    scale: Optional[float] = None,
    chunk_size: Optional[int] = None,
) -> Tensor:
    """MoDA attention: hybrid SDPA + fused depth Triton kernel.

    Same interface as moda_attention_naive. Inputs must be on CUDA.

    Strategy:
    1. Sequence attention via SDPA -> O_seq, LSE_seq
    2. Fused Triton kernel: depth attention + LSE-weighted combination

    Args:
        Q: [B, H_q, T, d] queries.
        K: [B, H_k, T, d] sequence keys.
        V: [B, H_k, T, d] sequence values.
        K_depth: [B, H_k, T*L, d] depth keys.
        V_depth: [B, H_k, T*L, d] depth values.
        num_layers: Number of layers (L).
        scale: Attention scale. Defaults to 1/sqrt(d).
        chunk_size: Unused (kept for API compatibility).

    Returns:
        [B, H_q, T, d] output tensor.
    """
    B, H_q, T, d = Q.shape
    _, H_k, _, _ = K.shape
    G = H_q // H_k
    L = num_layers

    if scale is None:
        scale = d ** -0.5

    # ===== Phase 1: Sequence attention via SDPA =====
    K_seq = K if G == 1 else K.repeat_interleave(G, dim=1)
    V_seq = V if G == 1 else V.repeat_interleave(G, dim=1)

    O_seq, lse_seq_raw, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(
        Q, K_seq, V_seq, None, True, 0.0, True, scale=scale,
    )
    # LSE may be padded; trim to T
    lse_seq = lse_seq_raw[:, :, :T]  # [B, H_q, T]

    if not HAS_TRITON:
        # Fallback: PyTorch depth + combine
        return _depth_combine_pytorch(Q, K_depth, V_depth, O_seq, lse_seq, L, G, scale)

    # ===== Phase 2: Fused depth + combine (Triton) =====
    O = torch.empty_like(Q)
    D_PAD = max(16, triton.next_power_of_2(d))
    BLOCK_T = max(16, min(128, triton.next_power_of_2(T)))

    grid = (triton.cdiv(T, BLOCK_T), B, H_q)

    # Ensure lse_seq is contiguous for the kernel
    if not lse_seq.is_contiguous():
        lse_seq = lse_seq.contiguous()

    _depth_combine_kernel[grid](
        O_seq, lse_seq,
        Q, K_depth, V_depth,
        O,
        O_seq.stride(0), O_seq.stride(1), O_seq.stride(2), O_seq.stride(3),
        lse_seq.stride(0), lse_seq.stride(1), lse_seq.stride(2),
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K_depth.stride(0), K_depth.stride(1), K_depth.stride(2), K_depth.stride(3),
        V_depth.stride(0), V_depth.stride(1), V_depth.stride(2), V_depth.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        T=T, D=d, D_PAD=D_PAD, L=L, G=G,
        scale=scale,
        BLOCK_T=BLOCK_T,
    )

    return O


def _depth_combine_pytorch(Q, K_depth, V_depth, O_seq, lse_seq, L, G, scale):
    """Fallback: depth attention + combine in pure PyTorch."""
    B, H_q, T, d = Q.shape
    H_k = H_q // G

    K_depth_r = K_depth.reshape(B, H_k, T, L, d)
    V_depth_r = V_depth.reshape(B, H_k, T, L, d)
    Q_g = Q.reshape(B, H_k, G, T, d)

    scores_depth = torch.einsum('bhgtd,bhtld->bhgtl', Q_g, K_depth_r) * scale
    lse_depth = torch.logsumexp(scores_depth, dim=-1).reshape(B, H_q, T)
    w_depth = torch.softmax(scores_depth, dim=-1)
    O_depth = torch.einsum('bhgtl,bhtld->bhgtd', w_depth, V_depth_r).reshape(B, H_q, T, d)

    M = torch.maximum(lse_seq, lse_depth)
    w_s = torch.exp(lse_seq - M)
    w_d = torch.exp(lse_depth - M)
    denom = (w_s + w_d).unsqueeze(-1)
    return (w_s.unsqueeze(-1) * O_seq + w_d.unsqueeze(-1) * O_depth) / denom
