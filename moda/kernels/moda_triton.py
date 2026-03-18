"""Triton kernel for MoDA attention (fused forward pass).

Fuses sequence attention and depth attention into a single kernel using
online softmax (Flash Attention style), so both phases share softmax state.
"""

import torch
from torch import Tensor
from typing import Optional

try:
    import triton
    import triton.language as tl
except ImportError:
    raise ImportError("Triton >= 3.0 required for moda_triton. Install with: pip install triton>=3.0")


@triton.jit
def _moda_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    K_depth_ptr, V_depth_ptr,
    O_ptr,
    stride_qn, stride_qt, stride_qd,
    stride_kn, stride_kt, stride_kd,
    stride_vn, stride_vt, stride_vd,
    stride_kdn, stride_kdt, stride_kdd,
    stride_vdn, stride_vdt, stride_vdd,
    stride_on, stride_ot, stride_od,
    T: tl.constexpr, d: tl.constexpr, L: tl.constexpr,
    scale,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fused MoDA forward kernel with online softmax.

    Inputs are pre-expanded for GQA and reshaped to [N, T, d] where N = B*H_q.
    Grid: (cdiv(T, BLOCK_Q), N)
    """
    pid_q = tl.program_id(0)
    pid_n = tl.program_id(1)

    q_block_start = pid_q * BLOCK_Q
    q_offsets = q_block_start + tl.arange(0, BLOCK_Q)
    d_offsets = tl.arange(0, d)

    # Load Q block: [BLOCK_Q, d]
    q_ptrs = Q_ptr + pid_n * stride_qn + q_offsets[:, None] * stride_qt + d_offsets[None, :] * stride_qd
    q_mask = q_offsets[:, None] < T
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    # Online softmax state
    m = tl.full([BLOCK_Q], value=float("-inf"), dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q], dtype=tl.float32)
    o = tl.zeros([BLOCK_Q, d], dtype=tl.float32)

    # ========== Sequence phase (causal) ==========
    for k_block_start in range(0, T, BLOCK_K):
        k_offsets = k_block_start + tl.arange(0, BLOCK_K)

        k_ptrs = K_ptr + pid_n * stride_kn + k_offsets[:, None] * stride_kt + d_offsets[None, :] * stride_kd
        k_mask = k_offsets[:, None] < T
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # [BLOCK_Q, BLOCK_K]
        s = tl.dot(q, tl.trans(k)) * scale

        # Causal mask
        causal = q_offsets[:, None] >= k_offsets[None, :]
        valid = (q_offsets[:, None] < T) & (k_offsets[None, :] < T)
        s = tl.where(causal & valid, s, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])

        v_ptrs = V_ptr + pid_n * stride_vn + k_offsets[:, None] * stride_vt + d_offsets[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        o = o * alpha[:, None] + tl.dot(p, v)
        acc = acc * alpha + tl.sum(p, axis=1)
        m = m_new

    # ========== Depth phase ==========
    depth_start = q_block_start * L
    depth_len = BLOCK_Q * L
    # Upper bound for valid entries
    depth_end_val = tl.minimum((q_block_start + BLOCK_Q), T) * L

    for d_block_start in range(0, depth_len, BLOCK_K):
        d_abs_offsets = depth_start + d_block_start + tl.arange(0, BLOCK_K)

        kd_ptrs = K_depth_ptr + pid_n * stride_kdn + d_abs_offsets[:, None] * stride_kdt + d_offsets[None, :] * stride_kdd
        kd_mask = d_abs_offsets[:, None] < depth_end_val
        kd = tl.load(kd_ptrs, mask=kd_mask, other=0.0).to(tl.float32)

        s = tl.dot(q, tl.trans(kd)) * scale

        # Depth mask: query at pos i attends to depth entry j if i == floor(j/L)
        d_token_pos = d_abs_offsets // L
        depth_valid = (q_offsets[:, None] == d_token_pos[None, :])
        bounds_valid = (q_offsets[:, None] < T) & (d_abs_offsets[None, :] < depth_end_val)
        s = tl.where(depth_valid & bounds_valid, s, float("-inf"))

        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])

        vd_ptrs = V_depth_ptr + pid_n * stride_vdn + d_abs_offsets[:, None] * stride_vdt + d_offsets[None, :] * stride_vdd
        vd = tl.load(vd_ptrs, mask=kd_mask, other=0.0).to(tl.float32)

        o = o * alpha[:, None] + tl.dot(p, vd)
        acc = acc * alpha + tl.sum(p, axis=1)
        m = m_new

    # Normalize
    o = o / acc[:, None]

    # Store
    o_ptrs = O_ptr + pid_n * stride_on + q_offsets[:, None] * stride_ot + d_offsets[None, :] * stride_od
    o_mask = q_offsets[:, None] < T
    tl.store(o_ptrs, o, mask=o_mask)


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
    """MoDA attention using fused Triton kernel.

    Same interface as moda_attention_naive. Inputs must be on CUDA.

    Args:
        Q: [B, H_q, T, d] queries.
        K: [B, H_k, T, d] sequence keys.
        V: [B, H_k, T, d] sequence values.
        K_depth: [B, H_k, T*L, d] depth keys.
        V_depth: [B, H_k, T*L, d] depth values.
        num_layers: Number of layers (L).
        scale: Attention scale. Defaults to 1/sqrt(d).
        chunk_size: Unused (chunking is implicit in the kernel).

    Returns:
        [B, H_q, T, d] output tensor.
    """
    B, H_q, T, d = Q.shape
    _, H_k, _, _ = K.shape
    G = H_q // H_k
    L = num_layers

    if scale is None:
        scale = d ** -0.5

    # Expand KV for GQA
    if G > 1:
        K = K.repeat_interleave(G, dim=1)
        V = V.repeat_interleave(G, dim=1)
        K_depth = K_depth.repeat_interleave(G, dim=1)
        V_depth = V_depth.repeat_interleave(G, dim=1)

    N = B * H_q

    # Reshape to [N, T, d] / [N, T*L, d]
    Q_flat = Q.reshape(N, T, d).contiguous()
    K_flat = K.reshape(N, T, d).contiguous()
    V_flat = V.reshape(N, T, d).contiguous()
    K_depth_flat = K_depth.reshape(N, T * L, d).contiguous()
    V_depth_flat = V_depth.reshape(N, T * L, d).contiguous()

    O_flat = torch.empty_like(Q_flat)

    # Block sizes: tl.dot requires inner dim >= 16
    BLOCK_Q = max(16, min(64, triton.next_power_of_2(T)))
    BLOCK_K = max(16, min(64, triton.next_power_of_2(max(T, L))))

    grid = (triton.cdiv(T, BLOCK_Q), N)

    _moda_fwd_kernel[grid](
        Q_flat, K_flat, V_flat,
        K_depth_flat, V_depth_flat,
        O_flat,
        Q_flat.stride(0), Q_flat.stride(1), Q_flat.stride(2),
        K_flat.stride(0), K_flat.stride(1), K_flat.stride(2),
        V_flat.stride(0), V_flat.stride(1), V_flat.stride(2),
        K_depth_flat.stride(0), K_depth_flat.stride(1), K_depth_flat.stride(2),
        V_depth_flat.stride(0), V_depth_flat.stride(1), V_depth_flat.stride(2),
        O_flat.stride(0), O_flat.stride(1), O_flat.stride(2),
        T=T, d=d, L=L,
        scale=scale,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K,
    )

    return O_flat.reshape(B, H_q, T, d)
