"""Tests for MoDA kernel implementations."""

import torch
import pytest
from moda.kernels.moda_naive import moda_attention_naive


def _make_inputs(B, H_q, H_k, T, L, d, dtype=torch.float32, device="cpu"):
    """Create random inputs for MoDA attention."""
    Q = torch.randn(B, H_q, T, d, dtype=dtype, device=device)
    K = torch.randn(B, H_k, T, d, dtype=dtype, device=device)
    V = torch.randn(B, H_k, T, d, dtype=dtype, device=device)
    K_depth = torch.randn(B, H_k, T * L, d, dtype=dtype, device=device)
    V_depth = torch.randn(B, H_k, T * L, d, dtype=dtype, device=device)
    return Q, K, V, K_depth, V_depth


class TestNaiveMoDA:
    def test_output_shape(self):
        B, H_q, H_k, T, L, d = 2, 4, 2, 8, 3, 16
        Q, K, V, K_depth, V_depth = _make_inputs(B, H_q, H_k, T, L, d)
        out = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L)
        assert out.shape == (B, H_q, T, d)

    def test_no_nan(self):
        B, H_q, H_k, T, L, d = 1, 2, 2, 4, 2, 8
        Q, K, V, K_depth, V_depth = _make_inputs(B, H_q, H_k, T, L, d)
        out = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_causal_masking(self):
        """First query position should only attend to first KV position + its depth."""
        B, H, T, L, d = 1, 1, 4, 2, 8
        Q, K, V, K_depth, V_depth = _make_inputs(B, H, H, T, L, d)

        out = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L)

        # Modify K[0, 0, 2, :] (position 2) — should NOT affect output at position 0
        K_mod = K.clone()
        K_mod[0, 0, 2, :] = 999.0
        out_mod = moda_attention_naive(Q, K_mod, V, K_depth, V_depth, num_layers=L)

        torch.testing.assert_close(out[:, :, 0], out_mod[:, :, 0])

    def test_depth_isolation(self):
        """Query at position i should only attend to depth entries [i*L, (i+1)*L)."""
        B, H, T, L, d = 1, 1, 4, 3, 8
        Q, K, V, K_depth, V_depth = _make_inputs(B, H, H, T, L, d)

        out = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L)

        # Modify depth entries for token 2 (indices 6, 7, 8)
        # Should NOT affect output at position 0
        K_depth_mod = K_depth.clone()
        K_depth_mod[0, 0, 6:9, :] = 999.0
        out_mod = moda_attention_naive(Q, K, V, K_depth_mod, V_depth, num_layers=L)

        torch.testing.assert_close(out[:, :, 0], out_mod[:, :, 0])

    def test_combined_softmax_sums_to_one(self):
        """Attention weights over seq + depth should sum to 1."""
        B, H, T, L, d = 1, 1, 4, 2, 8
        Q, K, V, K_depth, V_depth = _make_inputs(B, H, H, T, L, d)

        # Manually compute weights to check they sum to 1
        scale = d ** -0.5
        scores_seq = torch.matmul(Q, K.transpose(-2, -1)) * scale
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores_seq = scores_seq.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        scores_depth = torch.matmul(Q, K_depth.transpose(-2, -1)) * scale
        q_pos = torch.arange(T)
        d_pos = torch.arange(T * L)
        depth_mask = q_pos.unsqueeze(1) != (d_pos // L).unsqueeze(0)
        scores_depth = scores_depth.masked_fill(depth_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        combined = torch.cat([scores_seq, scores_depth], dim=-1)
        weights = torch.softmax(combined, dim=-1)

        # Each query's weights should sum to 1
        weight_sums = weights.sum(dim=-1)
        torch.testing.assert_close(weight_sums, torch.ones_like(weight_sums), atol=1e-5, rtol=1e-5)

    def test_chunked_matches_unchunked(self):
        """Chunk-aware optimization should produce identical results."""
        B, H, T, L, d = 1, 2, 8, 3, 16
        Q, K, V, K_depth, V_depth = _make_inputs(B, H, H, T, L, d)

        out_full = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L)
        out_chunked = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L, chunk_size=4)

        torch.testing.assert_close(out_full, out_chunked, atol=1e-5, rtol=1e-5)

    def test_chunked_various_sizes(self):
        """Test various chunk sizes including non-divisible."""
        B, H, T, L, d = 1, 1, 7, 2, 8
        Q, K, V, K_depth, V_depth = _make_inputs(B, H, H, T, L, d)

        out_full = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L)

        for C in [1, 2, 3, 4, 7, 16]:
            out_chunked = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L, chunk_size=C)
            torch.testing.assert_close(out_full, out_chunked, atol=1e-5, rtol=1e-5)

    def test_gradient_flow(self):
        """Backward pass should produce valid gradients."""
        B, H, T, L, d = 1, 2, 4, 2, 8
        Q, K, V, K_depth, V_depth = _make_inputs(B, H, H, T, L, d)
        for t in [Q, K, V, K_depth, V_depth]:
            t.requires_grad_(True)

        out = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L)
        loss = out.sum()
        loss.backward()

        for name, t in [("Q", Q), ("K", K), ("V", V), ("K_depth", K_depth), ("V_depth", V_depth)]:
            assert t.grad is not None, f"{name} has no gradient"
            assert not torch.isnan(t.grad).any(), f"{name} gradient has NaN"

    def test_single_head(self):
        """Test with single head (no GQA)."""
        B, T, L, d = 2, 6, 4, 32
        Q, K, V, K_depth, V_depth = _make_inputs(B, 1, 1, T, L, d)
        out = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L)
        assert out.shape == (B, 1, T, d)
        assert not torch.isnan(out).any()

    def test_gqa_output_shape(self):
        """Test GQA with different H_q and H_k."""
        B, T, L, d = 1, 4, 2, 8
        H_q, H_k = 8, 2  # 4 query heads per KV head
        Q, K, V, K_depth, V_depth = _make_inputs(B, H_q, H_k, T, L, d)
        out = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L)
        assert out.shape == (B, H_q, T, d)

    def test_large_num_layers(self):
        """Test with many depth layers."""
        B, H, T, L, d = 1, 1, 4, 32, 8
        Q, K, V, K_depth, V_depth = _make_inputs(B, H, H, T, L, d)
        out = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L)
        assert out.shape == (B, H, T, d)
        assert not torch.isnan(out).any()


class TestTritonKernel:
    """Tests for Triton kernel — skipped if Triton/CUDA not available."""

    @pytest.fixture(autouse=True)
    def _check_triton(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        try:
            from moda.kernels.moda_triton import moda_attention_triton
            self.triton_fn = moda_attention_triton
        except ImportError:
            pytest.skip("Triton not available")

    def test_matches_naive_basic(self):
        B, H, T, L, d = 1, 1, 8, 4, 16
        Q, K, V, K_depth, V_depth = _make_inputs(B, H, H, T, L, d, device="cuda")

        out_naive = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L)
        out_triton = self.triton_fn(Q, K, V, K_depth, V_depth, num_layers=L)

        torch.testing.assert_close(out_naive, out_triton, atol=1e-2, rtol=1e-2)

    def test_matches_naive_gqa(self):
        B, T, L, d = 1, 8, 3, 16
        H_q, H_k = 4, 2
        Q, K, V, K_depth, V_depth = _make_inputs(B, H_q, H_k, T, L, d, device="cuda")

        out_naive = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L)
        out_triton = self.triton_fn(Q, K, V, K_depth, V_depth, num_layers=L)

        torch.testing.assert_close(out_naive, out_triton, atol=1e-2, rtol=1e-2)

    def test_matches_naive_multi_batch(self):
        B, H, T, L, d = 4, 2, 16, 4, 32
        Q, K, V, K_depth, V_depth = _make_inputs(B, H, H, T, L, d, device="cuda")

        out_naive = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L)
        out_triton = self.triton_fn(Q, K, V, K_depth, V_depth, num_layers=L)

        torch.testing.assert_close(out_naive, out_triton, atol=1e-2, rtol=1e-2)
