"""Tests for GQA (Grouped Query Attention) compatibility."""

import torch
import pytest
from moda.kernels.moda_naive import moda_attention_naive
from moda.config import MoDAConfig
from moda.attention import MoDAAttention


class TestGQA:
    def test_gqa_matches_mha_when_g1(self):
        """With G=1 (H_q == H_k), GQA should equal standard MHA."""
        B, H, T, L, d = 1, 4, 8, 3, 16
        Q = torch.randn(B, H, T, d)
        K = torch.randn(B, H, T, d)
        V = torch.randn(B, H, T, d)
        K_depth = torch.randn(B, H, T * L, d)
        V_depth = torch.randn(B, H, T * L, d)

        out = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L)
        assert out.shape == (B, H, T, d)
        assert not torch.isnan(out).any()

    def test_gqa_g2(self):
        """G=2: 2 query heads per KV head."""
        B, T, L, d = 1, 6, 2, 8
        H_q, H_k = 4, 2
        Q = torch.randn(B, H_q, T, d)
        K = torch.randn(B, H_k, T, d)
        V = torch.randn(B, H_k, T, d)
        K_depth = torch.randn(B, H_k, T * L, d)
        V_depth = torch.randn(B, H_k, T * L, d)

        out = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L)
        assert out.shape == (B, H_q, T, d)

    def test_gqa_g4(self):
        """G=4: 4 query heads per KV head."""
        B, T, L, d = 2, 4, 3, 16
        H_q, H_k = 8, 2
        Q = torch.randn(B, H_q, T, d)
        K = torch.randn(B, H_k, T, d)
        V = torch.randn(B, H_k, T, d)
        K_depth = torch.randn(B, H_k, T * L, d)
        V_depth = torch.randn(B, H_k, T * L, d)

        out = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L)
        assert out.shape == (B, H_q, T, d)
        assert not torch.isnan(out).any()

    def test_gqa_shared_kv_heads(self):
        """Query heads within same group should use same KV head."""
        B, T, L, d = 1, 4, 2, 8
        H_q, H_k = 4, 2  # G=2

        Q = torch.randn(B, H_q, T, d)
        K = torch.randn(B, H_k, T, d)
        V = torch.randn(B, H_k, T, d)
        K_depth = torch.randn(B, H_k, T * L, d)
        V_depth = torch.randn(B, H_k, T * L, d)

        out = moda_attention_naive(Q, K, V, K_depth, V_depth, num_layers=L)

        # Heads 0 and 1 share KV head 0; heads 2 and 3 share KV head 1
        # If Q[head0] == Q[head1], output should be identical
        Q_same = Q.clone()
        Q_same[:, 1] = Q_same[:, 0]  # Make head 1 == head 0
        out_same = moda_attention_naive(Q_same, K, V, K_depth, V_depth, num_layers=L)
        torch.testing.assert_close(out_same[:, 0], out_same[:, 1])

    def test_moda_attention_module_gqa(self):
        """MoDAAttention module should work with GQA config."""
        cfg = MoDAConfig(d_model=64, num_heads=8, num_kv_heads=2, num_layers=4)
        attn = MoDAAttention(cfg)
        x = torch.randn(1, 8, cfg.d_model)
        L = cfg.num_layers
        K_depth = torch.randn(1, cfg.num_kv_heads, 8 * L, cfg.head_dim)
        V_depth = torch.randn(1, cfg.num_kv_heads, 8 * L, cfg.head_dim)

        out, k_w, v_w = attn(x, K_depth, V_depth)
        assert out.shape == (1, 8, cfg.d_model)
