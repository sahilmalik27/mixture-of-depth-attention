"""Tests for MoDAAttention module."""

import torch
import pytest
from moda.config import MoDAConfig
from moda.attention import MoDAAttention


class TestMoDAAttention:
    def _make_config(self, **kwargs):
        defaults = dict(d_model=64, num_heads=4, num_kv_heads=2, num_layers=3)
        defaults.update(kwargs)
        return MoDAConfig(**defaults)

    def test_forward_no_depth(self):
        """Without depth cache, should behave like standard attention."""
        cfg = self._make_config()
        attn = MoDAAttention(cfg)
        x = torch.randn(2, 8, cfg.d_model)

        out, k_write, v_write = attn(x)
        assert out.shape == (2, 8, cfg.d_model)
        assert k_write.shape == (2, cfg.num_kv_heads, 8, cfg.head_dim)
        assert v_write.shape == (2, cfg.num_kv_heads, 8, cfg.head_dim)

    def test_forward_with_depth(self):
        """With depth KV, should fuse sequence + depth attention."""
        cfg = self._make_config()
        attn = MoDAAttention(cfg)
        B, T = 2, 8
        L = 4  # some preceding layers
        x = torch.randn(B, T, cfg.d_model)
        K_depth = torch.randn(B, cfg.num_kv_heads, T * L, cfg.head_dim)
        V_depth = torch.randn(B, cfg.num_kv_heads, T * L, cfg.head_dim)

        out, k_write, v_write = attn(x, K_depth, V_depth)
        assert out.shape == (B, T, cfg.d_model)

    def test_depth_changes_output(self):
        """Adding depth KV should change the attention output."""
        cfg = self._make_config()
        attn = MoDAAttention(cfg)
        attn.eval()
        B, T, L = 1, 4, 3
        x = torch.randn(B, T, cfg.d_model)

        with torch.no_grad():
            out_no_depth, _, _ = attn(x)

            K_depth = torch.randn(B, cfg.num_kv_heads, T * L, cfg.head_dim)
            V_depth = torch.randn(B, cfg.num_kv_heads, T * L, cfg.head_dim)
            out_with_depth, _, _ = attn(x, K_depth, V_depth)

        # Outputs should differ
        assert not torch.allclose(out_no_depth, out_with_depth, atol=1e-3)

    def test_gradient_through_depth(self):
        """Gradients should flow through depth KV."""
        cfg = self._make_config()
        attn = MoDAAttention(cfg)
        B, T, L = 1, 4, 3
        x = torch.randn(B, T, cfg.d_model, requires_grad=True)
        K_depth = torch.randn(B, cfg.num_kv_heads, T * L, cfg.head_dim, requires_grad=True)
        V_depth = torch.randn(B, cfg.num_kv_heads, T * L, cfg.head_dim, requires_grad=True)

        out, _, _ = attn(x, K_depth, V_depth)
        out.sum().backward()

        assert x.grad is not None
        assert K_depth.grad is not None
        assert V_depth.grad is not None
