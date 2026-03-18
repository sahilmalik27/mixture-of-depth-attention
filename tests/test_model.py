"""Tests for MoDATransformerBlock and MoDAModel."""

import torch
import pytest
from moda.config import MoDAConfig
from moda.model import MoDATransformerBlock, MoDAModel
from moda.cache import DepthKVCache


class TestMoDATransformerBlock:
    def test_forward_no_cache(self):
        cfg = MoDAConfig(d_model=64, num_heads=4, num_kv_heads=2, num_layers=3)
        block = MoDATransformerBlock(cfg, layer_idx=0)
        x = torch.randn(2, 8, cfg.d_model)

        out = block(x)
        assert out.shape == (2, 8, cfg.d_model)

    def test_forward_with_cache(self):
        cfg = MoDAConfig(d_model=64, num_heads=4, num_kv_heads=2, num_layers=3)
        block = MoDATransformerBlock(cfg, layer_idx=0)
        B, T = 2, 8

        cache = DepthKVCache(B, T, cfg.num_layers * 2, cfg.num_kv_heads, cfg.head_dim)
        x = torch.randn(B, T, cfg.d_model)

        out = block(x, depth_cache=cache)
        assert out.shape == (B, T, cfg.d_model)

        # Cache should have been written to
        K, V = cache.read(seq_len=T)
        assert not (K == 0).all()  # something was written

    def test_gradient_flow(self):
        cfg = MoDAConfig(d_model=64, num_heads=4, num_kv_heads=2, num_layers=3)
        block = MoDATransformerBlock(cfg, layer_idx=0)
        x = torch.randn(2, 4, cfg.d_model, requires_grad=True)

        out = block(x)
        out.sum().backward()
        assert x.grad is not None


class TestMoDAModel:
    def test_forward_shape(self):
        cfg = MoDAConfig(d_model=64, num_heads=4, num_kv_heads=2, num_layers=3, max_seq_len=32)
        model = MoDAModel(cfg)
        x = torch.randn(2, 8, cfg.d_model)

        out = model(x)
        assert out.shape == (2, 8, cfg.d_model)

    def test_forward_no_depth(self):
        cfg = MoDAConfig(d_model=64, num_heads=4, num_kv_heads=2, num_layers=3, max_seq_len=32)
        model = MoDAModel(cfg)
        x = torch.randn(2, 8, cfg.d_model)

        out = model(x, use_depth_cache=False)
        assert out.shape == (2, 8, cfg.d_model)

    def test_depth_changes_output(self):
        cfg = MoDAConfig(d_model=64, num_heads=4, num_kv_heads=2, num_layers=3, max_seq_len=32)
        model = MoDAModel(cfg)
        model.eval()
        x = torch.randn(1, 8, cfg.d_model)

        with torch.no_grad():
            out_no_depth = model(x, use_depth_cache=False)
            out_with_depth = model(x, use_depth_cache=True)

        # Outputs should differ (depth cache enables cross-layer attention)
        assert not torch.allclose(out_no_depth, out_with_depth, atol=1e-3)

    def test_gradient_flow(self):
        cfg = MoDAConfig(d_model=64, num_heads=4, num_kv_heads=2, num_layers=2, max_seq_len=16)
        model = MoDAModel(cfg)
        x = torch.randn(1, 4, cfg.d_model, requires_grad=True)

        out = model(x)
        out.sum().backward()
        assert x.grad is not None

    def test_pre_norm(self):
        cfg = MoDAConfig(
            d_model=64, num_heads=4, num_kv_heads=2, num_layers=2,
            max_seq_len=16, post_norm=False,
        )
        model = MoDAModel(cfg)
        x = torch.randn(1, 4, cfg.d_model)
        out = model(x)
        assert out.shape == (1, 4, cfg.d_model)
