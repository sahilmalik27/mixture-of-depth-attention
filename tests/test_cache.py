"""Tests for DepthKVCache."""

import torch
import pytest
from moda.cache import DepthKVCache


class TestDepthKVCache:
    def test_basic_write_read(self):
        B, T, L, H_k, d = 2, 4, 3, 2, 8
        cache = DepthKVCache(B, T, L, H_k, d)

        # Write layer 0
        k0 = torch.randn(B, H_k, T, d)
        v0 = torch.randn(B, H_k, T, d)
        cache.write(0, k0, v0)

        K_depth, V_depth = cache.read(seq_len=T)
        assert K_depth.shape == (B, H_k, T * L, d)
        assert V_depth.shape == (B, H_k, T * L, d)

        # Check that layer 0 entries are at indices 0, L, 2L, 3L, ...
        for t in range(T):
            idx = t * L + 0
            torch.testing.assert_close(K_depth[:, :, idx], k0[:, :, t])
            torch.testing.assert_close(V_depth[:, :, idx], v0[:, :, t])

    def test_multiple_layers(self):
        B, T, L, H_k, d = 1, 3, 4, 1, 4
        cache = DepthKVCache(B, T, L, H_k, d)

        ks, vs = [], []
        for layer in range(L):
            k = torch.randn(B, H_k, T, d)
            v = torch.randn(B, H_k, T, d)
            cache.write(layer, k, v)
            ks.append(k)
            vs.append(v)

        K_depth, V_depth = cache.read(seq_len=T)

        # Verify layout: token t, layer l is at index t*L + l
        for t in range(T):
            for l in range(L):
                idx = t * L + l
                torch.testing.assert_close(K_depth[:, :, idx], ks[l][:, :, t])
                torch.testing.assert_close(V_depth[:, :, idx], vs[l][:, :, t])

    def test_reset(self):
        B, T, L, H_k, d = 1, 2, 2, 1, 4
        cache = DepthKVCache(B, T, L, H_k, d)
        cache.write(0, torch.ones(B, H_k, T, d), torch.ones(B, H_k, T, d))
        cache.reset()
        K, V = cache.read(seq_len=T)
        assert (K == 0).all()
        assert (V == 0).all()

    def test_partial_read(self):
        B, T, L, H_k, d = 1, 8, 3, 1, 4
        cache = DepthKVCache(B, T, L, H_k, d)
        cache.write(0, torch.randn(B, H_k, T, d), torch.randn(B, H_k, T, d))

        # Read only first 4 tokens worth
        K, V = cache.read(seq_len=4)
        assert K.shape == (B, H_k, 4 * L, d)
