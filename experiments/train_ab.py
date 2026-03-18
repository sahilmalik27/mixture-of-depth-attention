#!/usr/bin/env python3
"""A/B experiment: Standard attention vs MoDA attention on 125M GPT.

Both models share the same Llama-style architecture (RMSNorm, SwiGLU, RoPE, GQA).
The only difference: MoDA model has depth KV projections and uses
moda_attention_naive to fuse sequence + depth attention in a single softmax.

Usage:
    python train_ab.py --mode baseline --max-hours 4
    python train_ab.py --mode moda --max-hours 4
"""

import os
import sys
import time
import math
import pickle
import gc
import json
import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt_util
import pyarrow.parquet as pq

# Add MoDA library
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moda.kernels import moda_attention_naive

# ============================================================================
# Model Architecture (125M Llama-style)
# ============================================================================

@dataclass
class Config:
    vocab_size: int = 8192
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 768
    max_seq_len: int = 2048
    mlp_ratio: float = 4.0
    norm_eps: float = 1e-5
    moda_chunk_size: int = 256


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[None, :xq_.shape[1], None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ---- SwiGLU MLP (shared) ----

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = int(config.n_embd * config.mlp_ratio)
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---- Baseline attention ----

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = self.n_head // self.n_kv_head

        self.c_q = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, freqs_cis):
        B, T, C = x.shape
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(B, T, self.n_head, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(B, T, self.n_head, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


# ---- MoDA attention (adds depth KV read/write) ----

class MoDACausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = self.n_head // self.n_kv_head
        self.scale = self.head_dim ** -0.5
        self.chunk_size = config.moda_chunk_size

        self.c_q = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Depth KV write projections
        self.c_depth_k = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_depth_v = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)

    def forward(self, x, freqs_cis, K_depth=None, V_depth=None, num_depth_layers=0):
        B, T, C = x.shape
        H_q, H_k, d = self.n_head, self.n_kv_head, self.head_dim

        q = self.c_q(x).view(B, T, H_q, d)
        k = self.c_k(x).view(B, T, H_k, d)
        v = self.c_v(x).view(B, T, H_k, d)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Depth KV writes (from input, before RoPE — position encoded via masking)
        k_write = self.c_depth_k(x).view(B, T, H_k, d).transpose(1, 2)
        v_write = self.c_depth_v(x).view(B, T, H_k, d).transpose(1, 2)

        q = q.transpose(1, 2)  # [B, H_q, T, d]
        k = k.transpose(1, 2)  # [B, H_k, T, d]
        v = v.transpose(1, 2)

        if K_depth is not None and num_depth_layers > 0:
            y = moda_attention_naive(
                q, k, v, K_depth, V_depth,
                num_layers=num_depth_layers,
                scale=self.scale,
                chunk_size=self.chunk_size,
            )
        else:
            # First layer: no depth cache yet, use standard causal attention
            if self.n_rep > 1:
                k_exp = k.repeat_interleave(self.n_rep, dim=1)
                v_exp = v.repeat_interleave(self.n_rep, dim=1)
            else:
                k_exp, v_exp = k, v
            y = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), k_write, v_write


# ---- Blocks ----

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.ln_2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln_1(x), freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x


class MoDABlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = MoDACausalSelfAttention(config, layer_idx)
        self.ln_2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)

        # FFN depth KV write projections
        head_dim = config.n_embd // config.n_head
        self.ffn_depth_k = nn.Linear(config.n_embd, config.n_kv_head * head_dim, bias=False)
        self.ffn_depth_v = nn.Linear(config.n_embd, config.n_kv_head * head_dim, bias=False)
        self.n_kv_head = config.n_kv_head
        self.head_dim = head_dim

    def forward(self, x, freqs_cis, depth_keys, depth_vals):
        B, T, C = x.shape
        H_k, d = self.n_kv_head, self.head_dim

        # Build depth cache tensors from accumulated lists
        L = len(depth_keys)
        K_depth, V_depth = None, None
        if L > 0:
            # Stack: [L, B, H_k, T, d] -> permute to [B, H_k, T, L, d] -> reshape [B, H_k, T*L, d]
            K_depth = torch.stack(depth_keys, dim=0).permute(1, 2, 3, 0, 4).reshape(B, H_k, T * L, d)
            V_depth = torch.stack(depth_vals, dim=0).permute(1, 2, 3, 0, 4).reshape(B, H_k, T * L, d)

        # Attention (pre-norm)
        attn_out, k_write_attn, v_write_attn = self.attn(
            self.ln_1(x), freqs_cis, K_depth, V_depth, num_depth_layers=L,
        )
        x = x + attn_out

        # FFN (pre-norm)
        ffn_out = self.mlp(self.ln_2(x))
        x = x + ffn_out

        # FFN depth KV writes
        k_write_ffn = self.ffn_depth_k(x).view(B, T, H_k, d).transpose(1, 2)
        v_write_ffn = self.ffn_depth_v(x).view(B, T, H_k, d).transpose(1, 2)

        new_keys = [k_write_attn, k_write_ffn]
        new_vals = [v_write_attn, v_write_ffn]
        return x, new_keys, new_vals


# ---- Models ----

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd, config.norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

        head_dim = config.n_embd // config.n_head
        self.register_buffer("freqs_cis", precompute_freqs_cis(head_dim, config.max_seq_len))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.transformer.wte(idx)
        freqs_cis = self.freqs_cis[:T]

        for block in self.transformer.h:
            x = block(x, freqs_cis)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class MoDAGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([MoDABlock(config, i) for i in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd, config.norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

        head_dim = config.n_embd // config.n_head
        self.register_buffer("freqs_cis", precompute_freqs_cis(head_dim, config.max_seq_len))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.transformer.wte(idx)
        freqs_cis = self.freqs_cis[:T]

        depth_keys = []
        depth_vals = []
        for block in self.transformer.h:
            # Gradient checkpointing: recompute each block's forward during backward
            # to avoid storing all depth cache intermediates (GQA expansions, attn matrices).
            # Flatten depth lists into *args for checkpoint compatibility.
            def _run_block(blk, x_in, fc, *flat_depth):
                n = len(flat_depth) // 2
                dk = list(flat_depth[:n])
                dv = list(flat_depth[n:])
                x_out, nk, nv = blk(x_in, fc, dk, dv)
                return (x_out,) + tuple(nk) + tuple(nv)

            flat = tuple(depth_keys) + tuple(depth_vals)
            result = ckpt_util.checkpoint(
                _run_block, block, x, freqs_cis, *flat,
                use_reentrant=False,
            )
            x = result[0]
            depth_keys = depth_keys + [result[1], result[2]]
            depth_vals = depth_vals + [result[3], result[4]]

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Data Loading (from train_1b.py)
# ============================================================================

class ShardDataLoader:
    def __init__(self, data_dir, tokenizer, seq_len=2048, batch_size=8, val_shard=6542):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.val_shard = val_shard

        self.shards = sorted([
            f for f in self.data_dir.glob("shard_*.parquet")
            if f"shard_{val_shard:05d}" not in f.name
        ])
        print(f"Found {len(self.shards)} training shards")

        self.current_shard_idx = 0
        self.token_buffer = []
        self.buffer_pos = 0
        self._load_shard()

    def _load_shard(self):
        if self.current_shard_idx >= len(self.shards):
            self.current_shard_idx = 0
        shard_path = self.shards[self.current_shard_idx]
        table = pq.read_table(shard_path, columns=["text"])
        texts = table.column("text").to_pylist()

        tokens = []
        for text in texts:
            encoded = self.tokenizer.encode(text)
            tokens.extend(encoded)

        self.token_buffer = tokens
        self.buffer_pos = 0
        self.current_shard_idx += 1
        print(f"Loaded shard {shard_path.name}: {len(tokens):,} tokens")

    def get_batch(self):
        total_tokens = self.batch_size * (self.seq_len + 1)

        while len(self.token_buffer) - self.buffer_pos < total_tokens:
            remaining = self.token_buffer[self.buffer_pos:]
            self._load_shard()
            self.token_buffer = remaining + self.token_buffer
            self.buffer_pos = 0

        tokens = self.token_buffer[self.buffer_pos:self.buffer_pos + total_tokens]
        self.buffer_pos += total_tokens

        tokens = torch.tensor(tokens, dtype=torch.long).view(self.batch_size, self.seq_len + 1)
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        return x, y


# ============================================================================
# Training
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "moda"], required=True)
    parser.add_argument("--max-hours", type=float, default=4.0)
    args = parser.parse_args()

    # ---- Constants ----
    CACHE_DIR = os.path.expanduser("~/.cache/autoresearch")
    DATA_DIR = os.path.join(CACHE_DIR, "data")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
    CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints", args.mode)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # MoDA needs more memory for depth KV stacking + GQA expansion in the kernel.
    # Halve micro-batch and double grad_accum to keep effective batch = 32.
    if args.mode == "moda":
        BATCH_SIZE = 4
        GRAD_ACCUM = 8
    else:
        BATCH_SIZE = 8
        GRAD_ACCUM = 4
    SEQ_LEN = 2048
    MAX_LR = 6e-4
    MIN_LR = 6e-5
    WARMUP_STEPS = 200
    LOG_INTERVAL = 10
    EVAL_INTERVAL = 50 if args.mode == "moda" else 100
    SAVE_INTERVAL = 500
    EVAL_STEPS = 20

    # ---- Device setup ----
    mem_frac = 0.20 if args.mode == "moda" else 0.15
    torch.cuda.set_per_process_memory_fraction(mem_frac)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # ---- Tokenizer ----
    tokenizer_path = os.path.join(CACHE_DIR, "tokenizer", "tokenizer.pkl")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer loaded (n_vocab={tokenizer.n_vocab})")

    # ---- Model ----
    config = Config()
    if args.mode == "baseline":
        model = GPT(config)
    else:
        model = MoDAGPT(config)

    total_params = model.count_params()
    print(f"Mode: {args.mode} | Params: {total_params:,} ({total_params/1e6:.1f}M)")
    model = model.to(device=device, dtype=dtype)

    if torch.cuda.is_available():
        print(f"Model memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MAX_LR,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # ---- Data ----
    loader = ShardDataLoader(DATA_DIR, tokenizer, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)

    # ---- LR schedule ----
    DECAY_STEPS = 50000

    def get_lr(step):
        if step < WARMUP_STEPS:
            return MAX_LR * (step + 1) / WARMUP_STEPS
        progress = min((step - WARMUP_STEPS) / DECAY_STEPS, 1.0)
        return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * progress))

    # ---- Validation ----
    def evaluate():
        model.eval()
        val_shard = os.path.join(DATA_DIR, "shard_06542.parquet")
        table = pq.read_table(val_shard, columns=["text"])
        texts = table.column("text").to_pylist()
        tokens = []
        for text in texts[:100]:
            tokens.extend(tokenizer.encode(text))

        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for i in range(min(EVAL_STEPS, len(tokens) // (SEQ_LEN + 1))):
                start = i * (SEQ_LEN + 1)
                chunk = torch.tensor(
                    tokens[start:start + SEQ_LEN + 1], dtype=torch.long, device=device
                ).unsqueeze(0)
                x, y = chunk[:, :-1], chunk[:, 1:]
                with torch.amp.autocast("cuda", dtype=dtype):
                    _, loss = model(x, y)
                total_loss += loss.item()
                count += 1
        model.train()
        return total_loss / max(count, 1)

    # ---- Resume checkpoint ----
    step = 0
    tokens_processed = 0
    ckpt_path = os.path.join(CKPT_DIR, "latest.pt")
    if os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt.get("step", 0)
        tokens_processed = ckpt.get("tokens", 0)
        print(f"Resumed at step {step}")

    # ---- Log files ----
    train_log_path = os.path.join(LOG_DIR, f"{args.mode}.jsonl")
    val_log_path = os.path.join(LOG_DIR, f"{args.mode}_val.jsonl")
    train_log = open(train_log_path, "a")
    val_log = open(val_log_path, "a")

    # ---- Training loop ----
    max_seconds = args.max_hours * 3600
    print(f"\n{'='*60}")
    print(f"Training {args.mode} for {args.max_hours} hours")
    print(f"Effective batch: {BATCH_SIZE * GRAD_ACCUM} seqs = {BATCH_SIZE * GRAD_ACCUM * SEQ_LEN:,} tok")
    print(f"{'='*60}\n")

    model.train()
    t_start = time.time()
    running_loss = 0.0
    step_t0 = time.time()

    while time.time() - t_start < max_seconds:
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()
        accum_loss = 0.0

        for micro_step in range(GRAD_ACCUM):
            x, y = loader.get_batch()
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", dtype=dtype):
                _, loss = model(x, y)
            loss = loss / GRAD_ACCUM
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        step_tokens = BATCH_SIZE * GRAD_ACCUM * SEQ_LEN
        tokens_processed += step_tokens
        running_loss += accum_loss

        # Log
        if step % LOG_INTERVAL == 0:
            now = time.time()
            avg_loss = running_loss / LOG_INTERVAL
            bpb = avg_loss / math.log(2)
            elapsed = now - t_start
            step_elapsed = now - step_t0
            tok_per_sec = (LOG_INTERVAL * step_tokens) / step_elapsed

            entry = {
                "step": step,
                "loss": round(avg_loss, 5),
                "bpb": round(bpb, 5),
                "lr": round(lr, 8),
                "tok_per_sec": round(tok_per_sec),
                "timestamp": round(elapsed, 1),
            }
            train_log.write(json.dumps(entry) + "\n")
            train_log.flush()

            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            remaining = max_seconds - elapsed
            print(
                f"step {step:6d} | loss {avg_loss:.4f} | bpb {bpb:.4f} | "
                f"lr {lr:.2e} | {tok_per_sec:,.0f} tok/s | "
                f"{tokens_processed/1e6:.1f}M tok | mem {mem_gb:.1f}GB | "
                f"{remaining/3600:.1f}h left"
            )

            running_loss = 0.0
            step_t0 = now

        # Validate
        if step % EVAL_INTERVAL == 0:
            val_loss = evaluate()
            val_bpb = val_loss / math.log(2)
            elapsed = time.time() - t_start
            entry = {
                "step": step,
                "val_loss": round(val_loss, 5),
                "val_bpb": round(val_bpb, 5),
                "timestamp": round(elapsed, 1),
            }
            val_log.write(json.dumps(entry) + "\n")
            val_log.flush()
            print(f"  >>> val_loss={val_loss:.4f} val_bpb={val_bpb:.4f}")

        # Checkpoint
        if step % SAVE_INTERVAL == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "tokens": tokens_processed,
                "config": config.__dict__,
                "mode": args.mode,
            }, ckpt_path)
            print(f"  Checkpoint saved at step {step}")

        # Memory cleanup
        if step % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Final validation
    val_loss = evaluate()
    val_bpb = val_loss / math.log(2)
    elapsed = time.time() - t_start
    entry = {
        "step": step,
        "val_loss": round(val_loss, 5),
        "val_bpb": round(val_bpb, 5),
        "timestamp": round(elapsed, 1),
    }
    val_log.write(json.dumps(entry) + "\n")
    val_log.flush()

    # Final checkpoint
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens": tokens_processed,
        "config": config.__dict__,
        "mode": args.mode,
    }, ckpt_path)

    train_log.close()
    val_log.close()

    print(f"\n{'='*60}")
    print(f"Done: {args.mode} | {step} steps | {tokens_processed/1e6:.1f}M tokens | "
          f"final val_bpb={val_bpb:.4f} | {elapsed/3600:.2f}h")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
