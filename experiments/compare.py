#!/usr/bin/env python3
"""Compare baseline vs MoDA training runs and generate results.md."""

import json
import os
from pathlib import Path


def load_jsonl(path):
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    script_dir = Path(__file__).parent
    log_dir = script_dir / "logs"

    baseline_train = load_jsonl(log_dir / "baseline.jsonl")
    baseline_val = load_jsonl(log_dir / "baseline_val.jsonl")
    moda_train = load_jsonl(log_dir / "moda.jsonl")
    moda_val = load_jsonl(log_dir / "moda_val.jsonl")

    # Extract key metrics
    def summarize(train_entries, val_entries, name):
        info = {"name": name}
        if train_entries:
            info["total_steps"] = train_entries[-1]["step"]
            info["final_train_bpb"] = train_entries[-1]["bpb"]
            info["final_train_loss"] = train_entries[-1]["loss"]
            tok_per_sec_vals = [e["tok_per_sec"] for e in train_entries if "tok_per_sec" in e]
            info["avg_tok_per_sec"] = sum(tok_per_sec_vals) / len(tok_per_sec_vals) if tok_per_sec_vals else 0
            info["total_hours"] = train_entries[-1].get("timestamp", 0) / 3600
        if val_entries:
            best = min(val_entries, key=lambda e: e["val_bpb"])
            info["best_val_bpb"] = best["val_bpb"]
            info["best_val_step"] = best["step"]
            info["final_val_bpb"] = val_entries[-1]["val_bpb"]
            info["final_val_step"] = val_entries[-1]["step"]

            # val_bpb at step milestones
            val_by_step = {e["step"]: e["val_bpb"] for e in val_entries}
            info["val_by_step"] = val_by_step
        return info

    b = summarize(baseline_train, baseline_val, "Baseline")
    m = summarize(moda_train, moda_val, "MoDA")

    # Find common eval steps
    common_steps = sorted(set(b.get("val_by_step", {}).keys()) & set(m.get("val_by_step", {}).keys()))

    # Generate markdown
    lines = []
    lines.append("# A/B Experiment Results: Baseline vs MoDA (125M GPT)\n")
    lines.append("## Summary\n")
    lines.append("| Metric | Baseline | MoDA |")
    lines.append("|--------|----------|------|")

    def row(label, bkey, mkey=None):
        mkey = mkey or bkey
        bv = b.get(bkey, "N/A")
        mv = m.get(mkey, "N/A")
        if isinstance(bv, float):
            bv = f"{bv:.4f}"
        if isinstance(mv, float):
            mv = f"{mv:.4f}"
        lines.append(f"| {label} | {bv} | {mv} |")

    row("Total steps", "total_steps")
    row("Best val_bpb", "best_val_bpb")
    row("Best val_bpb step", "best_val_step")
    row("Final val_bpb", "final_val_bpb")
    row("Final train_bpb", "final_train_bpb")
    row("Avg tok/sec", "avg_tok_per_sec")
    row("Training hours", "total_hours")

    lines.append("")

    # val_bpb curve at common steps
    if common_steps:
        lines.append("## Validation BPB Curve (common steps)\n")
        lines.append("| Step | Baseline val_bpb | MoDA val_bpb | Delta |")
        lines.append("|------|-----------------|-------------|-------|")
        for s in common_steps:
            bv = b["val_by_step"][s]
            mv = m["val_by_step"][s]
            delta = mv - bv
            sign = "+" if delta > 0 else ""
            lines.append(f"| {s} | {bv:.4f} | {mv:.4f} | {sign}{delta:.4f} |")
        lines.append("")

    # Throughput comparison
    if b.get("avg_tok_per_sec") and m.get("avg_tok_per_sec"):
        ratio = m["avg_tok_per_sec"] / b["avg_tok_per_sec"] if b["avg_tok_per_sec"] > 0 else 0
        overhead = (1 - ratio) * 100
        lines.append("## Throughput\n")
        lines.append(f"- Baseline: **{b['avg_tok_per_sec']:,.0f}** tok/sec")
        lines.append(f"- MoDA: **{m['avg_tok_per_sec']:,.0f}** tok/sec")
        lines.append(f"- MoDA overhead: **{overhead:.1f}%** slower")
        lines.append("")

    # Token-efficiency comparison
    if common_steps and len(common_steps) >= 2:
        lines.append("## Token Efficiency\n")
        lines.append("MoDA processes fewer tokens per step due to depth attention overhead,")
        lines.append("so the key question is whether it achieves better val_bpb *per step*.\n")
        last = common_steps[-1]
        bv = b["val_by_step"][last]
        mv = m["val_by_step"][last]
        if mv < bv:
            lines.append(f"At step {last}: MoDA val_bpb ({mv:.4f}) < Baseline ({bv:.4f}) — ")
            lines.append("MoDA learns more efficiently per gradient step.\n")
        elif mv > bv:
            lines.append(f"At step {last}: MoDA val_bpb ({mv:.4f}) > Baseline ({bv:.4f}) — ")
            lines.append("Baseline is more efficient per gradient step.\n")
        else:
            lines.append(f"At step {last}: Both models have similar val_bpb ({bv:.4f}).\n")

    # All baseline val entries
    if baseline_val:
        lines.append("## All Baseline Validation Points\n")
        lines.append("| Step | val_bpb |")
        lines.append("|------|---------|")
        for e in baseline_val:
            lines.append(f"| {e['step']} | {e['val_bpb']:.4f} |")
        lines.append("")

    # All MoDA val entries
    if moda_val:
        lines.append("## All MoDA Validation Points\n")
        lines.append("| Step | val_bpb |")
        lines.append("|------|---------|")
        for e in moda_val:
            lines.append(f"| {e['step']} | {e['val_bpb']:.4f} |")
        lines.append("")

    md = "\n".join(lines)
    out_path = script_dir / "results.md"
    with open(out_path, "w") as f:
        f.write(md)
    print(f"Results written to {out_path}")
    print(md)


if __name__ == "__main__":
    main()
