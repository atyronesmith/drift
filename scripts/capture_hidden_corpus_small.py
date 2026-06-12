#!/usr/bin/env python3
"""Capture small-model hidden states for cross-model probe transfer (Phase 4).

Loads a small MLX model in-process (not via the server, which serves the
32B) and captures all-layer last-input-token hidden states for the same
tickets as the 32B corpus, using the same prompt convention.

Probes trained on these features against the 32B's output labels test
whether large-model decisions are linearly decodable from small-model
representations — if so, runtime = small-model prefill + matmul.

Resumable: existing npz files are skipped.

Usage:
    python scripts/capture_hidden_corpus_small.py --tickets 500 \
        --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "helpdesk_tickets.jsonl"
OUT_ROOT = PROJECT_ROOT / "results" / "hidden_probe_corpus"


def format_ticket(ticket: dict) -> str:
    return "\n".join([
        f"Ticket ID:    {ticket['id']}",
        f"Subject:      {ticket['subject']}",
        f"From:         {ticket['requesterEmail']}",
        f"Created:      {ticket['createdAt']}",
        f"Priority:     {ticket['priority']}",
        f"Category:     {ticket['category']}",
        "",
        "Description:",
        ticket["description"],
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickets", type=int, default=500)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    args = parser.parse_args()

    from mlx_lm import load

    from drift.probing.mlx_hidden import _probe_hidden_with_model

    system_prompt = (PROJECT_ROOT / "tasks" / "helpdesk_triage" / "triage_system.md").read_text()
    tickets = [json.loads(line) for line in DATASET_PATH.read_text().splitlines()][:args.tickets]

    model_slug = re.sub(r"[^\w.-]", "_", args.model.split("/")[-1])
    out_dir = OUT_ROOT / f"hidden_{model_slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...")
    mlx_model, tokenizer = load(args.model)

    times = []
    shape = None
    for i, ticket in enumerate(tickets):
        tid = ticket["id"]
        npz = out_dir / f"{tid}.npz"
        if npz.exists():
            continue
        t0 = time.time()
        hs = _probe_hidden_with_model(
            mlx_model, tokenizer,
            system=system_prompt, prompt=format_ticket(ticket),
            model_name=args.model, layers="all", positions="last_input",
            prefill_only=True,
        )
        ms = (time.time() - t0) * 1000
        times.append(ms)
        pos = hs.positions[0]
        arr = np.array([hs.vector(layer, pos) for layer in hs.layers], dtype=np.float16)
        shape = arr.shape
        np.savez_compressed(npz, states=arr)
        if (i + 1) % 25 == 0 or i == 0:
            print(f"[{i+1:>3}/{len(tickets)}] {tid} {ms:,.0f}ms [{arr.shape[0]}x{arr.shape[1]}]")

    meta = {
        "model": args.model,
        "positions": "last_input",
        "layers": "all",
        "prompt_convention": "system=triage_system.md, user=format_ticket(ticket)",
        "mean_capture_ms": round(sum(times) / len(times)) if times else None,
        "n_tickets": len(tickets),
        "shape": list(shape) if shape is not None else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone. {len(times)} new captures -> {out_dir}")


if __name__ == "__main__":
    main()
