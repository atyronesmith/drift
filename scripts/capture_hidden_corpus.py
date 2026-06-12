#!/usr/bin/env python3
"""Capture a hidden-probe training corpus from the drift MLX server.

For each helpdesk ticket:
  1. Labels: full triage output via /v1/chat/completions (rendered task prompts)
  2. Features: hidden states at ALL layers, last input token, via /v1/hidden
     using the runtime convention (system = task system prompt, user = raw
     formatted ticket text) — must match run_hybrid's hidden fetch exactly.

Resumable: already-captured ticket ids are skipped on re-run.

Output (results/hidden_probe_corpus/):
  labels.jsonl        — {"ticket_id", "index", "output": <parsed JSON>}
  hidden/<id>.npz     — states float16 [n_layers, d_model]
  meta.json           — model, n_layers, d_model, prompt convention, timings

Usage:
    python scripts/capture_hidden_corpus.py --tickets 100 [--base-url http://localhost:8080]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from drift.probing.mlx_hidden import probe_hidden_remote

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "helpdesk_tickets.jsonl"
OUT_DIR = PROJECT_ROOT / "results" / "hidden_probe_corpus"


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


def chat_completion(base_url: str, system: str, user: str, timeout: int = 600) -> str:
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 1024,
    }
    result = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{base_url}/v1/chat/completions",
         "-H", "Content-Type: application/json", "-d", json.dumps(payload)],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl exit {result.returncode}: {result.stderr}")
    data = json.loads(result.stdout)
    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")
    return data["choices"][0]["message"]["content"]


def parse_json_output(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickets", type=int, default=100)
    parser.add_argument("--base-url", default="http://localhost:8080")
    args = parser.parse_args()

    system_prompt = (PROJECT_ROOT / "tasks" / "helpdesk_triage" / "triage_system.md").read_text()
    user_template = (PROJECT_ROOT / "tasks" / "helpdesk_triage" / "triage_user.md").read_text()

    tickets = [json.loads(line) for line in DATASET_PATH.read_text().splitlines()][:args.tickets]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hidden_dir = OUT_DIR / "hidden"
    hidden_dir.mkdir(exist_ok=True)
    labels_path = OUT_DIR / "labels.jsonl"

    done = set()
    if labels_path.exists():
        for line in labels_path.read_text().splitlines():
            done.add(json.loads(line)["ticket_id"])

    meta = {
        "model": "mlx-community/Qwen2.5-32B-Instruct-4bit",
        "prompt_convention": "hidden: system=triage_system.md, user=format_ticket(ticket); "
                             "labels: system=triage_system.md, user=triage_user.md{input}",
        "positions": "last_input",
        "layers": "all",
    }

    label_ms, hidden_ms = [], []
    with open(labels_path, "a") as labels_out:
        for i, ticket in enumerate(tickets):
            tid = ticket["id"]
            if tid in done and (hidden_dir / f"{tid}.npz").exists():
                continue

            input_text = format_ticket(ticket)
            print(f"[{i+1:>3}/{len(tickets)}] {tid} ...", end="", flush=True)

            # Labels (skip if already present from a previous partial run)
            if tid not in done:
                t0 = time.time()
                content = chat_completion(args.base_url, system_prompt,
                                          user_template.replace("{input}", input_text))
                lm = (time.time() - t0) * 1000
                label_ms.append(lm)
                output = parse_json_output(content)
                labels_out.write(json.dumps({
                    "ticket_id": tid, "index": i,
                    "output": output, "raw": None if output else content,
                }) + "\n")
                labels_out.flush()
                print(f" labels {lm:,.0f}ms", end="", flush=True)

            # Hidden states (runtime convention: user = raw input_text)
            if not (hidden_dir / f"{tid}.npz").exists():
                t0 = time.time()
                hs = probe_hidden_remote(
                    system=system_prompt, prompt=input_text,
                    base_url=args.base_url, layers="all", positions="last_input",
                )
                hm = (time.time() - t0) * 1000
                hidden_ms.append(hm)
                pos = hs.positions[0]
                arr = np.array(
                    [hs.vector(layer, pos) for layer in hs.layers], dtype=np.float16
                )
                np.savez_compressed(hidden_dir / f"{tid}.npz", states=arr)
                meta["n_layers"] = len(hs.layers)
                meta["d_model"] = arr.shape[1]
                meta["n_input_example"] = hs.n_input
                print(f" hidden {hm:,.0f}ms [{arr.shape[0]}x{arr.shape[1]}]")
            else:
                print()

    if label_ms:
        meta["mean_label_ms"] = round(sum(label_ms) / len(label_ms))
    if hidden_ms:
        meta["mean_hidden_ms"] = round(sum(hidden_ms) / len(hidden_ms))
    meta["n_tickets"] = len(tickets)
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone. Corpus in {OUT_DIR}")


if __name__ == "__main__":
    main()
