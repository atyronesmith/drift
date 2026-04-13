#!/usr/bin/env python3
"""Probe new LLM outputs through Llama Guard to expand the training set.

Finds all outputs in results/ that aren't in the existing guard_logprobs.json
and probes them through Llama Guard with logprobs. Merges results into the
existing dataset.

Usage:
    uv run python scripts/probe_new_guard_outputs.py
    uv run python scripts/probe_new_guard_outputs.py --max-outputs 5  # test run
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_full_content(results_dir: Path) -> dict[tuple, str]:
    """Load all LLM output content keyed by (experiment, timestamp, run_index, step)."""
    content_map = {}

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("__"):
            continue

        for ts_dir in sorted(exp_dir.iterdir()):
            if not ts_dir.is_dir():
                continue

            for run_file in sorted(ts_dir.glob("run_*.json")):
                try:
                    run = json.loads(run_file.read_text())
                except (json.JSONDecodeError, OSError):
                    continue

                run_idx = run.get("run_index", 0)
                run_type = run.get("type", "unknown")
                exp_name = exp_dir.name

                if run_type in ("baseline", "hybrid"):
                    content = run.get("response", {}).get("content", "")
                    if content:
                        content_map[(exp_name, ts_dir.name, run_idx, run_type)] = content

                elif run_type == "pipeline":
                    for step in run.get("steps", []):
                        if isinstance(step, dict):
                            name = step.get("name", "unknown")
                            content = step.get("response", {}).get("content", "")
                            if content:
                                content_map[(exp_name, ts_dir.name, run_idx, name)] = content
                        elif isinstance(step, list):
                            for iteration in step:
                                if isinstance(iteration, list):
                                    for s in iteration:
                                        name = s.get("name", "unknown")
                                        content = s.get("response", {}).get("content", "")
                                        if content:
                                            content_map[(exp_name, ts_dir.name, run_idx, name)] = content

    return content_map


def check_with_logprobs(content: str, model: str, base_url: str) -> dict:
    """Call Llama Guard with logprobs enabled."""
    messages = [
        {"role": "user", "content": "Analyze this output."},
        {"role": "assistant", "content": content},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 5,
    }

    url = f"{base_url}/v1/chat/completions"
    start = time.perf_counter()

    result = subprocess.run(
        ["curl", "-s", "-X", "POST", url,
         "-H", "Content-Type: application/json",
         "-d", json.dumps(payload)],
        capture_output=True, text=True, timeout=120,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    if result.returncode != 0:
        raise RuntimeError(f"curl exit {result.returncode}: {result.stderr}")

    data = json.loads(result.stdout)
    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")

    choice = data["choices"][0]
    response_text = choice["message"]["content"].strip()
    logprobs_data = choice.get("logprobs", {}).get("content", [])

    p_safe = 0.0
    p_unsafe = 0.0
    first_token = ""
    first_token_logprob = 0.0
    top_logprobs = []

    if logprobs_data:
        entry = logprobs_data[0]
        first_token = entry["token"].strip().lower()
        first_token_logprob = entry["logprob"]

        all_tokens = [{"token": entry["token"], "logprob": entry["logprob"]}]
        for alt in entry.get("top_logprobs", []):
            all_tokens.append({"token": alt["token"], "logprob": alt["logprob"]})

        top_logprobs = all_tokens

        for t in all_tokens:
            tok = t["token"].strip().lower()
            prob = math.exp(t["logprob"])
            if tok == "safe":
                p_safe = prob
            elif tok == "unsafe":
                p_unsafe = prob

    entropy = 0.0
    if p_safe > 0 and p_unsafe > 0:
        total = p_safe + p_unsafe
        ps = p_safe / total
        pu = p_unsafe / total
        entropy = -(ps * math.log2(ps) + pu * math.log2(pu))

    lines = response_text.split("\n")
    verdict = lines[0].strip().lower()
    category = lines[1].strip() if len(lines) > 1 and verdict == "unsafe" else None

    return {
        "safe": verdict == "safe",
        "category": category,
        "logprob_latency_ms": round(latency_ms, 1),
        "p_safe": round(p_safe, 6),
        "p_unsafe": round(p_unsafe, 6),
        "first_token": first_token,
        "first_token_logprob": round(first_token_logprob, 4),
        "top_logprobs": top_logprobs,
        "entropy": round(entropy, 4),
        "guard_response": response_text,
    }


def main():
    parser = argparse.ArgumentParser(description="Probe new outputs through Llama Guard")
    parser.add_argument("--results-dir", default="results",
                        help="Results directory")
    parser.add_argument("--existing", default="results/guard_logprobs.json",
                        help="Existing logprobs JSON to merge with")
    parser.add_argument("--model", default="llama-guard3:1b",
                        help="Guard model name")
    parser.add_argument("--base-url", default=None,
                        help="Ollama base URL")
    parser.add_argument("--max-outputs", type=int, default=0,
                        help="Max new outputs to probe (0 = all)")
    parser.add_argument("--output", "-o", default="results/guard_logprobs.json",
                        help="Output file (merges with existing)")
    args = parser.parse_args()

    base_url = (args.base_url or os.environ.get(
        "OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = Path(__file__).parent.parent / results_dir

    # Load existing logprobs
    existing_path = Path(args.existing)
    if not existing_path.is_absolute():
        existing_path = Path(__file__).parent.parent / existing_path
    existing = json.loads(existing_path.read_text()) if existing_path.exists() else []
    existing_keys = {
        (r["experiment"], r["timestamp"], r["run_index"], r["step"])
        for r in existing
    }
    print(f"Existing logprobs: {len(existing)} records")

    # Load all content
    content_map = load_full_content(results_dir)
    print(f"Total outputs in results/: {len(content_map)}")

    # Find new outputs
    new_outputs = []
    for key, content in sorted(content_map.items()):
        if key not in existing_keys:
            new_outputs.append((key, content))

    print(f"New outputs to probe: {len(new_outputs)}")

    if args.max_outputs:
        new_outputs = new_outputs[:args.max_outputs]

    if not new_outputs:
        print("No new outputs to probe.")
        return

    total = len(new_outputs)
    print(f"\nProbing {total} new outputs with {args.model}...")

    new_records = []
    for i, (key, content) in enumerate(new_outputs):
        exp, ts, run_idx, step = key
        content_len = len(content)

        print(f"  [{i+1}/{total}] {exp} step={step} ({content_len} chars)...",
              end="", flush=True)

        try:
            result = check_with_logprobs(content, args.model, base_url)

            verdict = "SAFE" if result["safe"] else f"UNSAFE ({result['category']})"
            conf = max(result["p_safe"], result["p_unsafe"])
            print(f" {verdict} P={conf:.4f} H={result['entropy']:.3f} "
                  f"({result['logprob_latency_ms']:.0f}ms)")

            new_records.append({
                "experiment": exp,
                "timestamp": ts,
                "run_index": run_idx,
                "step": step,
                **result,
            })

        except Exception as e:
            print(f" ERROR: {e}")
            new_records.append({
                "experiment": exp,
                "timestamp": ts,
                "run_index": run_idx,
                "step": step,
                "error": str(e),
            })

    # Merge and save
    merged = existing + new_records
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent.parent / output_path
    output_path.write_text(json.dumps(merged, indent=2))

    valid = [r for r in new_records if "error" not in r]
    safe_count = sum(1 for r in valid if r.get("safe"))
    unsafe_count = len(valid) - safe_count

    print(f"\nResults: {len(valid)} probed ({safe_count} safe, {unsafe_count} unsafe)")
    print(f"Merged total: {len(merged)} records → {output_path}")


if __name__ == "__main__":
    main()
