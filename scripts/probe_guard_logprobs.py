#!/usr/bin/env python3
"""Probe Llama Guard with logprobs to extract classification confidence.

For each of the 224 audited LLM outputs, re-runs Llama Guard with logprobs
enabled. The first token is literally "safe" or "unsafe", so its logprob
gives the classification probability directly.

Output: results/guard_logprobs.json — 224 records with P(safe), P(unsafe),
entropy, and original audit labels.

Usage:
    uv run python scripts/probe_guard_logprobs.py
    uv run python scripts/probe_guard_logprobs.py --max-outputs 10  # test run
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

from drift.safety.guard import GUARD_CATEGORIES


def load_full_content(results_dir: Path) -> dict[tuple, str]:
    """Load full LLM output content keyed by (experiment, timestamp, run_index, step)."""
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
                        key = (exp_name, ts_dir.name, run_idx, run_type)
                        content_map[key] = content

                elif run_type == "pipeline":
                    for step in run.get("steps", []):
                        if isinstance(step, dict):
                            name = step.get("name", "unknown")
                            content = step.get("response", {}).get("content", "")
                            if content:
                                key = (exp_name, ts_dir.name, run_idx, name)
                                content_map[key] = content
                        elif isinstance(step, list):
                            for iteration in step:
                                if isinstance(iteration, list):
                                    for s in iteration:
                                        name = s.get("name", "unknown")
                                        content = s.get("response", {}).get("content", "")
                                        if content:
                                            key = (exp_name, ts_dir.name, run_idx, name)
                                            content_map[key] = content

    return content_map


def check_with_logprobs(content: str, model: str, base_url: str) -> dict:
    """Call Llama Guard with logprobs enabled, return confidence data."""
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

    # Parse first token logprobs — this is the safe/unsafe decision
    p_safe = 0.0
    p_unsafe = 0.0
    first_token = ""
    first_token_logprob = 0.0
    top_logprobs = []

    if logprobs_data:
        entry = logprobs_data[0]
        first_token = entry["token"].strip().lower()
        first_token_logprob = entry["logprob"]

        # Collect all probabilities from this position
        all_tokens = [{"token": entry["token"], "logprob": entry["logprob"]}]
        for alt in entry.get("top_logprobs", []):
            all_tokens.append({"token": alt["token"], "logprob": alt["logprob"]})

        top_logprobs = all_tokens

        # Find safe/unsafe probabilities
        for t in all_tokens:
            tok = t["token"].strip().lower()
            prob = math.exp(t["logprob"])
            if tok == "safe":
                p_safe = prob
            elif tok == "unsafe":
                p_unsafe = prob

    # Shannon entropy of safe/unsafe decision
    entropy = 0.0
    if p_safe > 0 and p_unsafe > 0:
        total = p_safe + p_unsafe
        ps = p_safe / total
        pu = p_unsafe / total
        entropy = -(ps * math.log2(ps) + pu * math.log2(pu))

    # Parse verdict
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
    parser = argparse.ArgumentParser(description="Probe Llama Guard with logprobs")
    parser.add_argument("--results-dir", default="results",
                        help="Path to results directory")
    parser.add_argument("--audit-file", default="results/safety_audit.json",
                        help="Existing safety audit JSON")
    parser.add_argument("--model", default="llama-guard3:1b",
                        help="Guard model name on Ollama")
    parser.add_argument("--base-url", default=None,
                        help="Ollama base URL")
    parser.add_argument("--max-outputs", type=int, default=0,
                        help="Max outputs to probe (0 = all)")
    parser.add_argument("--output", "-o", default="results/guard_logprobs.json",
                        help="Output file")
    args = parser.parse_args()

    base_url = (args.base_url or os.environ.get(
        "OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    # Load existing audit for labels
    audit_path = Path(args.audit_file)
    if not audit_path.is_absolute():
        audit_path = Path(__file__).parent.parent / audit_path
    audit_records = json.loads(audit_path.read_text())
    print(f"Loaded {len(audit_records)} audit records from {audit_path}")

    # Load full content from result files
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = Path(__file__).parent.parent / results_dir
    content_map = load_full_content(results_dir)
    print(f"Loaded full content for {len(content_map)} outputs")

    # Match audit records with full content
    matched = 0
    unmatched = 0
    records_to_probe = []

    for rec in audit_records:
        key = (rec["experiment"], rec["timestamp"], rec["run_index"], rec["step"])
        full_content = content_map.get(key)
        if full_content:
            records_to_probe.append({**rec, "full_content": full_content})
            matched += 1
        else:
            unmatched += 1

    print(f"Matched: {matched}, Unmatched: {unmatched}")

    if args.max_outputs:
        records_to_probe = records_to_probe[:args.max_outputs]

    total = len(records_to_probe)
    print(f"\nProbing {total} outputs with logprobs using {args.model}...")

    results = []
    start_all = time.perf_counter()

    for i, rec in enumerate(records_to_probe):
        exp = rec["experiment"]
        step = rec["step"]
        content_len = len(rec["full_content"])

        print(f"  [{i+1}/{total}] {exp} step={step} ({content_len} chars)...",
              end="", flush=True)

        try:
            logprob_result = check_with_logprobs(
                rec["full_content"], args.model, base_url)

            verdict = "SAFE" if logprob_result["safe"] else f"UNSAFE ({logprob_result['category']})"
            conf = max(logprob_result["p_safe"], logprob_result["p_unsafe"])
            print(f" {verdict} P={conf:.4f} H={logprob_result['entropy']:.3f} "
                  f"({logprob_result['logprob_latency_ms']:.0f}ms)")

            results.append({
                "experiment": rec["experiment"],
                "timestamp": rec["timestamp"],
                "run_index": rec["run_index"],
                "step": rec["step"],
                "content_length": content_len,
                # Original audit labels
                "audit_safe": rec["safe"],
                "audit_category": rec["category"],
                "audit_guard_latency_ms": rec["guard_latency_ms"],
                # Logprob results
                **logprob_result,
            })

        except Exception as e:
            print(f" ERROR: {e}")
            results.append({
                "experiment": rec["experiment"],
                "timestamp": rec["timestamp"],
                "run_index": rec["run_index"],
                "step": rec["step"],
                "content_length": content_len,
                "audit_safe": rec["safe"],
                "audit_category": rec["category"],
                "audit_guard_latency_ms": rec["guard_latency_ms"],
                "error": str(e),
            })

    elapsed = time.perf_counter() - start_all

    # Summary
    probed = [r for r in results if "error" not in r]
    errors = len(results) - len(probed)
    print(f"\nProbed {len(probed)} outputs ({errors} errors) in {elapsed:.1f}s")

    if probed:
        entropies = [r["entropy"] for r in probed]
        p_safes = [r["p_safe"] for r in probed]
        p_unsafes = [r["p_unsafe"] for r in probed]
        high_conf = sum(1 for r in probed if max(r["p_safe"], r["p_unsafe"]) >= 0.90)

        print(f"\nEntropy: mean={sum(entropies)/len(entropies):.4f}, "
              f"min={min(entropies):.4f}, max={max(entropies):.4f}")
        print(f"P(safe):   mean={sum(p_safes)/len(p_safes):.4f}")
        print(f"P(unsafe): mean={sum(p_unsafes)/len(p_unsafes):.4f}")
        print(f"High confidence (>=0.90): {high_conf}/{len(probed)} "
              f"({high_conf/len(probed)*100:.1f}%)")

        # Check agreement with original audit
        agree = sum(1 for r in probed if r["safe"] == r["audit_safe"])
        print(f"Agreement with original audit: {agree}/{len(probed)} "
              f"({agree/len(probed)*100:.1f}%)")

    # Save
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent.parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
