#!/usr/bin/env python3
"""Experiment 19: Measure pod log triage — raw vs pre-digested.

Runs the pod_log_triage system prompt against both raw and pre-digested
versions of the input, measuring token savings, latency, and functional
test pass rates.

Usage:
    uv run python scripts/podlog_predigest_measure.py \
        --ollama-host http://localhost:11434 \
        --runs 5 \
        -o results/exp19-predigest-podlog/metrics.json
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Add logdigest to path (checkout directory is still named "denoise" locally)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "denoise" / "src"))
from logdigest.mustgather import predigest

OLLAMA_MODEL = "qwen2.5:32b"

TASK_DIR = Path(__file__).resolve().parent.parent / "tasks" / "pod_log_triage"
SYSTEM_PROMPT = (TASK_DIR / "podlog_system.md").read_text()
USER_TEMPLATE = (TASK_DIR / "podlog_user.md").read_text()
INPUT_TEXT = (TASK_DIR / "input.md").read_text()

# Load functional tests
spec = importlib.util.spec_from_file_location("functional_tests", TASK_DIR / "functional_tests.py")
ft_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft_mod)


def call_ollama(system: str, user: str, host: str) -> tuple[str, float]:
    """Call Ollama and return (response_text, elapsed_seconds)."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 1024},
    })

    start = time.time()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(payload)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["curl", "-s", "-X", "POST",
             f"{host}/api/chat",
             "-H", "Content-Type: application/json",
             "-d", f"@{tmp_path}"],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        os.unlink(tmp_path)
        return "[TIMEOUT after 600s]", 600.0
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  curl error: {result.stderr}", file=sys.stderr)
        return "", elapsed

    try:
        resp = json.loads(result.stdout)
        text = resp.get("message", {}).get("content", "")
        total_ns = resp.get("total_duration", 0)
        if total_ns > 0:
            elapsed = total_ns / 1e9
        return text.strip(), elapsed
    except json.JSONDecodeError:
        print(f"  JSON parse error: {result.stdout[:200]}", file=sys.stderr)
        return "", elapsed


def run_functional_tests(output: str) -> dict:
    """Run functional tests against LLM output. Returns {name: pass/fail}."""
    data = ft_mod.extract(output)
    results = {}
    for name, test_fn in ft_mod.tests.items():
        try:
            test_fn(data)
            results[name] = "pass"
        except AssertionError:
            results[name] = "fail"
    return results


def main():
    parser = argparse.ArgumentParser(description="Exp 19: raw vs pre-digested pod log triage")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per variant")
    parser.add_argument("--ollama-host", default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--output", "-o", help="Save results JSON")
    args = parser.parse_args()

    # Pre-digest the input
    digested_text = predigest(INPUT_TEXT)

    raw_chars = len(INPUT_TEXT)
    raw_lines = len(INPUT_TEXT.splitlines())
    digest_chars = len(digested_text)
    digest_lines = len(digested_text.splitlines())
    raw_tokens_est = raw_chars // 4
    digest_tokens_est = digest_chars // 4
    token_savings_pct = round((1 - digest_tokens_est / raw_tokens_est) * 100, 1) if raw_tokens_est > 0 else 0

    print(f"Input: {raw_lines} lines, ~{raw_tokens_est:,} tokens")
    print(f"Digested: {digest_lines} lines, ~{digest_tokens_est:,} tokens ({token_savings_pct}% saved)")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Runs per variant: {args.runs}")
    print(f"{'=' * 70}\n")

    # Show what pre-digestion removed
    print("--- Pre-digested output ---")
    for line in digested_text.splitlines()[:30]:
        print(f"  {line}")
    if digest_lines > 30:
        print(f"  ... ({digest_lines - 30} more lines)")
    print(f"{'=' * 70}\n")

    raw_results = []
    digest_results = []

    for run_idx in range(args.runs):
        print(f"--- Run {run_idx + 1}/{args.runs} ---")

        # Raw
        user_raw = USER_TEMPLATE.replace("{input}", INPUT_TEXT)
        print(f"  Raw ({raw_chars:,} chars)...", end="", flush=True)
        raw_output, raw_elapsed = call_ollama(SYSTEM_PROMPT, user_raw, args.ollama_host)
        raw_ft = run_functional_tests(raw_output)
        raw_pass = sum(1 for v in raw_ft.values() if v == "pass")
        print(f" {raw_elapsed:.1f}s, {raw_pass}/{len(raw_ft)} tests pass")
        raw_results.append({
            "output": raw_output,
            "elapsed": round(raw_elapsed, 2),
            "tests": raw_ft,
            "tests_passed": raw_pass,
            "tests_total": len(raw_ft),
        })

        # Pre-digested
        user_digest = USER_TEMPLATE.replace("{input}", digested_text)
        print(f"  Digested ({digest_chars:,} chars)...", end="", flush=True)
        digest_output, digest_elapsed = call_ollama(SYSTEM_PROMPT, user_digest, args.ollama_host)
        digest_ft = run_functional_tests(digest_output)
        digest_pass = sum(1 for v in digest_ft.values() if v == "pass")
        print(f" {digest_elapsed:.1f}s, {digest_pass}/{len(digest_ft)} tests pass")
        digest_results.append({
            "output": digest_output,
            "elapsed": round(digest_elapsed, 2),
            "tests": digest_ft,
            "tests_passed": digest_pass,
            "tests_total": len(digest_ft),
        })

        speedup = round((raw_elapsed - digest_elapsed) / raw_elapsed * 100, 1) if raw_elapsed > 0 else 0
        print(f"  Speedup: {speedup}% ({raw_elapsed:.1f}s → {digest_elapsed:.1f}s)")
        print()

    # Summary
    n = args.runs
    avg_raw_time = sum(r["elapsed"] for r in raw_results) / n
    avg_digest_time = sum(r["elapsed"] for r in digest_results) / n
    avg_speedup = round((avg_raw_time - avg_digest_time) / avg_raw_time * 100, 1) if avg_raw_time > 0 else 0

    avg_raw_pass = sum(r["tests_passed"] for r in raw_results) / n
    avg_digest_pass = sum(r["tests_passed"] for r in digest_results) / n

    # Per-test breakdown
    test_names = list(ft_mod.tests.keys())
    per_test = {}
    for name in test_names:
        raw_rate = sum(1 for r in raw_results if r["tests"][name] == "pass") / n
        digest_rate = sum(1 for r in digest_results if r["tests"][name] == "pass") / n
        per_test[name] = {"raw": round(raw_rate, 2), "digest": round(digest_rate, 2)}

    print(f"{'=' * 70}")
    print(f"RESULTS — Experiment 19: Pod Log Pre-digestion")
    print(f"{'=' * 70}")
    print(f"Token savings:          {token_savings_pct}% ({raw_tokens_est:,} → {digest_tokens_est:,})")
    print(f"Avg raw latency:        {avg_raw_time:.1f}s")
    print(f"Avg digest latency:     {avg_digest_time:.1f}s")
    print(f"Avg speedup:            {avg_speedup}%")
    print(f"Avg raw test pass:      {avg_raw_pass:.1f}/{len(test_names)}")
    print(f"Avg digest test pass:   {avg_digest_pass:.1f}/{len(test_names)}")
    print()
    print(f"--- Per-test pass rates ---")
    for name, rates in per_test.items():
        delta = "=" if rates["raw"] == rates["digest"] else ("+" if rates["digest"] > rates["raw"] else "-")
        print(f"  {name:35s}  raw={rates['raw']:.0%}  digest={rates['digest']:.0%}  {delta}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "experiment": "exp19-predigest-podlog",
            "model": OLLAMA_MODEL,
            "n_runs": n,
            "input_raw_chars": raw_chars,
            "input_raw_lines": raw_lines,
            "input_raw_tokens_est": raw_tokens_est,
            "input_digest_chars": digest_chars,
            "input_digest_lines": digest_lines,
            "input_digest_tokens_est": digest_tokens_est,
            "token_savings_pct": token_savings_pct,
            "avg_raw_latency": round(avg_raw_time, 2),
            "avg_digest_latency": round(avg_digest_time, 2),
            "avg_speedup_pct": avg_speedup,
            "avg_raw_tests_passed": round(avg_raw_pass, 1),
            "avg_digest_tests_passed": round(avg_digest_pass, 1),
            "total_tests": len(test_names),
            "per_test_pass_rates": per_test,
            "raw_results": raw_results,
            "digest_results": digest_results,
        }
        output_path.write_text(json.dumps(output, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
