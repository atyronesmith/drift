#!/usr/bin/env python3
"""Measure LLM summarization: raw logs vs pre-digested logs.

Runs the ansible-log-analysis summarization prompt against both raw and
pre-digested versions of N logs, measuring latency and comparing outputs.

Usage:
    uv run python scripts/ansible_predigest_measure.py \
        --log-dir /path/to/logs/failed/ \
        --n 20 \
        --ollama-host http://localhost:11434
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Import pre-digestion from sibling script
sys.path.insert(0, str(Path(__file__).parent))
from ansible_log_predigest import predigest

OLLAMA_MODEL = "qwen2.5:32b"

SYSTEM_PROMPT = """You are an expert DevOps engineer specializing in Ansible automation and troubleshooting. Your task is to analyze Ansible error logs and create concise, actionable summaries that help developers and operations teams quickly understand and resolve issues.

## Summary Requirements

Create a **single-line summary** that follows this structure:
**[Error Type]: [Brief description of what failed and why]**

Your summary must identify:
1. **Error Category** - The type of failure (e.g., SSH Connection, Variable Error, AWS API Error)
2. **Failed Component** - What specific resource, task, or service failed
3. **Root Cause** - The underlying reason for the failure
4. **Context** - Where the error occurred (file, line, operation)

## Guidelines

**DO:**
- Use clear, technical language appropriate for DevOps professionals
- Include specific file names and line numbers when available
- Mention key variable names, hostnames, or resource identifiers
- Focus on the PRIMARY cause when multiple errors are present
- Use consistent error category naming
- Instead of saying the amounts of retries say 'x'
- dont say the line number in the summary.

**DON'T:**
- Include verbose stack traces or full error messages
- Add troubleshooting steps (this is summary only)
- Use vague terms like "something went wrong"
- Include multiple unrelated errors in one summary
- Exceed one concise sentence"""

USER_TEMPLATE = """ansible_log_error:
```
{error_log}
```
log_summary:"""


def call_ollama(system: str, user: str, host: str) -> tuple[str, float]:
    """Call Ollama and return (response_text, elapsed_seconds)."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 200},
    })

    # Write payload to temp file to avoid shell argument length limits
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
        # Use Ollama's own timing if available
        total_ns = resp.get("total_duration", 0)
        if total_ns > 0:
            elapsed = total_ns / 1e9
        return text.strip(), elapsed
    except json.JSONDecodeError:
        print(f"  JSON parse error: {result.stdout[:200]}", file=sys.stderr)
        return "", elapsed


def extract_error_type(summary: str) -> str:
    """Extract the error type prefix from a summary."""
    m = re.match(r"^\*?\*?\[?([^:\]]+?)[\]:]", summary.strip())
    if m:
        return m.group(1).strip().lower().rstrip("*")
    return summary.strip().lower()[:30]


def main():
    parser = argparse.ArgumentParser(
        description="Measure raw vs pre-digested summarization"
    )
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument(
        "--ollama-host", default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    parser.add_argument("--output", "-o", help="Save results JSON to file")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    all_logs = sorted(log_dir.glob("*.txt"))
    # Pick a mix of sizes: sort by size and sample evenly
    all_logs_by_size = sorted(all_logs, key=lambda f: f.stat().st_size)
    n = min(args.n, len(all_logs_by_size))
    step = max(1, len(all_logs_by_size) // n)
    log_files = [all_logs_by_size[i * step] for i in range(n)]
    host = args.ollama_host

    print(f"Measuring {len(log_files)} logs against {OLLAMA_MODEL} at {host}")
    print(f"{'='*70}\n")

    results = []
    raw_times = []
    digest_times = []
    matches = 0

    for idx, filepath in enumerate(log_files):
        raw_text = filepath.read_text(errors="replace")
        raw_lines = len(raw_text.splitlines())
        raw_chars = len(raw_text)

        digested_text = predigest(filepath)
        digest_lines = len(digested_text.splitlines())
        digest_chars = len(digested_text)

        reduction_pct = round((1 - digest_lines / raw_lines) * 100, 1) if raw_lines > 0 else 0

        print(f"[{idx+1}/{len(log_files)}] {filepath.name} "
              f"({raw_lines}→{digest_lines} lines, {reduction_pct}% reduction)")

        # Run raw
        print(f"  Raw ({raw_chars:,} chars)...", end="", flush=True)
        user_raw = USER_TEMPLATE.format(error_log=raw_text)
        raw_summary, raw_elapsed = call_ollama(SYSTEM_PROMPT, user_raw, host)
        print(f" {raw_elapsed:.1f}s")

        # Run pre-digested
        print(f"  Digested ({digest_chars:,} chars)...", end="", flush=True)
        user_digest = USER_TEMPLATE.format(error_log=digested_text)
        digest_summary, digest_elapsed = call_ollama(SYSTEM_PROMPT, user_digest, host)
        print(f" {digest_elapsed:.1f}s")

        # Compare
        raw_type = extract_error_type(raw_summary)
        digest_type = extract_error_type(digest_summary)
        type_match = raw_type == digest_type

        if type_match:
            matches += 1

        speedup = round((raw_elapsed - digest_elapsed) / raw_elapsed * 100, 1) if raw_elapsed > 0 else 0

        # Rough token estimates (~4 chars/token)
        raw_tokens_est = raw_chars // 4
        digest_tokens_est = digest_chars // 4
        token_savings_pct = round((1 - digest_tokens_est / raw_tokens_est) * 100, 1) if raw_tokens_est > 0 else 0

        print(f"  Raw:     {raw_summary[:100]}")
        print(f"  Digest:  {digest_summary[:100]}")
        print(f"  Match:   {'YES' if type_match else 'NO'} "
              f"(raw={raw_type}, digest={digest_type})")
        print(f"  Tokens:  ~{raw_tokens_est:,} → ~{digest_tokens_est:,} ({token_savings_pct}% saved)")
        print(f"  Speedup: {speedup}% ({raw_elapsed:.1f}s → {digest_elapsed:.1f}s)")
        print()

        raw_times.append(raw_elapsed)
        digest_times.append(digest_elapsed)
        results.append({
            "file": filepath.name,
            "raw_lines": raw_lines,
            "digest_lines": digest_lines,
            "raw_chars": raw_chars,
            "digest_chars": digest_chars,
            "raw_tokens_est": raw_tokens_est,
            "digest_tokens_est": digest_tokens_est,
            "token_savings_pct": token_savings_pct,
            "reduction_pct": reduction_pct,
            "raw_summary": raw_summary,
            "digest_summary": digest_summary,
            "raw_type": raw_type,
            "digest_type": digest_type,
            "type_match": type_match,
            "raw_elapsed": round(raw_elapsed, 2),
            "digest_elapsed": round(digest_elapsed, 2),
            "speedup_pct": speedup,
        })

    # Summary
    n = len(results)
    avg_raw = sum(raw_times) / n
    avg_digest = sum(digest_times) / n
    avg_speedup = round((avg_raw - avg_digest) / avg_raw * 100, 1)
    total_raw = sum(raw_times)
    total_digest = sum(digest_times)

    total_raw_tokens = sum(r["raw_tokens_est"] for r in results)
    total_digest_tokens = sum(r["digest_tokens_est"] for r in results)
    token_savings = round((1 - total_digest_tokens / total_raw_tokens) * 100, 1) if total_raw_tokens > 0 else 0
    timeouts = sum(1 for r in results if r["raw_summary"] == "[TIMEOUT after 600s]")

    print(f"{'='*70}")
    print(f"RESULTS — {n} logs")
    print(f"{'='*70}")
    print(f"Error type match:     {matches}/{n} ({round(matches/n*100, 1)}%)")
    print(f"Raw timeouts:         {timeouts}/{n}")
    print(f"Avg raw latency:      {avg_raw:.1f}s")
    print(f"Avg digest latency:   {avg_digest:.1f}s")
    print(f"Avg speedup:          {avg_speedup}%")
    print(f"Total raw time:       {total_raw:.1f}s")
    print(f"Total digest time:    {total_digest:.1f}s")
    print(f"Total time saved:     {total_raw - total_digest:.1f}s")
    print(f"")
    print(f"--- Token Savings (input) ---")
    print(f"Total raw tokens:     ~{total_raw_tokens:,}")
    print(f"Total digest tokens:  ~{total_digest_tokens:,}")
    print(f"Token savings:        {token_savings}%")

    if args.output:
        output = {
            "model": OLLAMA_MODEL,
            "n_logs": n,
            "error_type_match": matches,
            "error_type_match_pct": round(matches / n * 100, 1),
            "avg_raw_latency": round(avg_raw, 2),
            "avg_digest_latency": round(avg_digest, 2),
            "avg_speedup_pct": avg_speedup,
            "total_raw_time": round(total_raw, 2),
            "total_digest_time": round(total_digest, 2),
            "results": results,
        }
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
