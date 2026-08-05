#!/usr/bin/env python3
"""Generate LLM summaries for Ansible error logs.

Produces individual summary .txt files for use as drift inputs.

Usage:
    uv run python scripts/generate_summaries.py \
        --log-dir datasets/ansible_triage/inputs/ \
        --n 50 \
        -o datasets/ansible_summaries/
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

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

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(payload)
        tmp_path = f.name

    start = time.time()
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
        return "", 600.0
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    elapsed = time.time() - start

    if result.returncode != 0:
        return "", elapsed

    try:
        resp = json.loads(result.stdout)
        text = resp.get("message", {}).get("content", "")
        total_ns = resp.get("total_duration", 0)
        if total_ns > 0:
            elapsed = total_ns / 1e9
        return text.strip(), elapsed
    except json.JSONDecodeError:
        return "", elapsed


def main():
    parser = argparse.ArgumentParser(description="Generate summaries for Ansible logs")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    parser.add_argument("--predigest", action="store_true",
                        help="Pre-digest logs before summarizing")
    args = parser.parse_args()

    if args.predigest:
        sys.path.insert(0, str(Path(__file__).parent))
        from ansible_log_predigest import predigest

    log_dir = Path(args.log_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_logs = sorted(log_dir.glob("*.txt"), key=lambda f: f.stat().st_size)
    n = min(args.n, len(all_logs))
    step = max(1, len(all_logs) // n)
    log_files = [all_logs[i * step] for i in range(n)]

    print(f"Generating summaries for {len(log_files)} logs via {OLLAMA_MODEL}")
    print(f"Output: {out_dir}")
    if args.predigest:
        print("Pre-digestion: enabled")
    print(f"{'='*60}\n")

    success = 0
    for idx, filepath in enumerate(log_files):
        if args.predigest:
            text = predigest(filepath)
        else:
            text = filepath.read_text(errors="replace")

        chars = len(text)
        print(f"[{idx+1}/{len(log_files)}] {filepath.name} ({chars:,} chars)...",
              end="", flush=True)

        user = USER_TEMPLATE.format(error_log=text)
        summary, elapsed = call_ollama(SYSTEM_PROMPT, user, args.ollama_host)

        if summary:
            out_file = out_dir / f"{filepath.stem}.txt"
            out_file.write_text(summary)
            success += 1
            print(f" {elapsed:.1f}s — {summary[:80]}")
        else:
            print(f" FAILED ({elapsed:.1f}s)")

    print(f"\n{'='*60}")
    print(f"Generated {success}/{len(log_files)} summaries in {out_dir}")


if __name__ == "__main__":
    main()
