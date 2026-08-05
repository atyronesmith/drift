#!/usr/bin/env python3
"""Measure pre-digestion token savings across the full corpus.

No LLM calls — just measures raw vs digested sizes for all logs.
Fast: processes 500 logs in seconds.

Usage:
    uv run python scripts/predigest_corpus_stats.py \
        --log-dir datasets/ansible_triage/inputs/ \
        -o results/predigest_corpus_stats.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ansible_log_predigest import predigest


def main():
    parser = argparse.ArgumentParser(description="Corpus-wide pre-digestion stats")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_files = sorted(log_dir.glob("*.txt"))

    print(f"Analyzing {len(log_files)} logs from {log_dir}")
    print(f"{'='*70}\n")

    results = []
    total_raw_chars = 0
    total_digest_chars = 0
    total_raw_lines = 0
    total_digest_lines = 0

    for idx, filepath in enumerate(log_files):
        raw_text = filepath.read_text(errors="replace")
        raw_lines = len(raw_text.splitlines())
        raw_chars = len(raw_text)

        digested_text = predigest(filepath)
        digest_lines = len(digested_text.splitlines())
        digest_chars = len(digested_text)

        raw_tokens = raw_chars // 4
        digest_tokens = digest_chars // 4
        token_savings = round((1 - digest_tokens / raw_tokens) * 100, 1) if raw_tokens > 0 else 0
        line_reduction = round((1 - digest_lines / raw_lines) * 100, 1) if raw_lines > 0 else 0

        total_raw_chars += raw_chars
        total_digest_chars += digest_chars
        total_raw_lines += raw_lines
        total_digest_lines += digest_lines

        results.append({
            "file": filepath.name,
            "raw_lines": raw_lines,
            "digest_lines": digest_lines,
            "raw_chars": raw_chars,
            "digest_chars": digest_chars,
            "raw_tokens_est": raw_tokens,
            "digest_tokens_est": digest_tokens,
            "token_savings_pct": token_savings,
            "line_reduction_pct": line_reduction,
        })

        if (idx + 1) % 50 == 0 or idx + 1 == len(log_files):
            running_savings = round((1 - total_digest_chars / total_raw_chars) * 100, 1)
            print(f"  [{idx+1:3d}/{len(log_files)}] running token savings: {running_savings}%")

    # Summary stats
    total_raw_tokens = total_raw_chars // 4
    total_digest_tokens = total_digest_chars // 4
    overall_savings = round((1 - total_digest_tokens / total_raw_tokens) * 100, 1)
    overall_line_reduction = round((1 - total_digest_lines / total_raw_lines) * 100, 1)

    per_log_savings = [r["token_savings_pct"] for r in results]
    per_log_savings.sort()
    median_savings = per_log_savings[len(per_log_savings) // 2]
    p10_savings = per_log_savings[len(per_log_savings) // 10]
    p90_savings = per_log_savings[int(len(per_log_savings) * 0.9)]
    min_savings = per_log_savings[0]
    max_savings = per_log_savings[-1]

    # Size buckets
    small = [r for r in results if r["raw_chars"] < 5000]
    medium = [r for r in results if 5000 <= r["raw_chars"] < 50000]
    large = [r for r in results if r["raw_chars"] >= 50000]

    def bucket_savings(bucket):
        if not bucket:
            return 0, 0, 0
        raw = sum(r["raw_tokens_est"] for r in bucket)
        dig = sum(r["digest_tokens_est"] for r in bucket)
        return len(bucket), raw, round((1 - dig / raw) * 100, 1) if raw else 0

    print(f"\n{'='*70}")
    print(f"CORPUS STATISTICS — {len(log_files)} logs")
    print(f"{'='*70}")
    print(f"Total raw tokens:      ~{total_raw_tokens:,}")
    print(f"Total digest tokens:   ~{total_digest_tokens:,}")
    print(f"Overall token savings: {overall_savings}%")
    print(f"Overall line reduction:{overall_line_reduction}%")
    print(f"")
    print(f"--- Per-Log Distribution ---")
    print(f"Min savings:   {min_savings}%")
    print(f"P10 savings:   {p10_savings}%")
    print(f"Median savings:{median_savings}%")
    print(f"P90 savings:   {p90_savings}%")
    print(f"Max savings:   {max_savings}%")
    print(f"")
    print(f"--- By Size Bucket ---")
    for name, bucket in [("Small (<5KB)", small), ("Medium (5-50KB)", medium), ("Large (>50KB)", large)]:
        n, raw, sav = bucket_savings(bucket)
        print(f"{name:20s}: {n:3d} logs, ~{raw:,} tokens, {sav}% savings")

    # Save
    output = {
        "n_logs": len(log_files),
        "total_raw_tokens": total_raw_tokens,
        "total_digest_tokens": total_digest_tokens,
        "overall_token_savings_pct": overall_savings,
        "overall_line_reduction_pct": overall_line_reduction,
        "per_log_min": min_savings,
        "per_log_p10": p10_savings,
        "per_log_median": median_savings,
        "per_log_p90": p90_savings,
        "per_log_max": max_savings,
        "buckets": {
            "small": {"n": len(small), "savings_pct": bucket_savings(small)[2]},
            "medium": {"n": len(medium), "savings_pct": bucket_savings(medium)[2]},
            "large": {"n": len(large), "savings_pct": bucket_savings(large)[2]},
        },
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
