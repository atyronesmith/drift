#!/usr/bin/env python3
"""Compare raw vs pre-digested summaries for quality validation.

Checks:
1. Format compliance (single-line, error type prefix)
2. Semantic match (same error type identified)
3. Specificity (mentions file names, hostnames, error codes)
4. Failure modes (timeouts, rambling, format violations)

Usage:
    uv run python scripts/predigest_quality_compare.py \
        --raw-dir datasets/ansible_summaries_raw/ \
        --digest-dir datasets/ansible_summaries/ \
        -o results/predigest_quality.json
"""

import argparse
import json
import re
from pathlib import Path


def extract_error_type(summary: str) -> str:
    """Extract the error type prefix from a summary."""
    m = re.match(r"^\*?\*?\[?([^:\]]+?)[\]:]", summary.strip())
    if m:
        return m.group(1).strip().lower().rstrip("*")
    return ""


def is_single_line(summary: str) -> bool:
    """Check if summary is essentially one line (allow minor formatting)."""
    lines = [l for l in summary.strip().splitlines() if l.strip()]
    return len(lines) <= 2


def has_specifics(summary: str) -> bool:
    """Check if summary mentions specific details (files, hosts, errors)."""
    patterns = [
        r"[a-zA-Z0-9_-]+\.(yml|yaml|py|txt|conf|cfg)",  # file names
        r"[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.internal",  # hostnames
        r"[A-Z][a-z]+Error",  # error class names
        r"'[^']+'",  # quoted identifiers
        r"`[^`]+`",  # backtick identifiers
    ]
    return any(re.search(p, summary) for p in patterns)


def is_timeout_or_ramble(summary: str) -> bool:
    """Detect if the LLM timed out or rambled instead of summarizing."""
    if "[TIMEOUT" in summary:
        return True
    if len(summary) > 500:  # single-line summaries should be concise
        return True
    if "here is a summary" in summary.lower():
        return True
    if summary.count("\n") > 5:
        return True
    return False


def normalize_error_type(t: str) -> str:
    """Normalize error types for fuzzy matching."""
    t = t.lower().strip()
    # Common synonyms
    t = t.replace("undefined variable", "variable error")
    t = t.replace("variable error", "variable")
    t = t.replace("package dependency error", "dependency")
    t = t.replace("dependency error", "dependency")
    t = t.replace("ssh connection error", "ssh")
    t = t.replace("connection error", "connection")
    t = t.replace("aws api error", "aws")
    t = t.replace("api error", "api")
    t = t.replace("kubernetes api error", "kubernetes")
    t = t.replace("kubernetes resource error", "kubernetes")
    t = t.replace("module error", "module")
    t = t.replace("role not found error", "role not found")
    t = t.replace("role not found", "role")
    t = t.replace("retry error", "retry")
    t = t.replace("retry failure", "retry")
    t = t.replace("task failure", "task")
    return t.strip()


def main():
    parser = argparse.ArgumentParser(description="Compare raw vs pre-digested summaries")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--digest-dir", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    digest_dir = Path(args.digest_dir)

    # Find matching files
    digest_files = {f.name: f for f in sorted(digest_dir.glob("*.txt"))}
    raw_files = {f.name: f for f in sorted(raw_dir.glob("*.txt"))}
    common = sorted(set(digest_files.keys()) & set(raw_files.keys()))

    digest_only = sorted(set(digest_files.keys()) - set(raw_files.keys()))

    print(f"Comparing {len(common)} matched pairs")
    print(f"Digest-only (raw failed/timeout): {len(digest_only)}")
    print(f"{'='*70}\n")

    results = []
    exact_match = 0
    type_match = 0
    normalized_match = 0
    raw_failures = 0
    digest_better = 0
    raw_better = 0

    for fname in common:
        raw_text = raw_files[fname].read_text().strip()
        digest_text = digest_files[fname].read_text().strip()

        raw_type = extract_error_type(raw_text)
        digest_type = extract_error_type(digest_text)

        raw_norm = normalize_error_type(raw_type)
        digest_norm = normalize_error_type(digest_type)

        raw_single = is_single_line(raw_text)
        digest_single = is_single_line(digest_text)

        raw_specific = has_specifics(raw_text)
        digest_specific = has_specifics(digest_text)

        raw_bad = is_timeout_or_ramble(raw_text)
        digest_bad = is_timeout_or_ramble(digest_text)

        if raw_bad:
            raw_failures += 1

        if raw_type == digest_type:
            exact_match += 1
        if raw_norm == digest_norm:
            normalized_match += 1

        # Quality comparison
        raw_score = sum([raw_single, raw_specific, not raw_bad, bool(raw_type)])
        digest_score = sum([digest_single, digest_specific, not digest_bad, bool(digest_type)])

        if digest_score > raw_score:
            digest_better += 1
            quality = "DIGEST BETTER"
        elif raw_score > digest_score:
            raw_better += 1
            quality = "RAW BETTER"
        else:
            quality = "EQUAL"

        status = "MATCH" if raw_norm == digest_norm else "DIFFER"
        if raw_bad:
            status = "RAW FAILED"

        print(f"[{fname}] {status}")
        if raw_bad:
            print(f"  Raw:    {raw_text[:80]}... [FAILED]")
        else:
            print(f"  Raw:    {raw_text[:100]}")
        print(f"  Digest: {digest_text[:100]}")
        if status == "DIFFER" and not raw_bad:
            print(f"  Types:  raw='{raw_type}' → '{raw_norm}', digest='{digest_type}' → '{digest_norm}'")
        print()

        results.append({
            "file": fname,
            "raw_summary": raw_text[:200],
            "digest_summary": digest_text[:200],
            "raw_error_type": raw_type,
            "digest_error_type": digest_type,
            "raw_normalized": raw_norm,
            "digest_normalized": digest_norm,
            "exact_type_match": raw_type == digest_type,
            "normalized_match": raw_norm == digest_norm,
            "raw_single_line": raw_single,
            "digest_single_line": digest_single,
            "raw_has_specifics": raw_specific,
            "digest_has_specifics": digest_specific,
            "raw_failed": raw_bad,
            "digest_failed": digest_bad,
            "quality": quality,
        })

    # Digest-only analysis
    digest_only_results = []
    for fname in digest_only:
        digest_text = digest_files[fname].read_text().strip()
        digest_type = extract_error_type(digest_text)
        digest_single = is_single_line(digest_text)
        digest_specific = has_specifics(digest_text)
        digest_bad = is_timeout_or_ramble(digest_text)

        digest_only_results.append({
            "file": fname,
            "summary": digest_text[:200],
            "error_type": digest_type,
            "single_line": digest_single,
            "has_specifics": digest_specific,
            "failed": digest_bad,
        })

    # Summary
    n = len(common)
    n_total = len(common) + len(digest_only)

    print(f"{'='*70}")
    print(f"QUALITY REPORT — {n} matched pairs, {len(digest_only)} digest-only")
    print(f"{'='*70}")
    print(f"")
    print(f"--- Error Type Agreement ---")
    print(f"Exact match:      {exact_match}/{n} ({exact_match/n*100:.0f}%)")
    print(f"Normalized match: {normalized_match}/{n} ({normalized_match/n*100:.0f}%)")
    print(f"")
    print(f"--- Format Compliance ---")
    raw_single_n = sum(1 for r in results if r["raw_single_line"])
    digest_single_n = sum(1 for r in results if r["digest_single_line"])
    print(f"Single-line (raw):    {raw_single_n}/{n} ({raw_single_n/n*100:.0f}%)")
    print(f"Single-line (digest): {digest_single_n}/{n} ({digest_single_n/n*100:.0f}%)")
    print(f"")
    print(f"--- Specificity ---")
    raw_spec_n = sum(1 for r in results if r["raw_has_specifics"])
    digest_spec_n = sum(1 for r in results if r["digest_has_specifics"])
    print(f"Has specifics (raw):    {raw_spec_n}/{n} ({raw_spec_n/n*100:.0f}%)")
    print(f"Has specifics (digest): {digest_spec_n}/{n} ({digest_spec_n/n*100:.0f}%)")
    print(f"")
    print(f"--- Failure Modes ---")
    print(f"Raw failures (timeout/ramble): {raw_failures}/{n} ({raw_failures/n*100:.0f}%)")
    digest_failures = sum(1 for r in results if r["digest_failed"])
    print(f"Digest failures:               {digest_failures}/{n} ({digest_failures/n*100:.0f}%)")
    print(f"")
    print(f"--- Overall Quality ---")
    print(f"Digest better than raw: {digest_better}/{n}")
    print(f"Raw better than digest: {raw_better}/{n}")
    print(f"Equal quality:          {n - digest_better - raw_better}/{n}")

    # Excluding raw failures, what's the match rate?
    non_fail = [r for r in results if not r["raw_failed"]]
    if non_fail:
        nf_match = sum(1 for r in non_fail if r["normalized_match"])
        print(f"\nExcluding raw failures ({len(non_fail)} pairs):")
        print(f"  Normalized match: {nf_match}/{len(non_fail)} ({nf_match/len(non_fail)*100:.0f}%)")

    output = {
        "n_matched_pairs": n,
        "n_digest_only": len(digest_only),
        "exact_type_match_pct": round(exact_match / n * 100, 1),
        "normalized_match_pct": round(normalized_match / n * 100, 1),
        "raw_failures": raw_failures,
        "digest_failures": digest_failures,
        "digest_better": digest_better,
        "raw_better": raw_better,
        "results": results,
        "digest_only": digest_only_results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
