#!/usr/bin/env python3
"""Ansible log pre-digestion: corpus analysis and semantic compression.

Phase 1: Analyze 500 failed Ansible job logs to understand structure,
identify noise patterns, and quantify compression potential.

Usage:
    uv run python scripts/ansible_log_predigest.py --log-dir datasets/ansible_triage/inputs/
    uv run python scripts/ansible_log_predigest.py --log-dir datasets/ansible_triage/inputs/ --predigest job_1460374.txt
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Level 1: Line-level classification
# ---------------------------------------------------------------------------

# Patterns for lines that carry zero diagnostic signal
NOISE_PATTERNS = [
    # Ansible result lines for successful tasks
    (re.compile(r"^ok: \["), "ok"),
    (re.compile(r"^skipping: \["), "skipping"),
    (re.compile(r"^changed: \["), "changed"),
    (re.compile(r"^included: "), "included"),
    # Timestamp lines (standalone, not part of a TASK header)
    (re.compile(r"^\w+day \d{2} \w+ \d{4}\s+\d{2}:\d{2}:\d{2}"), "timestamp"),
    # Galaxy/collection install output
    (re.compile(r"^Starting galaxy (role|collection) install"), "galaxy"),
    (re.compile(r"^- (downloading|extracting|installing) role"), "galaxy"),
    (re.compile(r"^Process install dependency map"), "galaxy"),
    (re.compile(r"^(Starting|Rewriting) requirements"), "galaxy"),
    # Vault prompt
    (re.compile(r"^Vault password"), "vault"),
    # Blank lines
    (re.compile(r"^\s*$"), "blank"),
    # Retry lines (keep count, collapse later)
    (re.compile(r"^FAILED - RETRYING:"), "retry"),
    # Play recap timing lines at the end
    (re.compile(r"^[\w_\- :]+ -{3,} \d+\.\d+s$"), "recap_timing"),
    # Deprecation warnings
    (re.compile(r"^\[DEPRECATION WARNING\]"), "deprecation"),
    (re.compile(r"^This feature will be removed"), "deprecation"),
    # Warning lines
    (re.compile(r"^\[WARNING\]:"), "warning"),
    # JSON fragments from successful debug/assertion output
    (re.compile(r"^\}$"), "json_close"),
    (re.compile(r"^\{$"), "json_open"),
    (re.compile(r"^\},?$"), "json_close"),
    (re.compile(r'^"changed": (true|false),?$'), "json_frag"),
    (re.compile(r'^"msg": "(All assertions passed|Pre-Workload tasks completed|Workload Tasks completed|Post-Workload Tasks completed|Step \d+)'), "json_success_msg"),
    (re.compile(r'^"ansible_loop_var"'), "json_frag"),
    (re.compile(r'^"status": "CREATE_COMPLETE"'), "json_frag"),
    (re.compile(r'^"status_reason": null'), "json_frag"),
    (re.compile(r'^"data": \{'), "json_frag"),
    (re.compile(r'^"group_id":'), "json_frag"),
    (re.compile(r"^skipping: no hosts matched"), "no_hosts"),
    (re.compile(r"already installed in EE"), "galaxy_ee"),
    (re.compile(r"^={10,}"), "separator"),
    (re.compile(r"^NO MORE HOSTS LEFT"), "no_hosts"),
    (re.compile(r"^TASKS RECAP"), "recap"),
    (re.compile(r'^"msg": "Starting CSV: "'), "json_frag"),
]

# Patterns for lines that carry diagnostic signal
SIGNAL_PATTERNS = [
    (re.compile(r"^fatal: \["), "fatal"),
    (re.compile(r"^FAILED!"), "failed"),
    (re.compile(r"^ERROR"), "error"),
    (re.compile(r"^PLAY \["), "play_header"),
    (re.compile(r"^TASK \["), "task_header"),
    (re.compile(r"^PLAY RECAP"), "play_recap"),
    (re.compile(r"^    \"msg\":"), "error_msg"),
    (re.compile(r"^    \"stderr\":"), "stderr"),
    (re.compile(r"^    \"rc\":"), "return_code"),
]


def classify_line(line: str) -> str:
    """Classify a single log line as noise, signal, or unknown."""
    stripped = line.strip()
    for pattern, label in NOISE_PATTERNS:
        if pattern.search(stripped):
            return f"noise:{label}"
    for pattern, label in SIGNAL_PATTERNS:
        if pattern.search(stripped):
            return f"signal:{label}"
    return "unknown"


# ---------------------------------------------------------------------------
# Level 2: Block-level dedup detection
# ---------------------------------------------------------------------------

def find_retry_blocks(lines: list[str]) -> list[dict]:
    """Find consecutive retry lines and summarize them."""
    blocks = []
    i = 0
    while i < len(lines):
        if "FAILED - RETRYING:" in lines[i]:
            start = i
            # Extract the task name
            m = re.search(r"RETRYING: \[.*?\]: (.+?) \(\d+ retries", lines[i])
            task = m.group(1) if m else "unknown"
            while i < len(lines) and "FAILED - RETRYING:" in lines[i]:
                i += 1
            blocks.append({
                "start": start,
                "end": i,
                "count": i - start,
                "task": task,
            })
        else:
            i += 1
    return blocks


def find_repeated_ok_blocks(lines: list[str]) -> list[dict]:
    """Find consecutive ok/changed lines for the same task across items."""
    blocks = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if re.match(r"^(ok|changed): \[.+?\] => \(item=", stripped):
            start = i
            while i < len(lines) and re.match(
                r"^(ok|changed): \[.+?\] => \(item=", lines[i].strip()
            ):
                i += 1
            if i - start >= 3:  # Only flag blocks of 3+ items
                blocks.append({
                    "start": start,
                    "end": i,
                    "count": i - start,
                    "type": "item_loop",
                })
        else:
            i += 1
    return blocks


# ---------------------------------------------------------------------------
# Level 3: Large embedded data detection
# ---------------------------------------------------------------------------

def find_large_json_blobs(lines: list[str], min_chars: int = 500) -> list[dict]:
    """Find lines containing large JSON/dict blobs (API responses, etc)."""
    blobs = []
    for i, line in enumerate(lines):
        # Look for lines with deeply nested JSON (Kubernetes API responses, etc.)
        if len(line) > min_chars and (
            "'metadata':" in line
            or '"metadata":' in line
            or "'spec':" in line
            or '"spec":' in line
            or "fieldsV1" in line
        ):
            blobs.append({
                "line": i,
                "length": len(line),
                "type": "k8s_api_response",
            })
        elif len(line) > min_chars and ("REDACTED" in line or "Base64" in line):
            blobs.append({
                "line": i,
                "length": len(line),
                "type": "redacted_secret",
            })
        elif len(line) > min_chars:
            blobs.append({
                "line": i,
                "length": len(line),
                "type": "long_line",
            })
    return blobs


# ---------------------------------------------------------------------------
# Corpus analysis
# ---------------------------------------------------------------------------

def analyze_log(filepath: Path) -> dict:
    """Analyze a single log file and return statistics."""
    lines = filepath.read_text(errors="replace").splitlines()
    total = len(lines)

    classifications = Counter()
    for line in lines:
        cls = classify_line(line)
        classifications[cls] += 1

    retry_blocks = find_retry_blocks(lines)
    item_blocks = find_repeated_ok_blocks(lines)
    json_blobs = find_large_json_blobs(lines)

    noise_count = sum(v for k, v in classifications.items() if k.startswith("noise:"))
    signal_count = sum(v for k, v in classifications.items() if k.startswith("signal:"))
    unknown_count = classifications.get("unknown", 0)

    total_chars = sum(len(line) for line in lines)
    blob_chars = sum(b["length"] for b in json_blobs)

    return {
        "file": filepath.name,
        "total_lines": total,
        "total_chars": total_chars,
        "noise_lines": noise_count,
        "signal_lines": signal_count,
        "unknown_lines": unknown_count,
        "noise_pct": round(noise_count / total * 100, 1) if total > 0 else 0,
        "classifications": dict(classifications),
        "retry_blocks": len(retry_blocks),
        "retry_lines": sum(b["count"] for b in retry_blocks),
        "item_loop_blocks": len(item_blocks),
        "item_loop_lines": sum(b["count"] for b in item_blocks),
        "json_blobs": len(json_blobs),
        "json_blob_chars": blob_chars,
        "json_blob_pct": round(blob_chars / total_chars * 100, 1) if total_chars > 0 else 0,
    }


def analyze_corpus(log_dir: Path) -> dict:
    """Analyze all logs in a directory."""
    log_files = sorted(log_dir.glob("*.txt"))
    if not log_files:
        print(f"No .txt files found in {log_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing {len(log_files)} log files...", file=sys.stderr)

    results = []
    all_classifications = Counter()
    total_lines = 0
    total_chars = 0
    total_noise = 0
    total_signal = 0
    total_unknown = 0
    total_retry_lines = 0
    total_item_loop_lines = 0
    total_json_blob_chars = 0
    line_counts = []

    for f in log_files:
        r = analyze_log(f)
        results.append(r)
        for k, v in r["classifications"].items():
            all_classifications[k] += v
        total_lines += r["total_lines"]
        total_chars += r["total_chars"]
        total_noise += r["noise_lines"]
        total_signal += r["signal_lines"]
        total_unknown += r["unknown_lines"]
        total_retry_lines += r["retry_lines"]
        total_item_loop_lines += r["item_loop_lines"]
        total_json_blob_chars += r["json_blob_chars"]
        line_counts.append(r["total_lines"])

    line_counts.sort()
    n = len(line_counts)

    # Find most common unknown lines across corpus (Level 3 candidates)
    unknown_lines = Counter()
    for f in log_files:
        lines = f.read_text(errors="replace").splitlines()
        for line in lines:
            if classify_line(line) == "unknown":
                # Normalize: strip variable parts (timestamps, IDs, paths)
                normalized = re.sub(r"\b[0-9a-f]{8,}\b", "<ID>", line.strip())
                normalized = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", "<TS>", normalized)
                normalized = re.sub(r"/runner/project/\S+", "<PATH>", normalized)
                if len(normalized) < 200:  # Skip huge lines
                    unknown_lines[normalized] += 1

    return {
        "corpus_size": len(log_files),
        "total_lines": total_lines,
        "total_chars": total_chars,
        "line_counts": {
            "min": line_counts[0],
            "p25": line_counts[n // 4],
            "median": line_counts[n // 2],
            "p75": line_counts[3 * n // 4],
            "max": line_counts[-1],
            "mean": round(total_lines / n, 1),
        },
        "classification_totals": {
            "noise": total_noise,
            "signal": total_signal,
            "unknown": total_unknown,
            "noise_pct": round(total_noise / total_lines * 100, 1),
            "signal_pct": round(total_signal / total_lines * 100, 1),
            "unknown_pct": round(total_unknown / total_lines * 100, 1),
        },
        "classification_detail": dict(all_classifications.most_common()),
        "dedup_potential": {
            "retry_lines": total_retry_lines,
            "retry_pct": round(total_retry_lines / total_lines * 100, 1),
            "item_loop_lines": total_item_loop_lines,
            "item_loop_pct": round(total_item_loop_lines / total_lines * 100, 1),
        },
        "large_blob_potential": {
            "json_blob_chars": total_json_blob_chars,
            "json_blob_char_pct": round(
                total_json_blob_chars / total_chars * 100, 1
            ) if total_chars > 0 else 0,
        },
        "top_unknown_lines": dict(unknown_lines.most_common(30)),
        "per_file": results,
    }


# ---------------------------------------------------------------------------
# Pre-digestion (apply all levels)
# ---------------------------------------------------------------------------

def predigest(filepath: Path) -> str:
    """Apply all pre-digestion levels and return compressed log."""
    lines = filepath.read_text(errors="replace").splitlines()

    # Pre-scan: find which TASK headers are near errors (keep those)
    error_lines = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith("fatal:") or stripped.startswith("FAILED!")
                or stripped.startswith("failed:") or "UNREACHABLE!" in stripped):
            # Mark a window around the error
            for j in range(max(0, i - 5), min(len(lines), i + 3)):
                error_lines.add(j)

    output = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Level 2: Collapse retry blocks
        if "FAILED - RETRYING:" in stripped:
            m = re.search(r"RETRYING: \[(.+?)\]: (.+?) \((\d+) retries", stripped)
            host = m.group(1) if m else "?"
            task = m.group(2) if m else "?"
            start_retries = int(m.group(3)) if m else 0
            count = 0
            while i < len(lines) and "FAILED - RETRYING:" in lines[i]:
                count += 1
                i += 1
            # Check if next line is success
            succeeded = (
                i < len(lines) and lines[i].strip().startswith("ok:")
            )
            status = "then succeeded" if succeeded else "still failing"
            output.append(
                f"[RETRY: {task} on {host}, {count} retries, {status}]"
            )
            continue

        # Level 2: Collapse item loops (ok/changed across items)
        if re.match(r"^(ok|changed): \[.+?\] => \(item=", stripped):
            action = "ok" if stripped.startswith("ok:") else "changed"
            count = 0
            while i < len(lines) and re.match(
                r"^(ok|changed): \[.+?\] => \(item=", lines[i].strip()
            ):
                count += 1
                i += 1
            if count >= 3:
                output.append(f"[{action}: {count} items processed]")
                continue
            else:
                # Few enough to keep
                i -= count
                # Fall through to Level 1

        cls = classify_line(line)

        # Level 1: Strip pure noise
        if cls.startswith("noise:"):
            i += 1
            continue

        # Level 1: Strip standalone timestamp lines
        if cls == "noise:timestamp":
            i += 1
            continue

        # Level 1: Strip TASK headers that aren't near errors
        if cls == "signal:task_header" and i not in error_lines:
            i += 1
            continue

        # Level 1: Collapse recap timing (keep only first line as marker)
        if cls == "noise:recap_timing":
            if not output or not output[-1].startswith("[RECAP TIMING"):
                output.append("[RECAP TIMING: task duration summary omitted]")
            i += 1
            continue

        # Level 3: Truncate huge lines (K8s API responses etc)
        if len(line) > 500 and (
            "'metadata':" in line or '"metadata":' in line or "fieldsV1" in line
        ):
            # Extract just the key info
            m = re.search(r"'kind': '(\w+)'", line) or re.search(r'"kind": "(\w+)"', line)
            kind = m.group(1) if m else "resource"
            m = re.search(r"'name': '([\w-]+)'", line) or re.search(r'"name": "([\w-]+)"', line)
            name = m.group(1) if m else "?"
            m = re.search(r"'namespace': '([\w-]+)'", line) or re.search(r'"namespace": "([\w-]+)"', line)
            ns = m.group(1) if m else "?"
            output.append(f"[K8S API RESPONSE: {kind}/{name} in {ns} — full response omitted]")
            i += 1
            continue

        # Keep everything else
        output.append(line)
        i += 1

    return "\n".join(output)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_report(corpus: dict) -> None:
    """Print a human-readable corpus analysis report."""
    c = corpus
    ct = c["classification_totals"]
    lc = c["line_counts"]
    dp = c["dedup_potential"]
    bp = c["large_blob_potential"]

    print(f"\n{'='*70}")
    print(f"ANSIBLE LOG CORPUS ANALYSIS — {c['corpus_size']} files")
    print(f"{'='*70}\n")

    print(f"Total lines:  {c['total_lines']:,}")
    print(f"Total chars:  {c['total_chars']:,}")
    print(f"Lines/file:   min={lc['min']}, p25={lc['p25']}, "
          f"median={lc['median']}, p75={lc['p75']}, max={lc['max']}, "
          f"mean={lc['mean']}")

    print(f"\n--- Line Classification ---")
    print(f"Noise:        {ct['noise']:>8,}  ({ct['noise_pct']}%)")
    print(f"Signal:       {ct['signal']:>8,}  ({ct['signal_pct']}%)")
    print(f"Unknown:      {ct['unknown']:>8,}  ({ct['unknown_pct']}%)")

    print(f"\n--- Classification Detail ---")
    for label, count in sorted(
        c["classification_detail"].items(), key=lambda x: -x[1]
    ):
        pct = round(count / c["total_lines"] * 100, 1)
        print(f"  {label:<30s} {count:>8,}  ({pct}%)")

    print(f"\n--- Dedup Potential (Level 2) ---")
    print(f"Retry lines:     {dp['retry_lines']:>8,}  ({dp['retry_pct']}%)")
    print(f"Item loop lines: {dp['item_loop_lines']:>8,}  ({dp['item_loop_pct']}%)")

    print(f"\n--- Large Blob Potential (Level 3) ---")
    print(f"JSON blob chars: {bp['json_blob_chars']:>10,}  "
          f"({bp['json_blob_char_pct']}% of total chars)")

    print(f"\n--- Top 20 Unknown Line Patterns ---")
    for pattern, count in list(c["top_unknown_lines"].items())[:20]:
        trunc = pattern[:80] + "..." if len(pattern) > 80 else pattern
        print(f"  {count:>5}x  {trunc}")

    # Estimate compression
    strippable = ct["noise"] + dp["retry_lines"] + dp["item_loop_lines"]
    remaining = c["total_lines"] - strippable
    print(f"\n--- Compression Estimate ---")
    print(f"Strippable lines (L1+L2): {strippable:,} / {c['total_lines']:,} "
          f"({round(strippable/c['total_lines']*100, 1)}%)")
    print(f"Remaining after strip:    {remaining:,} lines "
          f"({round(remaining/c['total_lines']*100, 1)}%)")
    print(f"Blob char reduction (L3): ~{bp['json_blob_char_pct']}% of remaining chars")
    print()


def main():
    parser = argparse.ArgumentParser(description="Ansible log pre-digestion")
    parser.add_argument(
        "--log-dir", required=True,
        help="Directory containing failed job log .txt files",
    )
    parser.add_argument(
        "--predigest", metavar="FILENAME",
        help="Pre-digest a single file and print result (filename within log-dir)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output corpus analysis as JSON instead of human-readable",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"Not a directory: {log_dir}", file=sys.stderr)
        sys.exit(1)

    if args.predigest:
        filepath = log_dir / args.predigest
        if not filepath.exists():
            print(f"File not found: {filepath}", file=sys.stderr)
            sys.exit(1)
        result = predigest(filepath)
        original_lines = len(filepath.read_text(errors="replace").splitlines())
        result_lines = len(result.splitlines())
        print(result)
        print(
            f"\n--- Pre-digestion: {original_lines} → {result_lines} lines "
            f"({round((1 - result_lines/original_lines)*100, 1)}% reduction) ---",
            file=sys.stderr,
        )
    else:
        corpus = analyze_corpus(log_dir)
        if args.json:
            # Remove per_file for cleaner output
            corpus_summary = {k: v for k, v in corpus.items() if k != "per_file"}
            print(json.dumps(corpus_summary, indent=2))
        else:
            print_report(corpus)


if __name__ == "__main__":
    main()
