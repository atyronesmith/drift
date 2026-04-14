"""CodeShield audit of drift results — static analysis vs LLM guard comparison.

Runs Meta's CodeShield static analyzer on the same LLM outputs that
safety_audit.py checked with Llama Guard. Compares:
- What each tool flags (true positives, false positives)
- Latency per check
- Suitability for software engineering workloads

Usage:
    uv run python scripts/codeshield_audit.py [--results-dir results/]
"""

import asyncio
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codeshield.cs import CodeShield


def load_outputs(results_dir: Path) -> list[dict]:
    """Load all LLM outputs from results directory (same as safety_audit.py)."""
    outputs = []

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

                if run_type in ("baseline", "hybrid"):
                    content = run.get("response", {}).get("content", "")
                    if content:
                        outputs.append({
                            "experiment": exp_dir.name,
                            "timestamp": ts_dir.name,
                            "run_index": run_idx,
                            "step": run_type,
                            "content": content,
                            "content_length": len(content),
                        })

                elif run_type == "pipeline":
                    for step in run.get("steps", []):
                        if isinstance(step, dict):
                            name = step.get("name", "unknown")
                            content = step.get("response", {}).get("content", "")
                            if content:
                                outputs.append({
                                    "experiment": exp_dir.name,
                                    "timestamp": ts_dir.name,
                                    "run_index": run_idx,
                                    "step": name,
                                    "content": content,
                                    "content_length": len(content),
                                })
                        elif isinstance(step, list):
                            for iteration in step:
                                if isinstance(iteration, list):
                                    for s in iteration:
                                        name = s.get("name", "unknown")
                                        content = s.get("response", {}).get("content", "")
                                        if content:
                                            outputs.append({
                                                "experiment": exp_dir.name,
                                                "timestamp": ts_dir.name,
                                                "run_index": run_idx,
                                                "step": name,
                                                "content": content,
                                                "content_length": len(content),
                                            })

    return outputs


def extract_code_blocks(content: str) -> list[dict]:
    """Extract fenced code blocks from markdown-style LLM output."""
    blocks = []
    pattern = r"```(\w*)\n(.*?)```"
    for match in re.finditer(pattern, content, re.DOTALL):
        lang = match.group(1).lower() or "unknown"
        code = match.group(2).strip()
        if code and len(code) > 20:  # Skip trivial snippets
            blocks.append({"language": lang, "code": code})

    # If no fenced blocks but content looks like raw code, try the whole thing
    if not blocks:
        lines = content.strip().split("\n")
        code_indicators = ["import ", "def ", "class ", "from ", "apiVersion:", "kind:"]
        if any(lines[0].startswith(ind) for ind in code_indicators):
            lang = "yaml" if lines[0].startswith(("apiVersion:", "kind:")) else "python"
            blocks.append({"language": lang, "code": content.strip()})

    return blocks


async def scan_output(content: str) -> dict:
    """Scan a single output with CodeShield. Returns scan result dict."""
    code_blocks = extract_code_blocks(content)

    all_issues = []
    total_latency_ms = 0
    blocks_scanned = 0

    for block in code_blocks:
        start = time.perf_counter()
        try:
            result = await CodeShield.scan_code(block["code"])
            elapsed_ms = (time.perf_counter() - start) * 1000
            total_latency_ms += elapsed_ms
            blocks_scanned += 1

            if result.is_insecure and result.issues_found:
                for issue in result.issues_found:
                    all_issues.append({
                        "language": block["language"],
                        "cwe_id": issue.cwe_id,
                        "description": issue.description,
                        "severity": issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity),
                        "pattern_id": issue.pattern_id,
                        "line": issue.line,
                        "analyzer": issue.analyzer.value if hasattr(issue.analyzer, 'value') else str(issue.analyzer),
                    })
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            total_latency_ms += elapsed_ms
            blocks_scanned += 1
            all_issues.append({
                "language": block["language"],
                "cwe_id": "ERROR",
                "description": str(e),
                "severity": "error",
                "pattern_id": "scan_error",
                "line": 0,
                "analyzer": "error",
            })

    return {
        "has_code": len(code_blocks) > 0,
        "code_blocks": len(code_blocks),
        "blocks_scanned": blocks_scanned,
        "is_insecure": len(all_issues) > 0,
        "issues": all_issues,
        "latency_ms": total_latency_ms,
    }


async def audit_all(outputs: list[dict]) -> list[dict]:
    """Run CodeShield on all outputs."""
    results = []
    total = len(outputs)

    for i, output in enumerate(outputs):
        exp = output["experiment"]
        step = output["step"]
        run = output["run_index"]

        scan = await scan_output(output["content"])

        status = "NO_CODE"
        if scan["has_code"]:
            if scan["is_insecure"]:
                cwes = set(iss["cwe_id"] for iss in scan["issues"] if iss["cwe_id"] != "ERROR")
                status = f"INSECURE ({', '.join(sorted(cwes))})"
            else:
                status = "CLEAN"

        print(f"  [{i+1}/{total}] {exp} run {run} step={step} — "
              f"{scan['code_blocks']} blocks, {status} ({scan['latency_ms']:.0f}ms)")

        results.append({
            "experiment": output["experiment"],
            "timestamp": output["timestamp"],
            "run_index": output["run_index"],
            "step": output["step"],
            "content_length": output["content_length"],
            "content": output["content"][:200],
            **scan,
        })

    return results


def load_guard_results(path: Path) -> dict:
    """Load Llama Guard results keyed by (experiment, step, run_index)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    lookup = {}
    for r in data:
        key = (r["experiment"], r["step"], r["run_index"])
        lookup[key] = r
    return lookup


def print_summary(results: list[dict], guard_lookup: dict):
    """Print comparison summary."""
    total = len(results)
    with_code = [r for r in results if r["has_code"]]
    without_code = [r for r in results if not r["has_code"]]
    insecure = [r for r in with_code if r["is_insecure"]]
    clean = [r for r in with_code if not r["is_insecure"]]

    print("\n" + "=" * 70)
    print("CODESHIELD AUDIT SUMMARY")
    print("=" * 70)
    print(f"\nTotal outputs:       {total}")
    print(f"With code blocks:    {len(with_code)} ({len(with_code)/total*100:.1f}%)")
    print(f"Without code:        {len(without_code)} ({len(without_code)/total*100:.1f}%)")
    print(f"Insecure (flagged):  {len(insecure)} ({len(insecure)/len(with_code)*100:.1f}% of code outputs)" if with_code else "")
    print(f"Clean:               {len(clean)} ({len(clean)/len(with_code)*100:.1f}% of code outputs)" if with_code else "")

    # Issue breakdown by CWE
    cwe_counts = {}
    for r in insecure:
        for issue in r["issues"]:
            cwe = issue["cwe_id"]
            desc = issue["description"]
            key = f"{cwe}: {desc}"
            cwe_counts[key] = cwe_counts.get(key, 0) + 1

    if cwe_counts:
        print(f"\nIssues by type:")
        for key in sorted(cwe_counts, key=cwe_counts.get, reverse=True):
            print(f"  {key}: {cwe_counts[key]}")

    # Issue breakdown by analyzer
    analyzer_counts = {}
    for r in insecure:
        for issue in r["issues"]:
            analyzer_counts[issue["analyzer"]] = analyzer_counts.get(issue["analyzer"], 0) + 1

    if analyzer_counts:
        print(f"\nIssues by analyzer:")
        for a in sorted(analyzer_counts, key=analyzer_counts.get, reverse=True):
            print(f"  {a}: {analyzer_counts[a]}")

    # Per-step breakdown
    step_stats = {}
    for r in results:
        step = r["step"]
        if step not in step_stats:
            step_stats[step] = {"total": 0, "with_code": 0, "insecure": 0, "latencies": []}
        step_stats[step]["total"] += 1
        if r["has_code"]:
            step_stats[step]["with_code"] += 1
        if r["is_insecure"]:
            step_stats[step]["insecure"] += 1
        if r["latency_ms"] > 0:
            step_stats[step]["latencies"].append(r["latency_ms"])

    print(f"\n{'Step':<20s} {'Total':>6s} {'Code':>6s} {'Flagged':>8s} {'Rate':>6s} {'AvgMs':>7s}")
    print("-" * 55)
    for step in sorted(step_stats):
        d = step_stats[step]
        rate = d["insecure"] / d["with_code"] * 100 if d["with_code"] else 0
        avg_ms = sum(d["latencies"]) / len(d["latencies"]) if d["latencies"] else 0
        print(f"{step:<20s} {d['total']:>6d} {d['with_code']:>6d} {d['insecure']:>8d} "
              f"{rate:>5.1f}% {avg_ms:>6.0f}ms")

    # Latency stats
    latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
    if latencies:
        latencies.sort()
        print(f"\nCodeShield latency (code outputs only):")
        print(f"  Mean:   {sum(latencies)/len(latencies):.0f}ms")
        print(f"  Median: {latencies[len(latencies)//2]:.0f}ms")
        print(f"  P95:    {latencies[int(len(latencies)*0.95)]:.0f}ms")
        print(f"  Min:    {latencies[0]:.0f}ms")
        print(f"  Max:    {latencies[-1]:.0f}ms")

    # Comparison with Llama Guard
    if guard_lookup:
        print("\n" + "=" * 70)
        print("CODESHIELD vs LLAMA GUARD COMPARISON")
        print("=" * 70)

        both = 0
        cs_only = 0
        guard_only = 0
        neither = 0
        guard_no_match = 0

        for r in results:
            key = (r["experiment"], r["step"], r["run_index"])
            guard = guard_lookup.get(key)
            if not guard:
                guard_no_match += 1
                continue

            cs_flagged = r["is_insecure"]
            guard_flagged = guard.get("safe") is False

            if cs_flagged and guard_flagged:
                both += 1
            elif cs_flagged and not guard_flagged:
                cs_only += 1
            elif not cs_flagged and guard_flagged:
                guard_only += 1
            else:
                neither += 1

        matched = both + cs_only + guard_only + neither
        print(f"\nMatched outputs: {matched} (unmatched: {guard_no_match})")
        print(f"\n{'':>30s} {'Guard Safe':>12s} {'Guard Unsafe':>14s}")
        print(f"{'CodeShield Clean':>30s} {neither:>12d} {guard_only:>14d}")
        print(f"{'CodeShield Flagged':>30s} {cs_only:>12d} {both:>14d}")

        print(f"\n  Both flagged:            {both:>4d}  — potential true issues")
        print(f"  CodeShield only:         {cs_only:>4d}  — real CWEs that guard missed")
        print(f"  Guard only:              {guard_only:>4d}  — guard false positives (no CWE)")
        print(f"  Neither flagged:         {neither:>4d}  — clean by both")

        if guard_only > 0:
            print(f"\n  Guard-only flags are false positives: the guard flagged {guard_only} outputs")
            print(f"  that have no actual security vulnerability (no CWE detected by static analysis).")

        # Per-step comparison
        print(f"\n{'Step':<20s} {'CS Flag':>8s} {'Guard Flag':>11s} {'CS Rate':>8s} {'Guard Rate':>11s}")
        print("-" * 60)
        step_comp = {}
        for r in results:
            key = (r["experiment"], r["step"], r["run_index"])
            guard = guard_lookup.get(key)
            if not guard:
                continue
            step = r["step"]
            if step not in step_comp:
                step_comp[step] = {"total": 0, "cs": 0, "guard": 0}
            step_comp[step]["total"] += 1
            if r["is_insecure"]:
                step_comp[step]["cs"] += 1
            if guard.get("safe") is False:
                step_comp[step]["guard"] += 1

        for step in sorted(step_comp):
            d = step_comp[step]
            cs_rate = d["cs"] / d["total"] * 100 if d["total"] else 0
            g_rate = d["guard"] / d["total"] * 100 if d["total"] else 0
            print(f"{step:<20s} {d['cs']:>8d} {d['guard']:>11d} {cs_rate:>7.1f}% {g_rate:>10.1f}%")


def main():
    parser = argparse.ArgumentParser(description="CodeShield audit of drift results")
    parser.add_argument("--results-dir", default="results",
                       help="Path to results directory")
    parser.add_argument("--guard-results", default="results/safety_audit.json",
                       help="Path to Llama Guard audit results for comparison")
    parser.add_argument("--output", "-o", default="results/codeshield_audit.json",
                       help="Save results to JSON")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = Path(__file__).parent.parent / results_dir

    guard_path = Path(args.guard_results)
    if not guard_path.is_absolute():
        guard_path = Path(__file__).parent.parent / guard_path

    print(f"Loading outputs from {results_dir}...")
    outputs = load_outputs(results_dir)
    print(f"Found {len(outputs)} LLM outputs")

    if not outputs:
        print("No outputs found.")
        return

    # Deduplicate (same as safety_audit.py)
    seen = {}
    unique_outputs = []
    for o in outputs:
        key = o["content"][:500]
        if key not in seen:
            seen[key] = True
            unique_outputs.append(o)

    print(f"Unique outputs: {len(unique_outputs)}")

    # Load guard results for comparison
    guard_lookup = load_guard_results(guard_path)
    if guard_lookup:
        print(f"Loaded {len(guard_lookup)} Llama Guard results for comparison")
    else:
        print("No Llama Guard results found — running CodeShield only")

    print(f"\nRunning CodeShield scan...")
    start = time.perf_counter()
    results = asyncio.run(audit_all(unique_outputs))
    elapsed = time.perf_counter() - start

    print_summary(results, guard_lookup)
    print(f"\nTotal scan time: {elapsed:.1f}s")

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent.parent / out_path
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
