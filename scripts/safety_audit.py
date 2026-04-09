"""Safety audit of existing drift results.

Runs Llama Guard on saved LLM outputs to measure:
- Rejection rate per task type
- What categories get flagged
- False positive analysis (are flagged outputs actually harmful?)
- Per-task and per-step guard latency

This provides the first empirical data for the "retry death spiral"
analysis in docs/ai-safety-overhead.md.

Usage:
    python scripts/safety_audit.py [--results-dir results/] [--model llama-guard3:1b]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drift.safety.guard import GUARD_CATEGORIES, GuardModel


def load_outputs(results_dir: Path) -> list[dict]:
    """Load all LLM outputs from a results directory tree.

    Returns list of dicts with:
        experiment, timestamp, run_index, step (or "baseline"),
        content, content_length
    """
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
                            # Loop iterations
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


def audit_outputs(guard: GuardModel, outputs: list[dict], max_outputs: int = 0) -> list[dict]:
    """Run guard checks on all outputs."""
    results = []
    total = len(outputs) if max_outputs == 0 else min(max_outputs, len(outputs))

    for i, output in enumerate(outputs[:total]):
        exp = output["experiment"]
        step = output["step"]
        run = output["run_index"]

        print(f"  [{i+1}/{total}] {exp} run {run} step={step} ({output['content_length']} chars)...",
              end="", flush=True)

        try:
            result = guard.check(output["content"])
            verdict = "SAFE" if result.safe else f"UNSAFE ({result.category})"
            print(f" {verdict} ({result.latency_ms:.0f}ms)")

            results.append({
                **output,
                "safe": result.safe,
                "category": result.category,
                "guard_latency_ms": result.latency_ms,
                "guard_response": result.raw_response,
            })
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({
                **output,
                "safe": None,
                "category": None,
                "guard_latency_ms": None,
                "guard_response": str(e),
            })

    return results


def print_summary(results: list[dict]):
    """Print summary statistics."""
    total = len(results)
    errors = sum(1 for r in results if r["safe"] is None)
    checked = total - errors
    safe = sum(1 for r in results if r["safe"] is True)
    unsafe = sum(1 for r in results if r["safe"] is False)

    print("\n" + "=" * 70)
    print("SAFETY AUDIT SUMMARY")
    print("=" * 70)
    print(f"\nTotal outputs checked: {checked} ({errors} errors)")
    print(f"Safe:   {safe} ({safe/checked*100:.1f}%)" if checked else "")
    print(f"Unsafe: {unsafe} ({unsafe/checked*100:.1f}%)" if checked else "")

    if unsafe:
        rejection_rate = unsafe / checked
        retries_per_useful = 1 / (1 - rejection_rate) if rejection_rate < 1 else float("inf")
        print(f"\nRejection rate: {rejection_rate:.1%}")
        print(f"Expected retries per useful output: {retries_per_useful:.2f}x")

    # Per-experiment breakdown
    experiments = {}
    for r in results:
        if r["safe"] is None:
            continue
        exp = r["experiment"]
        if exp not in experiments:
            experiments[exp] = {"safe": 0, "unsafe": 0, "latencies": []}
        if r["safe"]:
            experiments[exp]["safe"] += 1
        else:
            experiments[exp]["unsafe"] += 1
        if r["guard_latency_ms"]:
            experiments[exp]["latencies"].append(r["guard_latency_ms"])

    print(f"\n{'Experiment':<45s} {'Safe':>5s} {'Unsafe':>7s} {'Rej%':>6s} {'AvgMs':>7s}")
    print("-" * 72)
    for exp in sorted(experiments):
        d = experiments[exp]
        t = d["safe"] + d["unsafe"]
        rej = d["unsafe"] / t * 100 if t else 0
        avg_ms = sum(d["latencies"]) / len(d["latencies"]) if d["latencies"] else 0
        print(f"{exp:<45s} {d['safe']:>5d} {d['unsafe']:>7d} {rej:>5.1f}% {avg_ms:>6.0f}ms")

    # Category breakdown
    categories = {}
    for r in results:
        if r["safe"] is False and r["category"]:
            cat = r["category"]
            categories[cat] = categories.get(cat, 0) + 1

    if categories:
        print(f"\nUnsafe categories:")
        for cat in sorted(categories, key=categories.get, reverse=True):
            desc = GUARD_CATEGORIES.get(cat, "Unknown")
            print(f"  {cat}: {categories[cat]} ({desc})")

    # Guard latency stats
    latencies = [r["guard_latency_ms"] for r in results if r["guard_latency_ms"]]
    if latencies:
        latencies.sort()
        print(f"\nGuard check latency:")
        print(f"  Mean:   {sum(latencies)/len(latencies):.0f}ms")
        print(f"  Median: {latencies[len(latencies)//2]:.0f}ms")
        print(f"  P95:    {latencies[int(len(latencies)*0.95)]:.0f}ms")
        print(f"  Min:    {latencies[0]:.0f}ms")
        print(f"  Max:    {latencies[-1]:.0f}ms")

    # Retry death spiral projection
    if checked and unsafe:
        print("\n" + "=" * 70)
        print("RETRY DEATH SPIRAL PROJECTION")
        print("=" * 70)
        rej = unsafe / checked
        for steps in [1, 3, 4, 5, 16]:
            retries = steps * (1 / (1 - rej) - 1)
            total_calls = steps + retries
            overhead_pct = retries / steps * 100
            print(f"  {steps:>2d}-step pipeline: {retries:.1f} expected retries, "
                  f"{total_calls:.1f} total calls ({overhead_pct:.0f}% overhead)")


def main():
    parser = argparse.ArgumentParser(description="Safety audit of drift results")
    parser.add_argument("--results-dir", default="results",
                       help="Path to results directory")
    parser.add_argument("--model", default="llama-guard3:1b",
                       help="Guard model name on Ollama")
    parser.add_argument("--base-url", default=None,
                       help="Ollama base URL")
    parser.add_argument("--max-outputs", type=int, default=0,
                       help="Max outputs to check (0 = all)")
    parser.add_argument("--output", "-o", default=None,
                       help="Save detailed results to JSON")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = Path(__file__).parent.parent / results_dir

    print(f"Loading outputs from {results_dir}...")
    outputs = load_outputs(results_dir)
    print(f"Found {len(outputs)} LLM outputs across {len(set(o['experiment'] for o in outputs))} experiments")

    if not outputs:
        print("No outputs found.")
        return

    # Deduplicate identical outputs (same content across runs = same guard result)
    seen = {}
    unique_outputs = []
    for o in outputs:
        key = o["content"][:500]  # First 500 chars as dedup key
        if key not in seen:
            seen[key] = True
            unique_outputs.append(o)

    print(f"Unique outputs (by content): {len(unique_outputs)}")
    print(f"Deduplication saved {len(outputs) - len(unique_outputs)} redundant checks")

    guard = GuardModel(model=args.model, base_url=args.base_url)

    print(f"\nRunning guard checks with {args.model}...")
    start = time.perf_counter()
    results = audit_outputs(guard, unique_outputs, args.max_outputs)
    elapsed = time.perf_counter() - start

    print_summary(results)
    print(f"\nTotal audit time: {elapsed:.1f}s")

    if args.output:
        out_path = Path(args.output)
        # Remove full content from saved results to save space
        save_results = []
        for r in results:
            r_copy = dict(r)
            r_copy["content"] = r_copy["content"][:200] + "..." if len(r_copy["content"]) > 200 else r_copy["content"]
            save_results.append(r_copy)
        out_path.write_text(json.dumps(save_results, indent=2))
        print(f"Detailed results saved to {out_path}")


if __name__ == "__main__":
    main()
