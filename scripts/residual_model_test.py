#!/usr/bin/env python3
"""Multi-model residual test: can a smaller model handle reasoning fields?

The hybrid engine extracts 7+ fields via rules. This script tests whether
a smaller model (e.g., qwen3:4b) can handle the remaining 4 reasoning fields
(root_cause_category, assignment_group, summary, next_steps) as well as the
full 32B model.

Usage:
    python scripts/residual_model_test.py --tickets 10
    python scripts/residual_model_test.py --tickets 20 --small-model qwen3:4b
"""

import argparse
import asyncio
import json
import sys
import time
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "helpdesk_tickets.jsonl"

LARGE_MODEL = "qwen2.5:32b"
SMALL_MODEL = "qwen3:4b"

# Assessment fields produced by the LLM (not rules)
LLM_FIELDS = ["root_cause_category", "assignment_group", "summary", "next_steps"]


def load_tickets(n: int) -> list[dict]:
    with open(DATASET_PATH) as f:
        tickets = [json.loads(line) for line in f]
    return tickets[:n]


def format_ticket(ticket: dict) -> str:
    return "\n".join([
        f"Ticket ID:    {ticket['id']}",
        f"Subject:      {ticket['subject']}",
        f"From:         {ticket['requesterEmail']}",
        f"Created:      {ticket['createdAt']}",
        f"Priority:     {ticket['priority']}",
        f"Category:     {ticket['category']}",
        "",
        "Description:",
        ticket["description"],
    ])


def check_assessment(output: dict) -> dict:
    """Check presence and quality of assessment fields."""
    a = output.get("assessment", {})
    return {
        "severity": a.get("severity"),
        "root_cause_category": a.get("root_cause_category"),
        "assignment_group": a.get("assignment_group"),
        "summary": a.get("summary"),
        "next_steps": a.get("next_steps"),
        "has_severity": a.get("severity") is not None,
        "has_root_cause": a.get("root_cause_category") is not None,
        "has_assignment": a.get("assignment_group") is not None,
        "has_summary": a.get("summary") is not None,
        "has_next_steps": isinstance(a.get("next_steps"), list) and len(a.get("next_steps", [])) > 0,
    }


class NothinkProvider:
    """Wrapper that prepends /nothink to user prompts for Qwen3 models."""

    def __init__(self, inner):
        self._inner = inner

    async def complete(self, system: str, prompt: str, **kwargs):
        return await self._inner.complete(system, "/nothink\n" + prompt, **kwargs)


async def run_hybrid_with_model(model: str, system_prompt: str, input_text: str,
                                 rules: list, llm_fields: list, base_url: str | None,
                                 nothink: bool = False) -> dict:
    """Run hybrid pipeline with a specific model."""
    from drift.hybrid.executor import run_hybrid
    from drift.providers.factory import make_provider

    provider = make_provider({"model": model, "base_url": base_url})
    if nothink:
        provider = NothinkProvider(provider)

    t0 = time.time()
    result = await run_hybrid(
        provider=provider,
        system_prompt=system_prompt,
        input_text=input_text,
        rules=rules,
        llm_fields=llm_fields,
        temperature=0,
        output_mode="merge",
    )
    elapsed_ms = (time.time() - t0) * 1000

    return {
        "model": model + (" /nothink" if nothink else ""),
        "output": result.merged,
        "llm_output": result.llm_output,
        "latency_ms": elapsed_ms,
        "llm_ms": result.response.total_ms if result.response else 0,
        "rules_matched": result.rules_matched,
    }


def compare_results(label: str, large_results: list, small_results: list,
                     tickets: list, large_model: str, small_label: str):
    """Print comparison summary between large and small model results."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    large_ms = [r["result"]["llm_ms"] for r in large_results]
    small_ms = [r["result"]["llm_ms"] for r in small_results]
    large_mean = sum(large_ms) / len(large_ms)
    small_mean = sum(small_ms) / len(small_ms)
    speedup = (large_mean - small_mean) / large_mean * 100

    print(f"\n  LLM Latency (residual fields only):")
    print(f"    {large_model:<25s}  {large_mean:>8,.0f} ms mean")
    print(f"    {small_label:<25s}  {small_mean:>8,.0f} ms mean")
    print(f"    Speedup:                   {speedup:>7.0f}%")

    presence_fields = ["has_severity", "has_root_cause", "has_assignment", "has_summary", "has_next_steps"]
    print(f"\n  Field Presence:")
    print(f"    {'Field':<25s} {large_model:>20s} {small_label:>20s}")
    print(f"    {'-'*25} {'-'*20} {'-'*20}")
    for field in presence_fields:
        l_rate = sum(r["assessment"][field] for r in large_results) / len(large_results) * 100
        s_rate = sum(r["assessment"][field] for r in small_results) / len(small_results) * 100
        print(f"    {field:<25s} {l_rate:>18.0f}%  {s_rate:>18.0f}%")

    print(f"\n  Field Agreement (small matches large):")
    agreement = {}
    for field in LLM_FIELDS:
        matches = 0
        total = 0
        for lr, sr in zip(large_results, small_results):
            lv = lr["assessment"].get(field)
            sv = sr["assessment"].get(field)
            if lv is None or sv is None:
                continue
            total += 1
            if field in ("summary", "next_steps"):
                matches += 1
            else:
                matches += (lv == sv)
        rate = matches / total * 100 if total else 0
        agreement[field] = rate
        print(f"    {field:<25s} {matches:>3}/{total:<3} ({rate:.0f}%)")

    disagreements = []
    for i, (lr, sr) in enumerate(zip(large_results, small_results)):
        for field in ["root_cause_category", "assignment_group", "severity"]:
            lv = lr["assessment"].get(field)
            sv = sr["assessment"].get(field)
            if lv and sv and lv != sv:
                disagreements.append({
                    "ticket": tickets[i]["id"],
                    "field": field,
                    "large": lv,
                    "small": sv,
                    "priority": tickets[i]["priority"],
                    "category": tickets[i]["category"],
                })

    if disagreements:
        print(f"\n  Disagreements ({len(disagreements)}):")
        for d in disagreements:
            print(f"    {d['ticket']} [{d['priority']}/{d['category']}] "
                  f"{d['field']}: {d['large']} vs {d['small']}")

    return {
        "small_model": small_label,
        "large_mean_ms": round(large_mean),
        "small_mean_ms": round(small_mean),
        "speedup_pct": round(speedup, 1),
        "agreement": agreement,
        "disagreements": disagreements,
    }


async def main():
    parser = argparse.ArgumentParser(description="Multi-model residual test")
    parser.add_argument("--tickets", type=int, default=10)
    parser.add_argument("--large-model", default=LARGE_MODEL)
    parser.add_argument("--small-models", nargs="+", default=[SMALL_MODEL],
                        help="Small models to test (e.g., qwen3:4b llama3.1:8b)")
    parser.add_argument("--nothink", nargs="*", default=[],
                        help="Models that should use /nothink (e.g., qwen3:4b)")
    args = parser.parse_args()

    nothink_set = set(args.nothink) if args.nothink else set()

    from drift.tasks import resolve_task_prompts

    cfg = yaml.safe_load(
        (PROJECT_ROOT / "experiments" / "baseline-helpdesk.yaml").read_text()
    )
    cfg = resolve_task_prompts(cfg, PROJECT_ROOT / "tasks")
    system_prompt = cfg["experiment"]["system_prompt"]
    base_url = cfg["experiment"].get("base_url")

    hybrid_spec = yaml.safe_load(
        (PROJECT_ROOT / "tasks" / "helpdesk_triage" / "hybrid_auto.yaml").read_text()
    )
    rules = hybrid_spec["rules"]
    llm_fields = hybrid_spec["llm_fields"]

    tickets = load_tickets(args.tickets)

    # Build list of (model, nothink, label) tuples
    candidates = []
    for model in args.small_models:
        if model in nothink_set:
            candidates.append((model, True, f"{model} /nothink"))
        else:
            candidates.append((model, False, model))

    print("=" * 70)
    print("MULTI-MODEL RESIDUAL TEST")
    print("=" * 70)
    print(f"  Reference:    {args.large_model}")
    print(f"  Candidates:   {', '.join(c[2] for c in candidates)}")
    print(f"  Tickets:      {len(tickets)}")
    print(f"  Rules:        {len(rules)} (extract ticket metadata + severity)")
    print(f"  LLM fields:   {', '.join(llm_fields)}")
    print()

    # Run reference model on all tickets first
    print("─" * 70)
    print(f"  Reference: {args.large_model}")
    print("─" * 70)
    large_results = []
    for i, ticket in enumerate(tickets):
        input_text = format_ticket(ticket)
        print(f"  {i+1:>2}/{len(tickets)} {ticket['id']}...", end="", flush=True)
        lr = await run_hybrid_with_model(
            args.large_model, system_prompt, input_text, rules, llm_fields, base_url)
        la = check_assessment(lr["output"])
        large_results.append({"ticket": ticket, "result": lr, "assessment": la})
        print(f" {lr['llm_ms']:.0f}ms")

    # Run each candidate
    all_candidate_results = {}
    for model, use_nothink, label in candidates:
        print()
        print("─" * 70)
        print(f"  Candidate: {label}")
        print("─" * 70)
        small_results = []
        for i, ticket in enumerate(tickets):
            input_text = format_ticket(ticket)
            print(f"  {i+1:>2}/{len(tickets)} {ticket['id']}...", end="", flush=True)
            sr = await run_hybrid_with_model(
                model, system_prompt, input_text, rules, llm_fields, base_url,
                nothink=use_nothink)
            sa = check_assessment(sr["output"])
            small_results.append({"ticket": ticket, "result": sr, "assessment": sa})
            print(f" {sr['llm_ms']:.0f}ms")

        all_candidate_results[label] = small_results

    # Print comparison for each candidate
    all_summaries = []
    for model, use_nothink, label in candidates:
        summary = compare_results(
            f"{args.large_model} vs {label}",
            large_results, all_candidate_results[label],
            tickets, args.large_model, label,
        )
        all_summaries.append(summary)

    # Final leaderboard
    print(f"\n\n{'=' * 70}")
    print(f"  LEADERBOARD")
    print(f"{'=' * 70}")
    print(f"\n  {'Model':<25s} {'Latency':>10s} {'Speedup':>10s} {'root_cause':>12s} {'assign_grp':>12s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")
    print(f"  {args.large_model:<25s} {large_results[0]['result']['llm_ms']:>8.0f}ms {'(ref)':>10s} {'(ref)':>12s} {'(ref)':>12s}")
    large_mean = sum(r["result"]["llm_ms"] for r in large_results) / len(large_results)
    print(f"  {'':25s} {large_mean:>8.0f}ms")
    for s in all_summaries:
        rc = s["agreement"].get("root_cause_category", 0)
        ag = s["agreement"].get("assignment_group", 0)
        print(f"  {s['small_model']:<25s} {s['small_mean_ms']:>8}ms {s['speedup_pct']:>9.0f}% {rc:>10.0f}%  {ag:>10.0f}%")

    # Save
    results_dir = PROJECT_ROOT / "results" / "residual_model_test"
    results_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "large_model": args.large_model,
        "n_tickets": len(tickets),
        "candidates": all_summaries,
    }
    out_path = results_dir / "residual_comparison.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
