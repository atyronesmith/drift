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


async def run_hybrid_with_model(model: str, system_prompt: str, input_text: str,
                                 rules: list, llm_fields: list, base_url: str | None) -> dict:
    """Run hybrid pipeline with a specific model."""
    from drift.hybrid.executor import run_hybrid
    from drift.providers.factory import make_provider

    provider = make_provider({"model": model, "base_url": base_url})

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
        "model": model,
        "output": result.merged,
        "llm_output": result.llm_output,
        "latency_ms": elapsed_ms,
        "llm_ms": result.response.total_ms if result.response else 0,
        "rules_matched": result.rules_matched,
    }


async def main():
    parser = argparse.ArgumentParser(description="Multi-model residual test")
    parser.add_argument("--tickets", type=int, default=10)
    parser.add_argument("--large-model", default=LARGE_MODEL)
    parser.add_argument("--small-model", default=SMALL_MODEL)
    args = parser.parse_args()

    from drift.tasks import resolve_task_prompts

    # Load config and hybrid spec
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

    print("=" * 70)
    print("MULTI-MODEL RESIDUAL TEST")
    print("=" * 70)
    print(f"  Large model:  {args.large_model}")
    print(f"  Small model:  {args.small_model}")
    print(f"  Tickets:      {len(tickets)}")
    print(f"  Rules:        {len(rules)} (extract ticket metadata + severity)")
    print(f"  LLM fields:   {', '.join(llm_fields)}")
    print()

    large_results = []
    small_results = []

    for i, ticket in enumerate(tickets):
        input_text = format_ticket(ticket)
        print(f"Ticket {i+1}/{len(tickets)}: {ticket['id']} "
              f"({ticket['priority']}/{ticket['category']})")

        # Run with large model
        print(f"  {args.large_model}...", end="", flush=True)
        lr = await run_hybrid_with_model(
            args.large_model, system_prompt, input_text, rules, llm_fields, base_url)
        la = check_assessment(lr["output"])
        large_results.append({"ticket": ticket, "result": lr, "assessment": la})
        print(f" {lr['llm_ms']:.0f}ms")

        # Run with small model
        print(f"  {args.small_model}...", end="", flush=True)
        sr = await run_hybrid_with_model(
            args.small_model, system_prompt, input_text, rules, llm_fields, base_url)
        sa = check_assessment(sr["output"])
        small_results.append({"ticket": ticket, "result": sr, "assessment": sa})
        print(f" {sr['llm_ms']:.0f}ms")

        # Quick comparison
        matches = []
        for field in LLM_FIELDS:
            lv = la.get(field)
            sv = sa.get(field)
            if field == "next_steps":
                match = la["has_next_steps"] and sa["has_next_steps"]
            elif field == "summary":
                match = la["has_summary"] and sa["has_summary"]
            else:
                match = lv == sv
            matches.append(match)

        match_str = " ".join("+" if m else "-" for m in matches)
        print(f"  Match: [{match_str}] ({sum(matches)}/{len(matches)})")
        print()

    # ── Summary ──
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Latency
    large_ms = [r["result"]["llm_ms"] for r in large_results]
    small_ms = [r["result"]["llm_ms"] for r in small_results]
    large_mean = sum(large_ms) / len(large_ms)
    small_mean = sum(small_ms) / len(small_ms)
    speedup = (large_mean - small_mean) / large_mean * 100

    print(f"\n  LLM Latency (residual fields only):")
    print(f"    {args.large_model:<20s}  {large_mean:>8,.0f} ms mean")
    print(f"    {args.small_model:<20s}  {small_mean:>8,.0f} ms mean")
    print(f"    Speedup:              {speedup:>7.0f}%")

    # Field presence
    print(f"\n  Field Presence:")
    presence_fields = ["has_severity", "has_root_cause", "has_assignment", "has_summary", "has_next_steps"]
    print(f"    {'Field':<25s} {args.large_model:>15s} {args.small_model:>15s}")
    print(f"    {'-'*25} {'-'*15} {'-'*15}")
    for field in presence_fields:
        l_rate = sum(r["assessment"][field] for r in large_results) / len(large_results) * 100
        s_rate = sum(r["assessment"][field] for r in small_results) / len(small_results) * 100
        print(f"    {field:<25s} {l_rate:>13.0f}%  {s_rate:>13.0f}%")

    # Field agreement (where both models produce values)
    print(f"\n  Field Agreement (small model matches large):")
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
                # For free-text fields, just check presence
                matches += 1
            else:
                matches += (lv == sv)
        rate = matches / total * 100 if total else 0
        print(f"    {field:<25s} {matches:>3}/{total:<3} ({rate:.0f}%)")

    # Per-ticket detail for disagreements
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

    # Save results
    results_dir = PROJECT_ROOT / "results" / "residual_model_test"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "large_model": args.large_model,
        "small_model": args.small_model,
        "n_tickets": len(tickets),
        "large_mean_ms": round(large_mean),
        "small_mean_ms": round(small_mean),
        "speedup_pct": round(speedup, 1),
        "disagreements": disagreements,
        "per_ticket": [
            {
                "id": tickets[i]["id"],
                "priority": tickets[i]["priority"],
                "category": tickets[i]["category"],
                "large_ms": round(large_ms[i]),
                "small_ms": round(small_ms[i]),
                "large_assessment": {k: v for k, v in large_results[i]["assessment"].items()
                                     if not k.startswith("has_")},
                "small_assessment": {k: v for k, v in small_results[i]["assessment"].items()
                                     if not k.startswith("has_")},
            }
            for i in range(len(tickets))
        ],
    }
    out_path = results_dir / "residual_comparison.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
