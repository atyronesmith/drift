#!/usr/bin/env python3
"""Compare baseline vs hybrid on multiple helpdesk tickets.

Runs both approaches on N tickets and compares latency, output quality,
and field accuracy.

Usage:
    python scripts/compare_helpdesk.py --tickets 10
"""

import argparse
import asyncio
import json
import time
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "helpdesk_tickets.jsonl"


def load_tickets(n: int) -> list[dict]:
    with open(DATASET_PATH) as f:
        tickets = [json.loads(line) for line in f]
    return tickets[:n]


def format_ticket(ticket: dict) -> str:
    lines = [
        f"Ticket ID:    {ticket['id']}",
        f"Subject:      {ticket['subject']}",
        f"From:         {ticket['requesterEmail']}",
        f"Created:      {ticket['createdAt']}",
        f"Priority:     {ticket['priority']}",
        f"Category:     {ticket['category']}",
        "",
        "Description:",
        ticket["description"],
    ]
    return "\n".join(lines)


async def run_baseline(provider, system_prompt: str, user_prompt_tpl: str,
                       input_text: str, temperature: float = 0) -> dict:
    """Run pure-LLM baseline on a single ticket."""
    prompt = user_prompt_tpl.replace("{input}", input_text)
    t0 = time.time()
    response = await provider.complete(system_prompt, prompt, temperature=temperature)
    elapsed_ms = (time.time() - t0) * 1000

    try:
        content = response.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"_raw": response.content}

    return {
        "type": "baseline",
        "output": parsed,
        "raw": response.content,
        "latency_ms": elapsed_ms,
        "total_ms": response.total_ms,
    }


async def run_hybrid(provider, system_prompt: str, input_text: str,
                     rules: list, llm_fields: list, temperature: float = 0) -> dict:
    """Run hybrid (rules + LLM) on a single ticket."""
    from drift.hybrid.executor import run_hybrid as _run_hybrid

    t0 = time.time()
    result = await _run_hybrid(
        provider=provider,
        system_prompt=system_prompt,
        input_text=input_text,
        rules=rules,
        llm_fields=llm_fields,
        temperature=temperature,
        output_mode="merge",
    )
    elapsed_ms = (time.time() - t0) * 1000

    return {
        "type": "hybrid",
        "output": result.merged,
        "extracted": result.extracted,
        "llm_output": result.llm_output,
        "raw_llm": result.raw_llm_response,
        "latency_ms": elapsed_ms,
        "total_ms": result.response.total_ms if result.response else 0,
        "rules_matched": result.rules_matched,
        "rules_applied": result.rules_applied,
    }


def check_fields(output: dict, ticket: dict) -> dict:
    """Check extracted ticket fields against ground truth."""
    checks = {}
    t = output.get("ticket", {})
    checks["id"] = t.get("id") == ticket["id"]
    checks["subject"] = t.get("subject") == ticket["subject"]
    checks["requester_email"] = t.get("requester_email") == ticket["requesterEmail"]
    checks["created_at"] = t.get("created_at") == ticket["createdAt"]
    checks["stated_priority"] = t.get("stated_priority") == ticket["priority"]
    checks["stated_category"] = t.get("stated_category") == ticket["category"]

    a = output.get("assessment", {})
    checks["has_severity"] = a.get("severity") is not None
    checks["has_root_cause"] = a.get("root_cause_category") is not None
    checks["has_summary"] = a.get("summary") is not None
    checks["has_next_steps"] = isinstance(a.get("next_steps"), list) and len(a.get("next_steps", [])) > 0

    return checks


async def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs hybrid on helpdesk tickets")
    parser.add_argument("--tickets", type=int, default=10, help="Number of tickets to process")
    args = parser.parse_args()

    from drift.tasks import resolve_task_prompts
    from drift.providers.factory import make_provider

    # Load configs
    baseline_cfg = yaml.safe_load(
        (PROJECT_ROOT / "experiments" / "baseline-helpdesk.yaml").read_text()
    )
    baseline_cfg = resolve_task_prompts(baseline_cfg, PROJECT_ROOT / "tasks")

    hybrid_cfg = yaml.safe_load(
        (PROJECT_ROOT / "experiments" / "auto-hybrid-helpdesk.yaml").read_text()
    )
    hybrid_cfg = resolve_task_prompts(hybrid_cfg, PROJECT_ROOT / "tasks")

    provider = make_provider(baseline_cfg["experiment"])
    system_prompt = baseline_cfg["experiment"]["system_prompt"]
    user_prompt_tpl = baseline_cfg["experiment"]["user_prompt"]

    spec = hybrid_cfg["experiment"]["hybrid_spec"]
    rules = spec["rules"]
    llm_fields = spec["llm_fields"]

    tickets = load_tickets(args.tickets)
    print(f"Running baseline vs hybrid on {len(tickets)} tickets\n")

    results_dir = PROJECT_ROOT / "results" / "helpdesk_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)

    baseline_results = []
    hybrid_results = []

    for i, ticket in enumerate(tickets):
        input_text = format_ticket(ticket)
        print(f"Ticket {i+1}/{len(tickets)}: {ticket['id']} - {ticket['subject'][:50]}")

        # Run baseline
        print(f"  Baseline...", end="", flush=True)
        br = await run_baseline(provider, system_prompt, user_prompt_tpl, input_text)
        bc = check_fields(br["output"], ticket)
        baseline_results.append({"ticket": ticket["id"], "result": br, "checks": bc})
        print(f" {br['total_ms']:.0f}ms  fields: {sum(bc.values())}/{len(bc)}")

        # Run hybrid
        print(f"  Hybrid...", end="", flush=True)
        hr = await run_hybrid(provider, system_prompt, input_text, rules, llm_fields)
        hc = check_fields(hr["output"], ticket)
        hybrid_results.append({"ticket": ticket["id"], "result": hr, "checks": hc})
        print(f"  {hr['total_ms']:.0f}ms  fields: {sum(hc.values())}/{len(hc)}  rules: {hr['rules_matched']}/{hr['rules_applied']}")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    b_latencies = [r["result"]["total_ms"] for r in baseline_results]
    h_latencies = [r["result"]["total_ms"] for r in hybrid_results]
    b_mean = sum(b_latencies) / len(b_latencies)
    h_mean = sum(h_latencies) / len(h_latencies)
    speedup = (b_mean - h_mean) / b_mean * 100

    print(f"\n  Latency:")
    print(f"    Baseline mean:  {b_mean:,.0f} ms")
    print(f"    Hybrid mean:    {h_mean:,.0f} ms")
    print(f"    Speedup:        {speedup:.0f}%")

    # Field accuracy
    b_checks = [r["checks"] for r in baseline_results]
    h_checks = [r["checks"] for r in hybrid_results]

    print(f"\n  Field accuracy (across {len(tickets)} tickets):")
    print(f"    {'Field':<25} {'Baseline':>10} {'Hybrid':>10}")
    print(f"    {'-'*25} {'-'*10} {'-'*10}")
    all_fields = list(b_checks[0].keys())
    for field in all_fields:
        b_rate = sum(c[field] for c in b_checks) / len(b_checks) * 100
        h_rate = sum(c[field] for c in h_checks) / len(h_checks) * 100
        print(f"    {field:<25} {b_rate:>8.0f}%  {h_rate:>8.0f}%")

    b_overall = sum(sum(c.values()) for c in b_checks) / (len(b_checks) * len(all_fields)) * 100
    h_overall = sum(sum(c.values()) for c in h_checks) / (len(h_checks) * len(all_fields)) * 100
    print(f"    {'OVERALL':<25} {b_overall:>8.0f}%  {h_overall:>8.0f}%")

    # Rules stats
    matched = [r["result"]["rules_matched"] for r in hybrid_results]
    print(f"\n  Hybrid rules: {min(matched)}/{rules.__len__()} min, {max(matched)}/{len(rules)} max matched")

    # Save results
    summary = {
        "tickets": len(tickets),
        "baseline_mean_ms": round(b_mean),
        "hybrid_mean_ms": round(h_mean),
        "speedup_pct": round(speedup, 1),
        "baseline_field_accuracy": round(b_overall, 1),
        "hybrid_field_accuracy": round(h_overall, 1),
        "per_ticket": [
            {
                "id": tickets[i]["id"],
                "baseline_ms": round(b_latencies[i]),
                "hybrid_ms": round(h_latencies[i]),
                "baseline_fields": sum(b_checks[i].values()),
                "hybrid_fields": sum(h_checks[i].values()),
            }
            for i in range(len(tickets))
        ],
    }
    (results_dir / "comparison.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Results saved to {results_dir / 'comparison.json'}")


if __name__ == "__main__":
    asyncio.run(main())
