#!/usr/bin/env python3
"""Test whether severity is conditionally deterministic given (priority, category).

Probes multiple tickets across different priority × category combinations
and checks if the LLM assigns consistent severity within each group.

Usage:
    python scripts/conditional_severity.py --per-group 3
"""

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "helpdesk_tickets.jsonl"


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


def parse_severity(response_text: str) -> str | None:
    """Extract severity from LLM JSON response."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        parsed = json.loads(text)
        return parsed.get("assessment", {}).get("severity")
    except json.JSONDecodeError:
        return None


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-group", type=int, default=3,
                        help="Tickets per (priority, category) group")
    parser.add_argument("--min-group-size", type=int, default=5,
                        help="Only test groups with at least this many tickets in dataset")
    args = parser.parse_args()

    import yaml
    from drift.tasks import resolve_task_prompts
    from drift.providers.factory import make_provider

    # Load config
    config = yaml.safe_load(
        (PROJECT_ROOT / "experiments" / "baseline-helpdesk.yaml").read_text()
    )
    config = resolve_task_prompts(config, PROJECT_ROOT / "tasks")
    exp = config["experiment"]
    provider = make_provider(exp)
    system_prompt = exp["system_prompt"]
    user_prompt_tpl = exp["user_prompt"]

    # Load and group tickets
    with open(DATASET_PATH) as f:
        all_tickets = [json.loads(line) for line in f]

    groups = defaultdict(list)
    for t in all_tickets:
        groups[(t["priority"], t["category"])].append(t)

    # Select groups with enough tickets
    test_groups = {
        k: v for k, v in groups.items()
        if len(v) >= args.min_group_size
    }
    print(f"Groups with >= {args.min_group_size} tickets: {len(test_groups)}")

    # Sample tickets from each group
    samples = []
    for (priority, category), tickets in sorted(test_groups.items()):
        for t in tickets[:args.per_group]:
            samples.append(t)

    print(f"Total tickets to probe: {len(samples)}")
    print()

    # Run each ticket through the LLM
    results = []
    for i, ticket in enumerate(samples):
        input_text = format_ticket(ticket)
        prompt = user_prompt_tpl.replace("{input}", input_text)

        print(f"  [{i+1:2d}/{len(samples)}] {ticket['priority']:8s} × {ticket['category']:15s}  ", end="", flush=True)
        response = await provider.complete(system_prompt, prompt, temperature=0)
        severity = parse_severity(response.content)
        print(f"→ {severity}  ({response.total_ms:.0f}ms)")

        results.append({
            "id": ticket["id"],
            "priority": ticket["priority"],
            "category": ticket["category"],
            "subject": ticket["subject"][:60],
            "severity": severity,
            "latency_ms": response.total_ms,
        })

    # Analyze: is severity consistent within each (priority, category) group?
    print()
    print("=" * 70)
    print("CONDITIONAL DETERMINISM ANALYSIS")
    print("=" * 70)

    group_results = defaultdict(list)
    for r in results:
        group_results[(r["priority"], r["category"])].append(r["severity"])

    deterministic_groups = 0
    total_groups = len(group_results)

    print(f"\n  {'Priority':<10} {'Category':<18} {'Severities':<25} {'Deterministic?'}")
    print(f"  {'-'*10} {'-'*18} {'-'*25} {'-'*15}")

    for (priority, category), severities in sorted(group_results.items()):
        unique = set(severities)
        is_deterministic = len(unique) == 1
        if is_deterministic:
            deterministic_groups += 1
        status = "YES" if is_deterministic else "NO"
        sev_str = ", ".join(severities)
        print(f"  {priority:<10} {category:<18} {sev_str:<25} {status}")

    print(f"\n  Deterministic groups: {deterministic_groups}/{total_groups}")
    print(f"  Rate: {deterministic_groups/total_groups*100:.0f}%")

    # If deterministic, show the mapping
    if deterministic_groups > 0:
        print(f"\n  POTENTIAL LOOKUP TABLE:")
        print(f"  {'-'*50}")
        for (priority, category), severities in sorted(group_results.items()):
            if len(set(severities)) == 1:
                print(f"  {priority:8s} + {category:15s} → {severities[0]}")

    # Show non-deterministic groups with detail
    non_det = [(k, v) for k, v in group_results.items() if len(set(v)) > 1]
    if non_det:
        print(f"\n  NON-DETERMINISTIC GROUPS (severity varies):")
        print(f"  {'-'*50}")
        for (priority, category), severities in non_det:
            tickets_in_group = [r for r in results
                                if r["priority"] == priority and r["category"] == category]
            for r in tickets_in_group:
                print(f"  {r['severity']}  {r['subject']}")

    # Save results
    results_dir = PROJECT_ROOT / "results" / "conditional_severity"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results.json").write_text(json.dumps(results, indent=2))
    (results_dir / "mapping.json").write_text(json.dumps({
        f"{p}+{c}": list(set(sevs))
        for (p, c), sevs in group_results.items()
    }, indent=2))
    print(f"\n  Results saved to {results_dir}")


if __name__ == "__main__":
    asyncio.run(main())
