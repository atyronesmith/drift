#!/usr/bin/env python3
"""Batch rule synthesis experiment.

Runs the auto-synthesis pipeline on N training tickets, then validates
the generated rules against M different test tickets.

Usage:
    python scripts/batch_synthesize.py --train 100 --test 100
    python scripts/batch_synthesize.py --train 10 --test 10 --dry-run
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "helpdesk_tickets.jsonl"
TASK_DIR = PROJECT_ROOT / "tasks" / "helpdesk_triage"
RESULTS_DIR = PROJECT_ROOT / "results" / "batch_synthesis"


def load_tickets(path: Path) -> list[dict]:
    """Load all tickets from JSONL."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def format_ticket(ticket: dict) -> str:
    """Format a ticket as input text for the LLM."""
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


def run_probe(input_text: str, config_path: Path, output_path: Path) -> dict | None:
    """Run drift probe on a single ticket."""
    # Write input to temp file for the --input override
    input_file = output_path.parent / "temp_input.md"
    input_file.write_text(input_text)

    cmd = [
        "drift", "probe",
        str(config_path),
        "--input", str(input_file),
        "-o", str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  probe failed: {result.stderr[:200]}")
            return None
        if output_path.exists():
            return json.loads(output_path.read_text())
        return None
    except subprocess.TimeoutExpired:
        print("  probe timed out")
        return None


def validate_rules_against_ticket(rules: list[dict], input_text: str) -> dict:
    """Test how many rules match against a given input text."""
    matched = 0
    failed = 0
    results = []

    for rule in rules:
        if rule["type"] == "literal":
            # Literals always "match" but may not be correct for this ticket
            results.append({"path": rule["path"], "type": "literal", "matched": True, "value": rule["value"]})
            matched += 1
            continue

        if rule["type"] in ("regex", "regex_all"):
            pattern = rule.get("pattern", "")
            try:
                m = re.search(pattern, input_text, re.MULTILINE)
                if m:
                    value = m.group(1).strip() if m.lastindex else m.group(0).strip()
                    results.append({"path": rule["path"], "type": rule["type"], "matched": True, "value": value})
                    matched += 1
                else:
                    results.append({"path": rule["path"], "type": rule["type"], "matched": False, "value": None})
                    failed += 1
            except re.error as e:
                results.append({"path": rule["path"], "type": rule["type"], "matched": False, "error": str(e)})
                failed += 1

    return {
        "total_rules": len(rules),
        "matched": matched,
        "failed": failed,
        "match_rate": matched / len(rules) if rules else 0,
        "details": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch rule synthesis experiment")
    parser.add_argument("--train", type=int, default=10, help="Number of training tickets")
    parser.add_argument("--test", type=int, default=10, help="Number of test tickets")
    parser.add_argument("--dry-run", action="store_true", help="Format tickets only, no LLM calls")
    parser.add_argument("--threshold", type=float, default=0.3, help="Entropy threshold")
    parser.add_argument("--probe-index", type=int, default=0,
                        help="Single ticket index to probe (generates rules from this one)")
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    tickets = load_tickets(DATASET_PATH)
    total = len(tickets)
    print(f"Loaded {total} tickets")

    if args.train + args.test > total:
        print(f"Not enough tickets: need {args.train + args.test}, have {total}")
        sys.exit(1)

    train_tickets = tickets[:args.train]
    test_tickets = tickets[args.train:args.train + args.test]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Format all tickets
    print(f"\nFormatting {args.train} training + {args.test} test tickets...")
    train_inputs = [format_ticket(t) for t in train_tickets]
    test_inputs = [format_ticket(t) for t in test_tickets]

    # Save formatted inputs for inspection
    (RESULTS_DIR / "train_inputs").mkdir(exist_ok=True)
    (RESULTS_DIR / "test_inputs").mkdir(exist_ok=True)
    for i, inp in enumerate(train_inputs):
        (RESULTS_DIR / "train_inputs" / f"ticket_{i:03d}.txt").write_text(inp)
    for i, inp in enumerate(test_inputs):
        (RESULTS_DIR / "test_inputs" / f"ticket_{i:03d}.txt").write_text(inp)

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Train ticket 0:\n{train_inputs[0][:500]}")
        print(f"\nTest ticket 0:\n{test_inputs[0][:500]}")
        return

    # Step 1: Probe a single training ticket to generate rules
    probe_idx = args.probe_index
    print(f"\n=== Step 1: Probe training ticket {probe_idx} ===")
    probe_config = PROJECT_ROOT / "experiments" / "probe-helpdesk.yaml"
    if not probe_config.exists():
        print(f"Creating probe config: {probe_config}")
        probe_config.write_text(
            "experiment:\n"
            "  name: probe-helpdesk\n"
            "  model: qwen2.5:32b\n"
            "  temperature: 0\n"
            "  task: helpdesk_triage\n"
            "  type: baseline\n"
            "  system_prompt_file: triage_system.md\n"
            "  user_prompt_file: triage_user.md\n"
        )

    probe_output = RESULTS_DIR / "probe.json"
    input_file = RESULTS_DIR / "train_inputs" / f"ticket_{probe_idx:03d}.txt"

    print(f"  Probing ticket: {train_tickets[probe_idx]['id']} - {train_tickets[probe_idx]['subject']}")
    probe_data = run_probe(train_inputs[probe_idx], probe_config, probe_output)

    if not probe_data:
        print("Probe failed. Exiting.")
        sys.exit(1)

    print(f"  Probe complete: {len(probe_data.get('tokens', []))} tokens")

    # Step 2: Synthesize rules from probe data
    print(f"\n=== Step 2: Synthesize rules (threshold={args.threshold}) ===")
    from drift.hybrid.autogen import synthesize_rules, render_hybrid_yaml, format_synthesis_report

    result = synthesize_rules(probe_data, train_inputs[probe_idx], entropy_threshold=args.threshold)
    print(format_synthesis_report(result))

    hybrid_yaml = render_hybrid_yaml(result)
    hybrid_path = RESULTS_DIR / "hybrid_auto.yaml"
    hybrid_path.write_text(hybrid_yaml)

    # Extract just the actionable rules (not llm_required)
    import yaml
    spec = yaml.safe_load(hybrid_yaml)
    rules = spec.get("rules", [])
    print(f"\n  Generated {len(rules)} executable rules")

    # Step 3: Validate rules against training set
    print(f"\n=== Step 3: Validate rules against {args.train} training tickets ===")
    train_results = []
    for i, inp in enumerate(train_inputs):
        vr = validate_rules_against_ticket(rules, inp)
        train_results.append(vr)
        if i < 5 or vr["match_rate"] < 0.5:
            print(f"  Train {i:3d}: {vr['matched']}/{vr['total_rules']} rules matched ({vr['match_rate']:.0%})")

    train_rates = [r["match_rate"] for r in train_results]
    train_mean = sum(train_rates) / len(train_rates) if train_rates else 0
    train_perfect = sum(1 for r in train_rates if r == 1.0)
    print(f"\n  Training set: mean match rate = {train_mean:.1%}")
    print(f"  Perfect match (100%): {train_perfect}/{args.train}")

    # Step 4: Validate rules against test set
    print(f"\n=== Step 4: Validate rules against {args.test} TEST tickets ===")
    test_results = []
    for i, inp in enumerate(test_inputs):
        vr = validate_rules_against_ticket(rules, inp)
        test_results.append(vr)
        if i < 5 or vr["match_rate"] < 0.5:
            print(f"  Test  {i:3d}: {vr['matched']}/{vr['total_rules']} rules matched ({vr['match_rate']:.0%})")

    test_rates = [r["match_rate"] for r in test_results]
    test_mean = sum(test_rates) / len(test_rates) if test_rates else 0
    test_perfect = sum(1 for r in test_rates if r == 1.0)
    print(f"\n  Test set: mean match rate = {test_mean:.1%}")
    print(f"  Perfect match (100%): {test_perfect}/{args.test}")

    # Step 5: Analyze which rules generalize
    print(f"\n=== Step 5: Rule generalization analysis ===")
    all_inputs = train_inputs + test_inputs
    rule_match_counts = {r.get("path", f"rule_{i}"): 0 for i, r in enumerate(rules)}

    for inp in all_inputs:
        vr = validate_rules_against_ticket(rules, inp)
        for detail in vr["details"]:
            if detail["matched"]:
                rule_match_counts[detail["path"]] += 1

    total_tickets = len(all_inputs)
    print(f"\n  {'Rule Path':<40} {'Match Rate':<12} {'Type':<10} Status")
    print(f"  {'-'*40} {'-'*12} {'-'*10} {'-'*20}")
    for rule in rules:
        path = rule.get("path", "?")
        count = rule_match_counts.get(path, 0)
        rate = count / total_tickets
        rule_type = rule.get("type", "?")
        status = "GENERALIZES" if rate > 0.9 else "PARTIAL" if rate > 0.5 else "OVERFIT"
        print(f"  {path:<40} {rate:>6.0%}       {rule_type:<10} {status}")

    # Save full results
    summary = {
        "probe_ticket": train_tickets[probe_idx]["id"],
        "threshold": args.threshold,
        "num_rules": len(rules),
        "train_size": args.train,
        "test_size": args.test,
        "train_mean_match_rate": round(train_mean, 4),
        "train_perfect_matches": train_perfect,
        "test_mean_match_rate": round(test_mean, 4),
        "test_perfect_matches": test_perfect,
        "rule_generalization": {
            path: round(count / total_tickets, 4)
            for path, count in rule_match_counts.items()
        },
    }
    (RESULTS_DIR / "batch_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Results saved to {RESULTS_DIR}")

    # Final verdict
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")
    generalizing = sum(1 for rate in rule_match_counts.values() if rate / total_tickets > 0.9)
    overfit = sum(1 for rate in rule_match_counts.values() if rate / total_tickets < 0.5)
    print(f"  Rules that generalize (>90% match): {generalizing}/{len(rules)}")
    print(f"  Rules that overfit (<50% match):    {overfit}/{len(rules)}")
    print(f"  Train match rate: {train_mean:.1%}")
    print(f"  Test match rate:  {test_mean:.1%}")
    gap = train_mean - test_mean
    print(f"  Train-test gap:   {gap:.1%}")
    if gap < 0.05:
        print(f"\n  Rules generalize well across tickets!")
    elif gap < 0.15:
        print(f"\n  Moderate overfitting — some rules are input-specific.")
    else:
        print(f"\n  Significant overfitting — rules are too specific to training ticket.")


if __name__ == "__main__":
    main()
