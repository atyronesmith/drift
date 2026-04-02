#!/usr/bin/env python3
"""Experiment: MLX attention-based rule discovery vs Ollama logprob-only.

Compares two approaches to classifying output fields:
  1. Logprob entropy (existing): low entropy → extractable, high → LLM
  2. Attention attribution (new): attends to input data → extractable,
     attends to system prompt/instructions → LLM reasoning

Probes multiple helpdesk tickets, builds rules from each signal,
and validates on held-out tickets.
"""

import asyncio
import json
import math
import os
import re
import sys
import time
from pathlib import Path

# Ensure drift is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def format_ticket(t: dict) -> str:
    return (
        f"Ticket ID:    {t['id']}\n"
        f"Subject:      {t['subject']}\n"
        f"From:         {t['requesterEmail']}\n"
        f"Created:      {t['createdAt']}\n"
        f"Priority:     {t['priority']}\n"
        f"Category:     {t['category']}\n"
        f"\nDescription:\n{t['description']}"
    )


def load_tickets(path="datasets/helpdesk_tickets.jsonl"):
    tickets = []
    with open(path) as f:
        for line in f:
            tickets.append(json.loads(line))
    return tickets


def select_tickets(tickets, n_probe=6, n_test=10, seed=42):
    """Select diverse probe and test sets."""
    import random
    random.seed(seed)

    # Pick diverse probe tickets by (priority, category)
    groups = {}
    for t in tickets:
        key = (t["priority"], t["category"])
        groups.setdefault(key, []).append(t)

    target_combos = [
        ("Medium", "Network"), ("High", "Software"), ("Medium", "Software"),
        ("High", "Network"), ("Medium", "Hardware"), ("Low", "Training"),
        ("Urgent", "Security"), ("High", "Security"),
    ]
    probe = []
    for p, c in target_combos:
        if (p, c) in groups and groups[(p, c)] and len(probe) < n_probe:
            probe.append(groups[(p, c)][0])

    # Fill if needed
    probe_ids = {t["id"] for t in probe}
    if len(probe) < n_probe:
        for t in tickets:
            if t["id"] not in probe_ids:
                probe.append(t)
                probe_ids.add(t["id"])
                if len(probe) >= n_probe:
                    break

    remaining = [t for t in tickets if t["id"] not in probe_ids]
    random.shuffle(remaining)
    test = remaining[:n_test]
    return probe, test


# ── Phase 1: MLX Attention Probing ──────────────────────────────────────

def run_mlx_attention_probes(probe_tickets, system_prompt, model="mlx-community/Qwen2.5-7B-Instruct-4bit"):
    """Probe each ticket with MLX attention capture."""
    from drift.probing.mlx_attention import probe_attention

    results = []
    for i, ticket in enumerate(probe_tickets):
        input_text = format_ticket(ticket)
        print(f"  MLX attention probe [{i+1}/{len(probe_tickets)}]: {ticket['id'][:8]}...", end=" ", flush=True)
        t0 = time.time()
        result = probe_attention(
            system=system_prompt,
            prompt=input_text,
            model=model,
            max_tokens=800,
            top_k_attend=20,
        )
        elapsed = time.time() - t0
        print(f"({elapsed:.1f}s, {len(result.output_tokens)} tokens)")
        results.append({
            "ticket": ticket,
            "input_text": input_text,
            "attention_result": result,
        })
    return results


def classify_fields_by_attention(attn_results, system_prompt_len_tokens):
    """Classify each output field as extraction vs reasoning based on attention.

    A field is "extraction" if its tokens predominantly attend to input data
    positions (after the system prompt). It's "reasoning" if tokens attend
    primarily to system prompt positions.
    """
    field_scores = {}  # field_name -> list of (input_ratio, ticket_id)

    for entry in attn_results:
        result = entry["attention_result"]
        ticket = entry["ticket"]
        n_input = result.n_input

        # Parse the output JSON to find field boundaries
        output_text = ""
        field_token_ranges = {}
        current_field = None
        field_start = None

        for i, probe in enumerate(result.output_tokens):
            output_text += probe.token

            # Detect JSON field names
            # Look for "field_name": pattern
            for match in re.finditer(r'"(\w+(?:\.\w+)*)":\s*$', output_text):
                if current_field and field_start is not None:
                    field_token_ranges[current_field] = (field_start, i)
                current_field = match.group(1)
                field_start = i + 1

        # Close last field
        if current_field and field_start is not None:
            field_token_ranges[current_field] = (field_start, len(result.output_tokens))

        # For each field, compute attention distribution
        for field_name, (start, end) in field_token_ranges.items():
            input_data_weight = 0
            system_prompt_weight = 0
            total_weight = 0

            for idx in range(start, min(end, len(result.output_tokens))):
                probe = result.output_tokens[idx]
                for pos, weight in probe.top_attended:
                    if math.isnan(weight):
                        continue
                    total_weight += weight
                    # System prompt tokens are roughly the first portion
                    # Input data starts after system prompt
                    if pos < system_prompt_len_tokens:
                        system_prompt_weight += weight
                    else:
                        input_data_weight += weight

            if total_weight > 0:
                input_ratio = input_data_weight / total_weight
            else:
                input_ratio = 0

            field_scores.setdefault(field_name, []).append(
                (input_ratio, ticket["id"])
            )

    # Aggregate across tickets
    field_classification = {}
    for field_name, scores in field_scores.items():
        ratios = [s[0] for s in scores]
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0
        field_classification[field_name] = {
            "avg_input_attention_ratio": avg_ratio,
            "n_tickets": len(scores),
            "ratios": ratios,
            "classification": "extraction" if avg_ratio > 0.5 else "reasoning",
        }

    return field_classification


# ── Phase 2: MLX Logprob Probing ────────────────────────────────────────

def run_mlx_logprob_probes(probe_tickets, system_prompt, model="mlx-community/Qwen2.5-7B-Instruct-4bit"):
    """Probe each ticket with MLX logprobs (entropy-only, no attention)."""
    from drift.probing.mlx_logprobs import probe_with_mlx

    results = []
    for i, ticket in enumerate(probe_tickets):
        input_text = format_ticket(ticket)
        print(f"  MLX logprob probe [{i+1}/{len(probe_tickets)}]: {ticket['id'][:8]}...", end=" ", flush=True)
        t0 = time.time()
        result = probe_with_mlx(
            system=system_prompt,
            prompt=input_text,
            model=model,
            max_tokens=800,
        )
        elapsed = time.time() - t0
        print(f"({elapsed:.1f}s, {len(result.tokens)} tokens)")
        results.append({
            "ticket": ticket,
            "input_text": input_text,
            "logprob_result": result,
        })
    return results


def classify_fields_by_entropy(logprob_results, entropy_threshold=0.3):
    """Classify fields by entropy (existing approach)."""
    field_entropies = {}

    for entry in logprob_results:
        result = entry["logprob_result"]
        ticket = entry["ticket"]

        output_text = ""
        current_field = None
        field_start = None
        field_token_ranges = {}

        for i, token_probe in enumerate(result.tokens):
            output_text += token_probe.token

            for match in re.finditer(r'"(\w+(?:\.\w+)*)":\s*$', output_text):
                if current_field and field_start is not None:
                    field_token_ranges[current_field] = (field_start, i)
                current_field = match.group(1)
                field_start = i + 1

        if current_field and field_start is not None:
            field_token_ranges[current_field] = (field_start, len(result.tokens))

        for field_name, (start, end) in field_token_ranges.items():
            entropies = []
            for idx in range(start, min(end, len(result.tokens))):
                entropies.append(result.tokens[idx].entropy)
            if entropies:
                avg_entropy = sum(entropies) / len(entropies)
                field_entropies.setdefault(field_name, []).append(
                    (avg_entropy, ticket["id"])
                )

    field_classification = {}
    for field_name, scores in field_entropies.items():
        avg_ent = sum(s[0] for s in scores) / len(scores) if scores else 0
        field_classification[field_name] = {
            "avg_entropy": avg_ent,
            "n_tickets": len(scores),
            "entropies": [s[0] for s in scores],
            "classification": "extraction" if avg_ent < entropy_threshold else "reasoning",
        }

    return field_classification


# ── Phase 3: Rule Generation ────────────────────────────────────────────

# These are the rules from attention analysis — we'll discover which fields
# attend to input and build regex rules for them
TICKET_EXTRACTION_RULES = {
    "ticket.id": r"Ticket\s+ID:\s*(\S+)",
    "ticket.subject": r"Subject:\s*(.+?)(?:\n|$)",
    "ticket.requester_email": r"From:\s*([\w.+-]+@[\w.-]+)",
    "ticket.created_at": r"Created:\s*(\S+)",
    "ticket.stated_priority": r"Priority:\s*(\S+)",
    "ticket.stated_category": r"Category:\s*(\S+)",
}

SEVERITY_CONDITIONAL = {
    ("Urgent", None): "P1",
    ("High", "Account"): "P3",
    ("High", None): "P2",
    ("Medium", None): "P3",
    ("Low", None): "P4",
}


def extract_with_rules(input_text, rules):
    """Apply regex rules to extract fields."""
    extracted = {}
    for field, pattern in rules.items():
        m = re.search(pattern, input_text, re.MULTILINE)
        if m:
            extracted[field] = m.group(1).strip()
    return extracted


def apply_severity_rule(extracted):
    """Apply conditional severity mapping."""
    priority = extracted.get("ticket.stated_priority", "")
    category = extracted.get("ticket.stated_category", "")

    # Check specific (priority, category) first
    for (p, c), sev in SEVERITY_CONDITIONAL.items():
        if c is not None and p == priority and c == category:
            return sev
    # Then priority-only
    for (p, c), sev in SEVERITY_CONDITIONAL.items():
        if c is None and p == priority:
            return sev
    return "P3"


# ── Phase 4: Validation ────────────────────────────────────────────────

def validate_rules(test_tickets, system_prompt, rules, model="mlx-community/Qwen2.5-7B-Instruct-4bit"):
    """Run LLM on test tickets, compare rule extractions with LLM output."""
    from drift.providers.mlx import MLXProvider

    provider = MLXProvider(model=model)
    results = []

    for i, ticket in enumerate(test_tickets):
        input_text = format_ticket(ticket)
        print(f"  Validating [{i+1}/{len(test_tickets)}]: {ticket['id'][:8]}...", end=" ", flush=True)

        # Get LLM output (ground truth)
        t0 = time.time()
        llm_resp = asyncio.run(provider.complete(
            system=system_prompt, prompt=input_text,
            max_tokens=800, temperature=0,
        ))
        elapsed = time.time() - t0

        # Parse LLM JSON
        content = llm_resp.content
        # Strip markdown fencing if present
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        try:
            llm_json = json.loads(content)
        except json.JSONDecodeError:
            print(f"JSON parse error, skipping")
            continue

        # Extract with rules
        rule_extracted = extract_with_rules(input_text, rules)
        rule_severity = apply_severity_rule(rule_extracted)

        # Compare
        ticket_result = {
            "ticket_id": ticket["id"],
            "priority": ticket["priority"],
            "category": ticket["category"],
            "llm_ms": round(llm_resp.total_ms, 1),
            "matches": {},
            "mismatches": {},
        }

        # Compare extraction fields
        llm_ticket = llm_json.get("ticket", {})
        field_map = {
            "ticket.id": llm_ticket.get("id", ""),
            "ticket.subject": llm_ticket.get("subject", ""),
            "ticket.requester_email": llm_ticket.get("requester_email", ""),
            "ticket.created_at": llm_ticket.get("created_at", ""),
            "ticket.stated_priority": llm_ticket.get("stated_priority", ""),
            "ticket.stated_category": llm_ticket.get("stated_category", ""),
        }

        for field, llm_val in field_map.items():
            rule_val = rule_extracted.get(field, "")
            if str(rule_val).strip() == str(llm_val).strip():
                ticket_result["matches"][field] = rule_val
            else:
                ticket_result["mismatches"][field] = {
                    "rule": rule_val, "llm": llm_val
                }

        # Compare severity
        llm_severity = llm_json.get("assessment", {}).get("severity", "")
        if rule_severity == llm_severity:
            ticket_result["matches"]["assessment.severity"] = rule_severity
        else:
            ticket_result["mismatches"]["assessment.severity"] = {
                "rule": rule_severity, "llm": llm_severity
            }

        n_match = len(ticket_result["matches"])
        n_total = n_match + len(ticket_result["mismatches"])
        print(f"({elapsed:.1f}s, {n_match}/{n_total} fields match)")
        results.append(ticket_result)

    return results


# ── Main ────────────────────────────────────────────────────────────────

def main():
    system_prompt = open("tasks/helpdesk_triage/triage_system.md").read()
    tickets = load_tickets()
    probe_tickets, test_tickets = select_tickets(tickets)

    print("=" * 70)
    print("EXPERIMENT: Attention-Based vs Logprob-Only Rule Discovery")
    print("=" * 70)
    print(f"Probe tickets: {len(probe_tickets)}")
    print(f"Test tickets:  {len(test_tickets)}")
    print(f"Model: mlx-community/Qwen2.5-7B-Instruct-4bit")
    print()

    model = "mlx-community/Qwen2.5-7B-Instruct-4bit"

    # ── Phase 1: MLX Attention Probing ──
    print("─" * 70)
    print("Phase 1: MLX Attention Probing")
    print("─" * 70)
    attn_results = run_mlx_attention_probes(probe_tickets, system_prompt, model)

    # Estimate system prompt token count
    sys_tokens = len(attn_results[0]["attention_result"].input_tokens)
    # System prompt is roughly the first part of input tokens before ticket data
    # The formatted input starts after the system prompt in the chat template
    # Approximate: count tokens in system prompt vs total input
    from mlx_lm import load as mlx_load
    _, tokenizer = mlx_load(model)
    sys_only = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}],
        tokenize=False, add_generation_prompt=False,
    )
    sys_prompt_token_count = len(tokenizer.encode(sys_only))
    print(f"\nSystem prompt tokens: ~{sys_prompt_token_count}")

    attn_classification = classify_fields_by_attention(attn_results, sys_prompt_token_count)

    print("\nAttention-based field classification:")
    print(f"  {'Field':<30s} {'Avg Input Attn':>15s} {'Class':>12s}  {'N':>3s}")
    print(f"  {'─'*30} {'─'*15} {'─'*12}  {'─'*3}")
    for field, info in sorted(attn_classification.items()):
        print(f"  {field:<30s} {info['avg_input_attention_ratio']:>14.1%} {info['classification']:>12s}  {info['n_tickets']:>3d}")

    # ── Phase 2: MLX Logprob Probing ──
    print()
    print("─" * 70)
    print("Phase 2: MLX Logprob Probing (entropy-only, same model)")
    print("─" * 70)
    logprob_results = run_mlx_logprob_probes(probe_tickets, system_prompt, model)

    entropy_classification = classify_fields_by_entropy(logprob_results)

    print("\nEntropy-based field classification:")
    print(f"  {'Field':<30s} {'Avg Entropy':>12s} {'Class':>12s}  {'N':>3s}")
    print(f"  {'─'*30} {'─'*12} {'─'*12}  {'─'*3}")
    for field, info in sorted(entropy_classification.items()):
        print(f"  {field:<30s} {info['avg_entropy']:>11.3f} {info['classification']:>12s}  {info['n_tickets']:>3d}")

    # ── Phase 3: Comparison ──
    print()
    print("─" * 70)
    print("Phase 3: Comparison — Attention vs Entropy Classification")
    print("─" * 70)

    all_fields = sorted(set(attn_classification.keys()) | set(entropy_classification.keys()))
    agreements = 0
    disagreements = 0
    print(f"  {'Field':<30s} {'Attention':>12s} {'Entropy':>12s} {'Agree?':>8s}")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*8}")
    for field in all_fields:
        a_class = attn_classification.get(field, {}).get("classification", "?")
        e_class = entropy_classification.get(field, {}).get("classification", "?")
        agree = a_class == e_class
        if agree:
            agreements += 1
        else:
            disagreements += 1
        marker = "YES" if agree else "*** NO"
        print(f"  {field:<30s} {a_class:>12s} {e_class:>12s} {marker:>8s}")
    print(f"\n  Agreement: {agreements}/{agreements+disagreements}")

    # ── Phase 4: Show detailed attention for interesting fields ──
    print()
    print("─" * 70)
    print("Phase 4: Detailed Attention Attribution for Key Fields")
    print("─" * 70)

    for entry in attn_results[:3]:  # first 3 tickets
        result = entry["attention_result"]
        ticket = entry["ticket"]
        print(f"\n  Ticket: {ticket['id'][:8]} ({ticket['priority']}/{ticket['category']})")
        print(f"  Output preview: {result.content[:100]}...")

        # Show per-field attribution
        output_text = ""
        current_field = None
        field_start = None

        for i, probe in enumerate(result.output_tokens):
            output_text += probe.token
            for match in re.finditer(r'"(\w+(?:\.\w+)*)":\s*$', output_text):
                if current_field and field_start is not None:
                    # Show this field's attribution
                    field_indices = list(range(field_start, i))
                    if field_indices:
                        attrib = result.field_attribution("", field_indices, top_k=5)
                        value_text = "".join(
                            result.output_tokens[j].token
                            for j in field_indices
                            if j < len(result.output_tokens)
                        ).strip()[:40]
                        # Categorize attention
                        input_w = sum(w for t, w in attrib if not math.isnan(w))
                        top_str = ", ".join(
                            f"{t!r}({w:.3f})"
                            for t, w in attrib[:3]
                            if not math.isnan(w)
                        )
                        print(f"    {current_field:<28s} = {value_text!r:>42s}  ← {top_str}")
                current_field = match.group(1)
                field_start = i + 1

    # ── Phase 5: Validate rules on held-out tickets ──
    print()
    print("─" * 70)
    print("Phase 5: Validation on Held-Out Tickets")
    print("─" * 70)

    # Determine which rules to use based on attention classification
    attn_rules = {}
    for field, info in attn_classification.items():
        if info["classification"] == "extraction":
            # Map field name to our extraction rules
            for rule_field, pattern in TICKET_EXTRACTION_RULES.items():
                # Match field names loosely
                short = rule_field.split(".")[-1]
                if short in field or field in short:
                    attn_rules[rule_field] = pattern

    entropy_rules = {}
    for field, info in entropy_classification.items():
        if info["classification"] == "extraction":
            for rule_field, pattern in TICKET_EXTRACTION_RULES.items():
                short = rule_field.split(".")[-1]
                if short in field or field in short:
                    entropy_rules[rule_field] = pattern

    print(f"\nAttention-derived rules: {len(attn_rules)} extraction fields")
    for f in sorted(attn_rules):
        print(f"  {f}")
    print(f"\nEntropy-derived rules: {len(entropy_rules)} extraction fields")
    for f in sorted(entropy_rules):
        print(f"  {f}")

    # Validate with ALL extraction rules (superset)
    print(f"\nValidating ALL extraction rules + severity conditional on {len(test_tickets)} tickets...")
    validation_results = validate_rules(
        test_tickets, system_prompt, TICKET_EXTRACTION_RULES, model
    )

    # Summary
    total_matches = sum(len(r["matches"]) for r in validation_results)
    total_mismatches = sum(len(r["mismatches"]) for r in validation_results)
    total_fields = total_matches + total_mismatches
    sev_matches = sum(1 for r in validation_results if "assessment.severity" in r["matches"])

    print(f"\n{'─' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'─' * 70}")
    print(f"Extraction rule accuracy: {total_matches}/{total_fields} ({100*total_matches/total_fields:.1f}%)")
    print(f"Severity rule accuracy:   {sev_matches}/{len(validation_results)} ({100*sev_matches/len(validation_results):.1f}%)")

    # Show mismatches
    if total_mismatches > 0:
        print(f"\nMismatches:")
        for r in validation_results:
            for field, vals in r["mismatches"].items():
                print(f"  {r['ticket_id'][:8]} {field}: rule={vals['rule']!r} vs llm={vals['llm']!r}")

    # Save results
    output = {
        "experiment": "attention_vs_logprobs",
        "model": model,
        "n_probe_tickets": len(probe_tickets),
        "n_test_tickets": len(test_tickets),
        "attention_classification": {
            k: {kk: vv for kk, vv in v.items() if kk != "ratios"}
            for k, v in attn_classification.items()
        },
        "entropy_classification": {
            k: {kk: vv for kk, vv in v.items() if kk != "entropies"}
            for k, v in entropy_classification.items()
        },
        "validation": {
            "extraction_accuracy": total_matches / total_fields if total_fields else 0,
            "severity_accuracy": sev_matches / len(validation_results) if validation_results else 0,
            "total_matches": total_matches,
            "total_mismatches": total_mismatches,
            "details": validation_results,
        },
    }

    os.makedirs("results", exist_ok=True)
    out_path = "results/attention_vs_logprobs.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
