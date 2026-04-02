#!/usr/bin/env python3
"""Attention field mapping: discover which input fields drive each output field.

Filters out system prompt and structural tokens, maps remaining attention
to input field regions (Ticket ID:, Subject:, Priority:, etc.), and reports
which input fields each output field attends to — candidates for mapping rules.
"""

import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

MODEL = "mlx-community/Qwen2.5-32B-Instruct-4bit"

# Input field labels and their regex patterns in the formatted ticket
INPUT_FIELD_LABELS = {
    "ticket_id": r"Ticket\s+ID:",
    "subject": r"Subject:",
    "from": r"From:",
    "created": r"Created:",
    "priority": r"Priority:",
    "category": r"Category:",
    "description": r"Description:",
}


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


def find_input_field_regions(input_tokens, sys_prompt_end):
    """Map each input token (after system prompt) to its input field region.

    Returns: list of (field_name, start_pos, end_pos) and
             dict mapping each token position to its field name.
    """
    # Reconstruct the user input text from tokens after system prompt
    # First, find where each field label appears in the token sequence
    token_text = ""
    token_offsets = []  # (start_char, end_char) for each token
    for i in range(sys_prompt_end, len(input_tokens)):
        start = len(token_text)
        token_text += input_tokens[i]
        end = len(token_text)
        token_offsets.append((start, end, i))

    # Find field label positions in the reconstructed text
    field_positions = []
    for field_name, pattern in INPUT_FIELD_LABELS.items():
        m = re.search(pattern, token_text)
        if m:
            field_positions.append((m.start(), field_name))

    field_positions.sort()

    # Map each token to a field region
    pos_to_field = {}
    for idx, (fp_start, field_name) in enumerate(field_positions):
        # Field region extends to the start of the next field (or end)
        if idx + 1 < len(field_positions):
            fp_end = field_positions[idx + 1][0]
        else:
            fp_end = len(token_text)

        # Find tokens in this range
        for char_start, char_end, token_pos in token_offsets:
            if char_start >= fp_start and char_start < fp_end:
                pos_to_field[token_pos] = field_name

    return pos_to_field


def analyze_ticket(provider, ticket, system_prompt, sys_prompt_end):
    """Probe one ticket and analyze field-to-field attention mapping."""
    input_text = format_ticket(ticket)

    result = provider.probe_attention(
        system=system_prompt,
        prompt=input_text,
        max_tokens=800,
        top_k_attend=50,  # get more positions for filtering
    )

    # Map input token positions to field regions
    pos_to_field = find_input_field_regions(result.input_tokens, sys_prompt_end)

    # Parse output fields from generated JSON
    output_text = ""
    current_field = None
    field_start = None
    field_token_ranges = {}

    for i, probe in enumerate(result.output_tokens):
        output_text += probe.token
        for match in re.finditer(r'"(\w+(?:\.\w+)*)":\s*$', output_text):
            if current_field and field_start is not None:
                field_token_ranges[current_field] = (field_start, i)
            current_field = match.group(1)
            field_start = i + 1

    if current_field and field_start is not None:
        field_token_ranges[current_field] = (field_start, len(result.output_tokens))

    # For each output field, compute attention distribution over INPUT FIELDS
    field_mapping = {}
    for out_field, (start, end) in field_token_ranges.items():
        # Aggregate attention from output tokens to input field regions
        field_weights = defaultdict(float)
        system_weight = 0.0
        structural_weight = 0.0  # \n, special tokens
        total_weight = 0.0

        for idx in range(start, min(end, len(result.output_tokens))):
            probe = result.output_tokens[idx]
            for pos, weight in probe.top_attended:
                if math.isnan(weight):
                    continue
                total_weight += weight

                if pos < sys_prompt_end:
                    system_weight += weight
                elif pos in pos_to_field:
                    field_weights[pos_to_field[pos]] += weight
                else:
                    # Token in user input but not mapped to a field
                    # (e.g., chat template tokens between system and user)
                    structural_weight += weight

        if total_weight == 0:
            continue

        # Normalize to percentages
        field_pcts = {
            f: w / total_weight for f, w in field_weights.items()
        }
        sys_pct = system_weight / total_weight
        struct_pct = structural_weight / total_weight

        # Get the value the model output for this field
        value_text = "".join(
            result.output_tokens[j].token
            for j in range(start, min(end, len(result.output_tokens)))
        ).strip()[:60]

        field_mapping[out_field] = {
            "value": value_text,
            "system_prompt_pct": sys_pct,
            "structural_pct": struct_pct,
            "input_field_pcts": dict(sorted(field_pcts.items(), key=lambda x: x[1], reverse=True)),
            "top_input_field": max(field_pcts, key=field_pcts.get) if field_pcts else None,
            "top_input_field_pct": max(field_pcts.values()) if field_pcts else 0,
        }

    return field_mapping, result


def main():
    from drift.providers.mlx_remote import MLXRemoteProvider
    provider = MLXRemoteProvider(model=MODEL)

    system_prompt = open("tasks/helpdesk_triage/triage_system.md").read()
    tickets = load_tickets()

    # Select diverse tickets
    target = [
        ("Medium", "Network"), ("High", "Software"), ("Urgent", "Security"),
        ("Medium", "Account"), ("High", "Network"), ("Low", "Training"),
    ]
    probe_tickets = []
    used = set()
    for p, c in target:
        for t in tickets:
            if t["priority"] == p and t["category"] == c and t["id"] not in used:
                probe_tickets.append(t)
                used.add(t["id"])
                break
    probe_tickets = probe_tickets[:6]

    print("=" * 70)
    print("ATTENTION FIELD MAPPING: Input Field -> Output Field Attribution")
    print("=" * 70)
    print(f"Server: {provider.base_url}")
    print(f"Model:  {MODEL}")
    print(f"Tickets: {len(probe_tickets)}")
    print()

    # Estimate system prompt boundary
    print("Estimating system prompt boundary...", end=" ", flush=True)
    boundary_result = provider.probe_attention(
        system=system_prompt, prompt="test", max_tokens=1, top_k_attend=5,
    )
    sys_prompt_end = 0
    for i, tok in enumerate(boundary_result.input_tokens):
        if "test" in tok.lower():
            sys_prompt_end = i
            break
    if sys_prompt_end == 0:
        sys_prompt_end = max(boundary_result.n_input - 10, 0)
    print(f"token {sys_prompt_end}")
    print()

    # Aggregate across tickets
    all_mappings = defaultdict(lambda: defaultdict(list))
    first_raw_result = None  # Save first probe for drift synthesize --attention

    for i, ticket in enumerate(probe_tickets):
        print(f"─" * 70)
        print(f"Ticket {i+1}/{len(probe_tickets)}: {ticket['id'][:8]} "
              f"({ticket['priority']}/{ticket['category']})")
        print(f"─" * 70)

        t0 = time.time()
        mapping, result = analyze_ticket(provider, ticket, system_prompt, sys_prompt_end)
        elapsed = time.time() - t0
        print(f"  Probed in {elapsed:.1f}s ({len(result.output_tokens)} tokens)")

        if first_raw_result is None:
            first_raw_result = result
        print()

        print(f"  {'Output Field':<24s} {'Value':<35s} {'Top Input Field':<16s} {'%':>6s}  {'Sys%':>5s}")
        print(f"  {'─'*24} {'─'*35} {'─'*16} {'─'*6}  {'─'*5}")

        for out_field, info in mapping.items():
            top_f = info["top_input_field"] or "-"
            top_pct = info["top_input_field_pct"]
            sys_pct = info["system_prompt_pct"]
            val = info["value"][:33]
            print(f"  {out_field:<24s} {val:<35s} {top_f:<16s} {top_pct:>5.1%}  {sys_pct:>5.1%}")

            # Track for aggregation
            for input_f, pct in info["input_field_pcts"].items():
                all_mappings[out_field][input_f].append(pct)

        print()

    # ── Aggregate Summary ──
    print()
    print("=" * 70)
    print("AGGREGATE: Average Input Field Attention per Output Field")
    print("=" * 70)
    print()
    print(f"  {'Output Field':<24s}", end="")
    input_fields = sorted(INPUT_FIELD_LABELS.keys())
    for f in input_fields:
        print(f" {f:>12s}", end="")
    print(f"  {'Rule Candidate?':>16s}")

    print(f"  {'─'*24}", end="")
    for _ in input_fields:
        print(f" {'─'*12}", end="")
    print(f"  {'─'*16}")

    rule_candidates = {}

    for out_field in sorted(all_mappings.keys()):
        print(f"  {out_field:<24s}", end="")
        for f in input_fields:
            vals = all_mappings[out_field].get(f, [])
            avg = sum(vals) / len(vals) if vals else 0
            print(f" {avg:>11.1%}", end="")

        sorted_fields = sorted(
            [(f, sum(all_mappings[out_field].get(f, [0])) / max(len(all_mappings[out_field].get(f, [1])), 1))
             for f in input_fields],
            key=lambda x: x[1], reverse=True,
        )
        top_pct = sorted_fields[0][1] if sorted_fields else 0
        second_pct = sorted_fields[1][1] if len(sorted_fields) > 1 else 0

        if top_pct > 0.08 and (second_pct == 0 or top_pct / max(second_pct, 0.001) > 1.5):
            rule_type = f"map({sorted_fields[0][0]})"
            rule_candidates[out_field] = sorted_fields[0][0]
        elif top_pct > 0.05:
            rule_type = f"weak({sorted_fields[0][0]})"
        else:
            rule_type = "reasoning"

        print(f"  {rule_type:>16s}")

    # ── Per-token density analysis ──
    # Description has many tokens, inflating its raw sum.
    # Normalize by token count per field region to find true signal.
    print()
    print()
    print("=" * 70)
    print("PER-TOKEN ATTENTION DENSITY (normalized by field token count)")
    print("=" * 70)
    print()
    print("  Raw sums favor long fields (description). Per-token density")
    print("  reveals which short fields get disproportionate attention.")
    print()

    # We need token counts per field region — re-probe one ticket to get them
    # Use the region sizes from the first probe
    sample_ticket = probe_tickets[0]
    input_text = format_ticket(sample_ticket)
    sample_result = provider.probe_attention(
        system=system_prompt, prompt=input_text,
        max_tokens=1, top_k_attend=5,
    )

    pos_to_field = find_input_field_regions(sample_result.input_tokens, sys_prompt_end)
    field_token_counts = defaultdict(int)
    for pos, field in pos_to_field.items():
        field_token_counts[field] += 1

    print(f"  Input field token counts:")
    for f in input_fields:
        print(f"    {f:<16s}: {field_token_counts.get(f, 0):>4d} tokens")
    print()

    print(f"  {'Output Field':<24s}", end="")
    for f in input_fields:
        print(f" {f:>12s}", end="")
    print(f"  {'Dominant?':>16s}")
    print(f"  {'─'*24}", end="")
    for _ in input_fields:
        print(f" {'─'*12}", end="")
    print(f"  {'─'*16}")

    density_candidates = {}

    for out_field in sorted(all_mappings.keys()):
        print(f"  {out_field:<24s}", end="")
        densities = {}
        for f in input_fields:
            vals = all_mappings[out_field].get(f, [])
            avg_pct = sum(vals) / len(vals) if vals else 0
            n_tokens = field_token_counts.get(f, 1)
            density = avg_pct / n_tokens * 100  # attention per token (scaled)
            densities[f] = density
            print(f" {density:>11.2f}", end="")

        # Find dominant field by density (excluding description)
        non_desc = {f: d for f, d in densities.items() if f != "description"}
        desc_density = densities.get("description", 0)

        if non_desc:
            top_f = max(non_desc, key=non_desc.get)
            top_d = non_desc[top_f]
            # Check if any short field has higher per-token density than description
            if top_d > desc_density:
                label = f"** {top_f}"
                density_candidates[out_field] = top_f
            elif top_d > desc_density * 0.5:
                label = f"~ {top_f}"
            else:
                label = "description"
        else:
            label = "description"

        print(f"  {label:>16s}")

    print()
    print("─" * 70)
    print("FIELDS WITH HIGHER PER-TOKEN DENSITY THAN DESCRIPTION")
    print("─" * 70)
    if density_candidates:
        for out_field, in_field in density_candidates.items():
            print(f"  {out_field:<24s} <- {in_field}")
    else:
        print("  (none — description dominates per-token density too)")

    print()
    print("─" * 70)
    print("DISCOVERED MAPPING CANDIDATES (raw sum)")
    print("─" * 70)
    for out_field, in_field in rule_candidates.items():
        pcts = all_mappings[out_field][in_field]
        avg_pct = sum(pcts) / len(pcts)
        print(f"  {out_field:<24s} <- {in_field:<16s} (avg {avg_pct:.1%} attention)")

    # Save
    os.makedirs("results", exist_ok=True)
    output = {
        "experiment": "attention_field_mapping",
        "model": MODEL,
        "n_tickets": len(probe_tickets),
        "sys_prompt_end_token": sys_prompt_end,
        "aggregate": {
            out: {
                inp: sum(vals) / len(vals)
                for inp, vals in fields.items()
            }
            for out, fields in all_mappings.items()
        },
        "rule_candidates": rule_candidates,
    }
    out_path = "results/attention_field_mapping.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save raw attention probe for use with: drift synthesize --attention
    if first_raw_result:
        raw = {
            "input_tokens": first_raw_result.input_tokens,
            "n_input": first_raw_result.n_input,
            "output_tokens": [
                {
                    "token": t.token,
                    "token_id": t.token_id,
                    "position": t.position,
                    "top_attended": t.top_attended,
                }
                for t in first_raw_result.output_tokens
            ],
            "content": first_raw_result.content,
            "model": MODEL,
        }
        raw_path = "results/attention_probe_raw.json"
        with open(raw_path, "w") as f:
            json.dump(raw, f)
        print(f"Raw attention probe saved to {raw_path}")


if __name__ == "__main__":
    main()
