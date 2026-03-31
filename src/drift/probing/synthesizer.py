"""Rule synthesizer: analyze probe data to identify extractable rules.

Takes logprob probe output and maps token-level entropy to JSON field paths,
producing a field-by-field assessment of what can be replaced with deterministic
code vs. what requires LLM inference.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class FieldRule:
    path: str
    value: str | int | float | list | None
    rule_type: str  # "literal", "extracted", "computed", "llm_required"
    confidence: float  # 0-1, based on token entropy
    entropy_tokens: list[dict] = field(default_factory=list)
    source_pattern: str | None = None


def _flatten_json(obj, prefix="") -> list[tuple[str, str, int, int]]:
    """Flatten JSON to (path, value_str, start_char, end_char) tuples."""
    results = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            results.extend(_flatten_json(val, new_prefix))
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            new_prefix = f"{prefix}[{i}]"
            results.extend(_flatten_json(val, new_prefix))
    else:
        results.append((prefix, str(obj) if obj is not None else "null", obj))
    return results


def _map_tokens_to_chars(tokens: list[dict]) -> list[tuple[int, int]]:
    """Map each token index to (start_char, end_char) in the full content string."""
    positions = []
    pos = 0
    for t in tokens:
        token_text = t["token"]
        positions.append((pos, pos + len(token_text)))
        pos += len(token_text)
    return positions


def _find_field_tokens(
    content: str,
    tokens: list[dict],
    token_positions: list[tuple[int, int]],
    field_path: str,
    field_value: str,
) -> list[int]:
    """Find which token indices correspond to a field's value in the JSON."""
    # Find the value string in the content
    # For JSON, the value appears after "field_name":
    # We need to find the exact position of this value
    escaped_val = json.dumps(field_value) if isinstance(field_value, str) else str(field_value)

    # Find all occurrences of this value in the content
    matches = []
    start = 0
    while True:
        idx = content.find(escaped_val, start)
        if idx == -1:
            # Try without quotes for numbers
            if not isinstance(field_value, str):
                idx = content.find(str(field_value), start)
            if idx == -1:
                break
        matches.append((idx, idx + len(escaped_val if idx == content.find(escaped_val, start) else str(field_value))))
        start = idx + 1

    if not matches:
        return []

    # Use the first match (could be smarter about disambiguation)
    val_start, val_end = matches[0]

    # Find overlapping token indices
    result = []
    for i, (ts, te) in enumerate(token_positions):
        if ts < val_end and te > val_start:
            result.append(i)

    return result


def analyze_fields(probe_data: dict, input_text: str | None = None) -> list[FieldRule]:
    """Analyze probe data to produce per-field rule assessments.

    Args:
        probe_data: Output from drift probe --output
        input_text: Original input text (for source pattern matching)

    Returns:
        List of FieldRule objects, one per JSON field.
    """
    tokens = probe_data["tokens"]
    content = "".join(t["token"] for t in tokens)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return []

    token_positions = _map_tokens_to_chars(tokens)
    high_entropy_indices = {ht["index"] for ht in probe_data.get("high_entropy_tokens", [])}
    high_entropy_map = {ht["index"]: ht for ht in probe_data.get("high_entropy_tokens", [])}

    flat_fields = _flatten_json(parsed)
    rules = []

    for path, val_str, raw_value in flat_fields:
        field_tokens = _find_field_tokens(content, tokens, token_positions, path, raw_value)

        if not field_tokens:
            rules.append(FieldRule(
                path=path,
                value=raw_value,
                rule_type="unknown",
                confidence=0.0,
            ))
            continue

        # Calculate field-level entropy
        field_entropies = [tokens[i]["entropy"] for i in field_tokens]
        max_entropy = max(field_entropies) if field_entropies else 0
        mean_entropy = sum(field_entropies) / len(field_entropies) if field_entropies else 0

        has_high_entropy = any(i in high_entropy_indices for i in field_tokens)
        entropy_detail = [
            high_entropy_map[i] for i in field_tokens if i in high_entropy_indices
        ]

        # Classify the field
        if max_entropy < 0.1:
            # Check if value exists in input text
            if input_text and str(raw_value) in input_text:
                rule_type = "extracted"
            elif isinstance(raw_value, (int, float)):
                rule_type = "computed"
            else:
                rule_type = "literal"
            confidence = 1.0 - mean_entropy
        elif max_entropy < 0.5:
            rule_type = "extracted"  # mostly deterministic, minor formatting drift
            confidence = 1.0 - mean_entropy
        else:
            rule_type = "llm_required"
            confidence = 1.0 - min(mean_entropy, 1.0)

        # Try to find source pattern in input
        source_pattern = None
        if input_text and isinstance(raw_value, str) and raw_value in input_text:
            # Find what's around the value in the input
            idx = input_text.index(raw_value)
            before = input_text[max(0, idx-30):idx].split("\n")[-1].strip()
            source_pattern = f"after '{before}'" if before else None

        rules.append(FieldRule(
            path=path,
            value=raw_value,
            rule_type=rule_type,
            confidence=round(confidence, 3),
            entropy_tokens=entropy_detail,
            source_pattern=source_pattern,
        ))

    return rules


def format_rules(rules: list[FieldRule]) -> str:
    """Format rule analysis as human-readable text."""
    lines = []
    lines.append("RULE EXTRACTION ANALYSIS")
    lines.append("=" * 70)

    # Summary
    by_type = {}
    for r in rules:
        by_type.setdefault(r.rule_type, []).append(r)

    total = len(rules)
    lines.append(f"  Total fields: {total}")
    for rtype in ["extracted", "literal", "computed", "llm_required", "unknown"]:
        count = len(by_type.get(rtype, []))
        if count:
            pct = count / total * 100
            lines.append(f"  {rtype:<16} {count:>4} ({pct:.0f}%)")

    extractable = len(by_type.get("extracted", [])) + len(by_type.get("literal", [])) + len(by_type.get("computed", []))
    lines.append(f"  Rule-extractable: {extractable}/{total} ({extractable/total*100:.0f}%)")
    lines.append("")

    # Detail by type
    for rtype, label in [
        ("extracted", "EXTRACTED (value found in input)"),
        ("literal", "LITERAL (fixed template value)"),
        ("computed", "COMPUTED (derived value)"),
        ("llm_required", "LLM REQUIRED (high entropy)"),
    ]:
        fields = by_type.get(rtype, [])
        if not fields:
            continue
        lines.append(f"  {label}:")
        lines.append("  " + "-" * 66)
        for r in fields:
            val_str = repr(r.value)[:40]
            conf = f"{r.confidence:.0%}"
            src = f"  [{r.source_pattern}]" if r.source_pattern else ""
            lines.append(f"    {r.path:<40} {val_str:<40} {conf}{src}")
            if r.entropy_tokens:
                for et in r.entropy_tokens:
                    alts = ", ".join(et["alternatives"][:3])
                    lines.append(f"      ^ token '{et['token']}' entropy={et['entropy']:.2f} alts: {alts}")
        lines.append("")

    return "\n".join(lines)
