"""Automated rule generation from probe data.

Takes logprob probe output + original input text and automatically generates
extraction rules by:
1. Identifying low-entropy (deterministic) fields in the LLM output
2. Tracing each deterministic value back to its source in the input
3. Analyzing surrounding context to generate regex patterns
4. Validating generated patterns against the input
5. Producing a complete hybrid spec (rules + llm_fields)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class GeneratedRule:
    path: str
    value: str | int | float | list | None
    rule_type: str  # regex, literal, regex_all
    pattern: str | None  # generated regex
    confidence: float
    entropy: float
    source_location: dict | None = None  # where in input the value was found
    validated: bool = False  # did the pattern extract the expected value?


@dataclass
class SynthesisResult:
    rules: list[GeneratedRule]
    llm_fields: list[str]
    stats: dict


def _flatten_json(obj, prefix=""):
    """Flatten JSON to (path, raw_value) tuples."""
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
        results.append((prefix, obj))
    return results


def _map_tokens_to_chars(tokens: list[dict]) -> list[tuple[int, int]]:
    """Map each token index to (start_char, end_char)."""
    positions = []
    pos = 0
    for t in tokens:
        positions.append((pos, pos + len(t["token"])))
        pos += len(t["token"])
    return positions


def _field_entropy(
    content: str,
    tokens: list[dict],
    token_positions: list[tuple[int, int]],
    raw_value,
) -> float | None:
    """Compute mean entropy for tokens that encode a field value."""
    if raw_value is None:
        return None

    val_str = json.dumps(raw_value) if isinstance(raw_value, str) else str(raw_value)

    idx = content.find(val_str)
    if idx == -1 and isinstance(raw_value, str):
        idx = content.find(raw_value)
        if idx != -1:
            val_str = raw_value
    if idx == -1:
        return None

    val_end = idx + len(val_str)

    entropies = []
    for i, (ts, te) in enumerate(token_positions):
        if ts < val_end and te > idx:
            entropies.append(tokens[i]["entropy"])

    return sum(entropies) / len(entropies) if entropies else None


def _find_value_context(value: str, input_text: str) -> dict | None:
    """Find a value in the input text and return its surrounding context.

    Returns dict with: position, before_text, after_text, line, label
    """
    idx = input_text.find(value)
    if idx == -1:
        return None

    # Get the line containing this value
    line_start = input_text.rfind("\n", 0, idx) + 1
    line_end = input_text.find("\n", idx)
    if line_end == -1:
        line_end = len(input_text)
    full_line = input_text[line_start:line_end]

    # Get text before the value on this line
    before_on_line = input_text[line_start:idx].strip()

    # Get text after the value on this line
    after_on_line = input_text[idx + len(value):line_end].strip()

    # Try to identify a label (e.g., "Reporter:", "Product:", "key=")
    label = _extract_label(before_on_line)

    return {
        "position": idx,
        "line": full_line.strip(),
        "before": before_on_line,
        "after": after_on_line,
        "label": label,
    }


def _extract_label(before_text: str) -> str | None:
    """Extract the label/key from text preceding a value.

    Recognizes patterns like:
      "Reporter:"  →  "Reporter"
      "team="      →  "team"
      "Product:   " → "Product"
    """
    # Pattern: "Word:" or "Word:   " (colon-delimited label)
    m = re.search(r"(\w[\w\s]*\w|\w)\s*:\s*$", before_text)
    if m:
        return m.group(1).strip()

    # Pattern: "key=" (equals-delimited)
    m = re.search(r"(\w+)\s*=\s*$", before_text)
    if m:
        return m.group(1)

    # Pattern: "- " (list item, look further back for context)
    if before_text.endswith("-") or before_text.endswith("- "):
        return None

    return None


def _generate_regex(value: str, context: dict, input_text: str) -> tuple[str | None, str]:
    """Generate a regex pattern to extract a value based on its context.

    Returns (pattern, rule_type) where rule_type is 'regex' or 'literal'.
    """
    label = context.get("label")
    before = context.get("before", "")
    after = context.get("after", "")

    if label:
        escaped_label = re.escape(label)

        # Determine terminator from what follows the value
        terminator = _detect_terminator(after, value, input_text, context["position"])

        # Build pattern: label + separator + capture group + terminator
        if ":" in before:
            # "Label:    value"
            pattern = f"{escaped_label}:\\s*({_value_pattern(value)}){terminator}"
        elif "=" in before:
            # "key=value"
            pattern = f"{escaped_label}=({_value_pattern(value)})"
        else:
            pattern = f"{escaped_label}\\s+({_value_pattern(value)}){terminator}"

        return pattern, "regex"

    # No label found — try anchoring on surrounding text
    if before:
        # Use the last few words before the value as anchor
        anchor = _make_anchor(before)
        if anchor:
            terminator = _detect_terminator(after, value, input_text, context["position"])
            pattern = f"{anchor}\\s*({_value_pattern(value)}){terminator}"
            return pattern, "regex"

    # Can't generate a reliable pattern
    return None, "literal"


def _value_pattern(value: str) -> str:
    """Generate an appropriate capture pattern for a value type."""
    # Numeric
    if re.match(r"^\d+$", value):
        return r"\d+"
    # Decimal
    if re.match(r"^\d+\.\d+$", value):
        return r"\d+\.\d+"
    # ID-like (letters, digits, hyphens)
    if re.match(r"^[A-Z]+-[\d-]+$", value):
        return r"[A-Z]+-[\d-]+"
    if re.match(r"^[A-Z]+-\d+$", value):
        return r"[A-Z]+-\d+"
    # Email
    if re.match(r"^[\w.+-]+@[\w.-]+$", value):
        return r"[\w.+-]+@[\w.-]+"
    # Version (v1.2.3)
    if re.match(r"^v?\d+\.\d+", value):
        return r"v?[\d.]+"
    # Short token (single word, no spaces)
    if " " not in value and len(value) < 30:
        return r"\S+"
    # General: non-greedy match to terminator
    return r".+?"


def _detect_terminator(after: str, value: str, input_text: str, pos: int) -> str:
    """Determine what terminates a value in the input text."""
    if not after:
        return r"\s*$"

    first_char = after[0] if after else ""

    if first_char == "(":
        return r"\s*\("
    if first_char == ")":
        return r"\)"
    if first_char == ",":
        return r"\s*,"
    if first_char == "|":
        return r"\s*\|"
    if first_char == '"':
        return r'"'

    # Default: match to end of line
    return r"\s*$"


def _make_anchor(before: str) -> str | None:
    """Create a regex anchor from preceding text."""
    # Take the last meaningful chunk
    words = before.split()
    if not words:
        return None
    # Use last 1-3 words as anchor
    anchor_words = words[-min(3, len(words)):]
    anchor = " ".join(anchor_words)
    return re.escape(anchor)


def _validate_pattern(pattern: str, input_text: str, expected: str) -> bool:
    """Test whether a generated pattern correctly extracts the expected value."""
    try:
        m = re.search(pattern, input_text, re.MULTILINE)
        if m and m.group(1).strip() == expected.strip():
            return True
        # Try case-insensitive
        m = re.search(pattern, input_text, re.MULTILINE | re.IGNORECASE)
        if m and m.group(1).strip() == expected.strip():
            return True
        return False
    except re.error:
        return False


def synthesize_rules(probe_data: dict, input_text: str, entropy_threshold: float = 0.3) -> SynthesisResult:
    """Auto-generate hybrid rules from probe data and input text.

    This is the main entry point. It:
    1. Parses the LLM output from probe data
    2. Computes per-field entropy from token logprobs
    3. For low-entropy fields, traces values to input and generates regex
    4. Validates each generated pattern
    5. Returns a SynthesisResult with rules and remaining LLM fields

    Args:
        probe_data: Output from `drift probe -o`
        input_text: The original input text fed to the LLM
        entropy_threshold: Fields below this are rule candidates (default 0.3)
    """
    tokens = probe_data["tokens"]
    content = "".join(t["token"] for t in tokens)

    # Strip code fences for JSON parsing
    parse_content = content.strip()
    if parse_content.startswith("```"):
        lines = parse_content.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        parse_content = "\n".join(lines)

    try:
        parsed = json.loads(parse_content)
    except json.JSONDecodeError:
        return SynthesisResult(
            rules=[], llm_fields=[],
            stats={"total_fields": 0, "rule_candidates": 0, "rules_generated": 0,
                   "rules_validated": 0, "llm_required": 0,
                   "error": "LLM output is not JSON — synthesize only works with JSON outputs"},
        )

    token_positions = _map_tokens_to_chars(tokens)
    flat_fields = _flatten_json(parsed)

    rules = []
    llm_fields = []
    stats = {"total_fields": 0, "rule_candidates": 0, "rules_generated": 0, "rules_validated": 0, "llm_required": 0}

    for path, raw_value in flat_fields:
        stats["total_fields"] += 1

        # Skip None/empty values
        if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
            llm_fields.append(path)
            stats["llm_required"] += 1
            continue

        # Compute field entropy
        entropy = _field_entropy(content, tokens, token_positions, raw_value)
        if entropy is None:
            llm_fields.append(path)
            stats["llm_required"] += 1
            continue

        # High entropy → LLM required
        if entropy > entropy_threshold:
            llm_fields.append(path)
            stats["llm_required"] += 1
            rules.append(GeneratedRule(
                path=path, value=raw_value, rule_type="llm_required",
                pattern=None, confidence=round(1.0 - min(entropy, 1.0), 3),
                entropy=round(entropy, 4),
            ))
            continue

        stats["rule_candidates"] += 1
        value_str = str(raw_value)

        # Try to find value in input and generate regex
        context = _find_value_context(value_str, input_text)

        if context:
            pattern, rule_type = _generate_regex(value_str, context, input_text)

            if pattern:
                validated = _validate_pattern(pattern, input_text, value_str)
                stats["rules_generated"] += 1
                if validated:
                    stats["rules_validated"] += 1

                rules.append(GeneratedRule(
                    path=path, value=raw_value, rule_type=rule_type,
                    pattern=pattern, confidence=round(1.0 - entropy, 3),
                    entropy=round(entropy, 4),
                    source_location=context, validated=validated,
                ))
                continue

        # Value not found in input or no pattern generated → literal if deterministic
        if entropy < 0.05:
            rules.append(GeneratedRule(
                path=path, value=raw_value, rule_type="literal",
                pattern=None, confidence=round(1.0 - entropy, 3),
                entropy=round(entropy, 4), validated=True,
            ))
            stats["rules_generated"] += 1
            stats["rules_validated"] += 1
        else:
            llm_fields.append(path)
            stats["llm_required"] += 1
            rules.append(GeneratedRule(
                path=path, value=raw_value, rule_type="llm_required",
                pattern=None, confidence=round(1.0 - entropy, 3),
                entropy=round(entropy, 4),
            ))

    # Deduplicate llm_fields to top-level keys
    llm_top_level = []
    seen = set()
    for f in llm_fields:
        top = f.split(".")[0].split("[")[0]
        if top not in seen:
            llm_top_level.append(top)
            seen.add(top)

    return SynthesisResult(rules=rules, llm_fields=llm_top_level, stats=stats)


def render_hybrid_yaml(result: SynthesisResult) -> str:
    """Render a SynthesisResult as a hybrid.yaml string."""
    lines = ["# Auto-generated by: drift synthesize", "# Edit as needed.", ""]
    lines.append("rules:")

    for rule in result.rules:
        if rule.rule_type in ("llm_required",):
            continue

        lines.append(f"  - path: {rule.path}")
        lines.append(f"    type: {rule.rule_type}")
        if rule.pattern:
            # Use YAML's own serialization for correct escaping
            import yaml as pyyaml
            yaml_str = pyyaml.dump(rule.pattern, default_style='"').strip()
            lines.append(f"    pattern: {yaml_str}")
        elif rule.rule_type == "literal":
            val = rule.value
            if isinstance(val, str):
                lines.append(f'    value: "{val}"')
            else:
                lines.append(f"    value: {val}")

        status = "validated" if rule.validated else "UNVALIDATED"
        lines.append(f"    # entropy={rule.entropy:.3f} confidence={rule.confidence:.0%} [{status}]")
        lines.append("")

    lines.append("llm_fields:")
    for f in result.llm_fields:
        lines.append(f"  - {f}")

    lines.append("")
    lines.append("output_mode: merge")

    return "\n".join(lines)


def format_synthesis_report(result: SynthesisResult) -> str:
    """Format a human-readable report of the synthesis."""
    lines = []
    lines.append("AUTOMATED RULE SYNTHESIS")
    lines.append("=" * 70)

    s = result.stats
    lines.append(f"  Total fields analyzed:    {s['total_fields']}")
    lines.append(f"  Rule candidates:          {s['rule_candidates']} (entropy < threshold)")
    lines.append(f"  Rules generated:          {s['rules_generated']}")
    lines.append(f"  Rules validated:           {s['rules_validated']}")
    lines.append(f"  LLM required:             {s['llm_required']}")
    if s['total_fields'] > 0:
        pct = s['rules_validated'] / s['total_fields'] * 100
        lines.append(f"  Automation rate:          {pct:.0f}%")
    lines.append("")

    # Show generated rules
    gen_rules = [r for r in result.rules if r.rule_type not in ("llm_required",)]
    if gen_rules:
        lines.append("  GENERATED RULES:")
        lines.append("  " + "-" * 66)
        for r in gen_rules:
            status = "OK" if r.validated else "FAIL"
            val_str = repr(r.value)[:35]
            if r.pattern:
                lines.append(f"  [{status:4}] {r.path:<35} {r.rule_type:<8} entropy={r.entropy:.3f}")
                lines.append(f"          pattern: {r.pattern}")
                lines.append(f"          value:   {val_str}")
            else:
                lines.append(f"  [{status:4}] {r.path:<35} {r.rule_type:<8} = {val_str}")
            lines.append("")

    # Show LLM-required fields
    llm_rules = [r for r in result.rules if r.rule_type == "llm_required"]
    if llm_rules:
        lines.append("  LLM-REQUIRED FIELDS:")
        lines.append("  " + "-" * 66)
        for r in llm_rules:
            val_str = repr(r.value)[:50]
            lines.append(f"         {r.path:<35} entropy={r.entropy:.3f}  {val_str}")
        lines.append("")

    if result.llm_fields:
        lines.append(f"  LLM top-level fields: {', '.join(result.llm_fields)}")

    return "\n".join(lines)
