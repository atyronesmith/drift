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
    rule_type: str  # regex, literal, regex_all, mapping, conditional
    pattern: str | None  # generated regex
    confidence: float
    entropy: float
    source_location: dict | None = None  # where in input the value was found
    validated: bool = False  # did the pattern extract the expected value?
    # For mapping rules
    source: str | None = None  # source field path
    map: dict | None = None  # value mapping
    default: str | int | float | None = None
    # For conditional rules
    conditions: list[dict] | None = None  # [{when: {field: val}, value: val}]


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


def _parse_llm_output(probe_data: dict) -> dict | None:
    """Parse JSON output from probe token data."""
    tokens = probe_data["tokens"]
    content = "".join(t["token"] for t in tokens)
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _get_nested(obj: dict, path: str):
    """Get a value from a nested dict using dot notation."""
    for key in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return None
    return obj


def _discover_tier2_rules(
    probes: list[tuple[dict, str]],
    tier1_paths: set[str],
    llm_paths: set[str],
    all_flat_fields: list[tuple[str, list]],
) -> list[GeneratedRule]:
    """Discover mapping and conditional rules from multi-probe data.

    For each field marked llm_required by single-probe analysis, check if
    the output is conditionally deterministic when grouped by Tier 1 fields.
    """
    # Collect all parsed outputs and input texts
    parsed_outputs = []
    for probe_data, input_text in probes:
        parsed = _parse_llm_output(probe_data)
        if parsed:
            parsed_outputs.append(parsed)

    if len(parsed_outputs) < 3:
        return []

    # Collect all values for each field across probes
    field_values = {}
    for path, _ in all_flat_fields:
        vals = []
        for parsed in parsed_outputs:
            v = _get_nested(parsed, path)
            vals.append(v)
        field_values[path] = vals

    # Only analyze fields that are in llm_paths (high entropy per single probe)
    # but might be conditionally deterministic
    tier2_rules = []

    for path in llm_paths:
        values = field_values.get(path, [])
        if not values or all(v is None for v in values):
            continue

        # Try each Tier 1 field as a potential source for mapping
        best_rule = _find_best_mapping(path, values, tier1_paths, field_values, parsed_outputs)
        if best_rule:
            tier2_rules.append(best_rule)

    return tier2_rules


def _find_best_mapping(
    target_path: str,
    target_values: list,
    tier1_paths: set[str],
    field_values: dict[str, list],
    parsed_outputs: list[dict],
) -> GeneratedRule | None:
    """Find the best mapping or conditional rule for a target field.

    Tries single-source mappings first, then multi-source conditionals.
    """
    n = len(target_values)

    # Try single-source mappings (simplest)
    best_source = None
    best_accuracy = 0.0
    best_mapping = None

    for source_path in tier1_paths:
        source_values = field_values.get(source_path, [])
        if len(source_values) != n:
            continue

        # Group target values by source value
        groups = {}
        for sv, tv in zip(source_values, target_values):
            if sv is None or tv is None:
                continue
            sk = str(sv)
            if sk not in groups:
                groups[sk] = []
            groups[sk].append(tv)

        if not groups:
            continue

        # Check consistency within each group
        mapping = {}
        consistent = 0
        total = 0
        for source_val, target_vals in groups.items():
            total += len(target_vals)
            unique = set(str(v) for v in target_vals)
            if len(unique) == 1:
                consistent += len(target_vals)
                mapping[source_val] = target_vals[0]

        if total == 0:
            continue

        accuracy = consistent / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_source = source_path
            best_mapping = mapping

    # Need at least 80% accuracy for a mapping rule
    if best_accuracy >= 0.8 and best_mapping:
        # Determine default: most common value not in mapping
        all_target = [str(v) for v in target_values if v is not None]
        mapped_source_vals = set(best_mapping.keys())
        unmapped_targets = []
        source_values = field_values.get(best_source, [])
        for sv, tv in zip(source_values, target_values):
            if sv is not None and str(sv) not in mapped_source_vals and tv is not None:
                unmapped_targets.append(tv)

        default = None
        if unmapped_targets:
            from collections import Counter
            default = Counter(str(v) for v in unmapped_targets).most_common(1)[0][0]

        return GeneratedRule(
            path=target_path,
            value=None,
            rule_type="mapping",
            pattern=None,
            confidence=round(best_accuracy, 3),
            entropy=0.0,
            validated=True,
            source=best_source,
            map=best_mapping,
            default=default,
        )

    # Try multi-source conditional (two sources combined)
    best_cond = _find_conditional(target_path, target_values, tier1_paths, field_values)
    if best_cond:
        return best_cond

    return None


def _find_conditional(
    target_path: str,
    target_values: list,
    tier1_paths: set[str],
    field_values: dict[str, list],
) -> GeneratedRule | None:
    """Find a conditional rule using combinations of source fields."""
    from itertools import combinations

    n = len(target_values)
    source_paths = [p for p in tier1_paths if len(field_values.get(p, [])) == n]

    # Try pairs of source fields
    for s1, s2 in combinations(source_paths, 2):
        vals1 = field_values[s1]
        vals2 = field_values[s2]

        groups = {}
        for v1, v2, tv in zip(vals1, vals2, target_values):
            if v1 is None or v2 is None or tv is None:
                continue
            key = (str(v1), str(v2))
            if key not in groups:
                groups[key] = []
            groups[key].append(tv)

        if not groups:
            continue

        consistent = 0
        total = 0
        conditions = []
        for (sv1, sv2), tvs in groups.items():
            total += len(tvs)
            unique = set(str(v) for v in tvs)
            if len(unique) == 1:
                consistent += len(tvs)
                conditions.append({
                    "when": {s1: sv1, s2: sv2},
                    "value": tvs[0],
                })

        if total == 0:
            continue

        accuracy = consistent / total
        if accuracy >= 0.8 and conditions:
            # Default: most common unmapped value
            default = None
            all_target = [str(v) for v in target_values if v is not None]
            if all_target:
                from collections import Counter
                default = Counter(all_target).most_common(1)[0][0]

            return GeneratedRule(
                path=target_path,
                value=None,
                rule_type="conditional",
                pattern=None,
                confidence=round(accuracy, 3),
                entropy=0.0,
                validated=True,
                conditions=conditions,
                default=default,
            )

    return None


def synthesize_rules_multi(
    probes: list[tuple[dict, str]],
    entropy_threshold: float = 0.3,
) -> SynthesisResult:
    """Auto-generate hybrid rules from multiple probes.

    Extends single-probe synthesis with Tier 2 discovery:
    1. Run single-probe synthesis on first probe to get Tier 1 rules
    2. Parse outputs from all probes
    3. For fields marked llm_required, check conditional determinism
    4. Emit mapping/conditional rules for Tier 2 fields

    Args:
        probes: List of (probe_data, input_text) pairs
        entropy_threshold: Fields below this are rule candidates
    """
    if not probes:
        return SynthesisResult(rules=[], llm_fields=[], stats={
            "total_fields": 0, "rule_candidates": 0, "rules_generated": 0,
            "rules_validated": 0, "llm_required": 0, "probes_used": 0,
        })

    # Step 1: Single-probe synthesis for Tier 1 rules (use first probe)
    base_result = synthesize_rules(probes[0][0], probes[0][1], entropy_threshold)

    tier1_paths = {r.path for r in base_result.rules if r.rule_type not in ("llm_required",)}
    llm_paths = {r.path for r in base_result.rules if r.rule_type == "llm_required"}

    if len(probes) < 3 or not llm_paths:
        # Not enough probes for Tier 2 discovery, return single-probe result
        base_result.stats["probes_used"] = len(probes)
        return base_result

    # Step 2: Collect all flat fields from first probe for field enumeration
    parsed_first = _parse_llm_output(probes[0][0])
    all_flat_fields = _flatten_json(parsed_first) if parsed_first else []

    # Step 3: Discover Tier 2 rules
    tier2_rules = _discover_tier2_rules(probes, tier1_paths, llm_paths, all_flat_fields)

    # Step 4: Merge results — replace llm_required rules with Tier 2 where found
    tier2_paths = {r.path for r in tier2_rules}
    final_rules = [r for r in base_result.rules if r.path not in tier2_paths]
    final_rules.extend(tier2_rules)

    # Update llm_fields — remove paths that are now Tier 2
    remaining_llm = [r.path for r in final_rules if r.rule_type == "llm_required"]
    llm_top_level = []
    seen = set()
    for f in remaining_llm:
        top = f.split(".")[0].split("[")[0]
        if top not in seen:
            llm_top_level.append(top)
            seen.add(top)

    # Also keep any top-level LLM fields from base that aren't fully covered
    for f in base_result.llm_fields:
        if f not in seen:
            # Check if any sub-fields under this top-level are still llm_required
            has_llm_subfield = any(
                r.path.startswith(f + ".") or r.path == f
                for r in final_rules if r.rule_type == "llm_required"
            )
            if has_llm_subfield and f not in seen:
                llm_top_level.append(f)
                seen.add(f)

    stats = dict(base_result.stats)
    stats["probes_used"] = len(probes)
    stats["tier2_discovered"] = len(tier2_rules)
    stats["tier2_types"] = {r.rule_type: 0 for r in tier2_rules}
    for r in tier2_rules:
        stats["tier2_types"][r.rule_type] += 1

    return SynthesisResult(rules=final_rules, llm_fields=llm_top_level, stats=stats)


def render_hybrid_yaml(result: SynthesisResult) -> str:
    """Render a SynthesisResult as a hybrid.yaml string."""
    import yaml as pyyaml

    lines = ["# Auto-generated by: drift synthesize", "# Edit as needed.", ""]
    lines.append("rules:")

    for rule in result.rules:
        if rule.rule_type in ("llm_required",):
            continue

        lines.append(f"  - path: {rule.path}")
        lines.append(f"    type: {rule.rule_type}")

        if rule.rule_type == "mapping":
            lines.append(f"    source: {rule.source}")
            map_str = pyyaml.dump(rule.map, default_flow_style=True).strip()
            lines.append(f"    map: {map_str}")
            if rule.default is not None:
                lines.append(f"    default: {_yaml_scalar(rule.default)}")
        elif rule.rule_type == "conditional":
            lines.append("    conditions:")
            for cond in (rule.conditions or []):
                when = cond["when"]
                when_str = pyyaml.dump(when, default_flow_style=True).strip()
                lines.append(f"      - when: {when_str}")
                lines.append(f"        value: {_yaml_scalar(cond['value'])}")
            if rule.default is not None:
                lines.append(f"    default: {_yaml_scalar(rule.default)}")
        elif rule.pattern:
            yaml_str = pyyaml.dump(rule.pattern, default_style='"').strip()
            lines.append(f"    pattern: {yaml_str}")
        elif rule.rule_type == "literal":
            lines.append(f"    value: {_yaml_scalar(rule.value)}")

        status = "validated" if rule.validated else "UNVALIDATED"
        lines.append(f"    # confidence={rule.confidence:.0%} [{status}]")
        lines.append("")

    lines.append("llm_fields:")
    for f in result.llm_fields:
        lines.append(f"  - {f}")

    lines.append("")
    lines.append("output_mode: merge")

    return "\n".join(lines)


def _yaml_scalar(val) -> str:
    """Format a value for inline YAML."""
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)


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
    if "probes_used" in s:
        lines.append(f"  Probes used:              {s['probes_used']}")
    if "tier2_discovered" in s:
        lines.append(f"  Tier 2 discovered:        {s['tier2_discovered']}")
    if s['total_fields'] > 0:
        total_rules = len([r for r in result.rules if r.rule_type not in ("llm_required",)])
        pct = total_rules / s['total_fields'] * 100
        lines.append(f"  Automation rate:          {pct:.0f}%")
    lines.append("")

    # Show Tier 1 rules
    tier1_rules = [r for r in result.rules if r.rule_type in ("regex", "regex_all", "literal")]
    if tier1_rules:
        lines.append("  TIER 1 RULES (regex/literal):")
        lines.append("  " + "-" * 66)
        for r in tier1_rules:
            status = "OK" if r.validated else "FAIL"
            val_str = repr(r.value)[:35]
            if r.pattern:
                lines.append(f"  [{status:4}] {r.path:<35} {r.rule_type:<8} entropy={r.entropy:.3f}")
                lines.append(f"          pattern: {r.pattern}")
                lines.append(f"          value:   {val_str}")
            else:
                lines.append(f"  [{status:4}] {r.path:<35} {r.rule_type:<8} = {val_str}")
            lines.append("")

    # Show Tier 2 rules
    tier2_rules = [r for r in result.rules if r.rule_type in ("mapping", "conditional")]
    if tier2_rules:
        lines.append("  TIER 2 RULES (mapping/conditional):")
        lines.append("  " + "-" * 66)
        for r in tier2_rules:
            status = "OK" if r.validated else "FAIL"
            if r.rule_type == "mapping":
                lines.append(f"  [{status:4}] {r.path:<35} mapping    confidence={r.confidence:.0%}")
                lines.append(f"          source: {r.source}")
                lines.append(f"          map: {r.map}")
                if r.default is not None:
                    lines.append(f"          default: {r.default}")
            elif r.rule_type == "conditional":
                lines.append(f"  [{status:4}] {r.path:<35} conditional confidence={r.confidence:.0%}")
                for cond in (r.conditions or []):
                    lines.append(f"          when {cond['when']} → {cond['value']}")
                if r.default is not None:
                    lines.append(f"          default: {r.default}")
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
