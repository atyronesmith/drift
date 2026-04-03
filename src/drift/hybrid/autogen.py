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
    rule_type: str  # regex, literal, regex_all, mapping, conditional, classifier
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
    # For classifier rules
    vocabulary: list[str] | None = None
    weights: list[list[float]] | None = None
    bias: list[float] | None = None
    classes: list[str] | None = None
    confidence_threshold: float | None = None
    structured_features: list[dict] | None = None


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


def _compute_attention_hints(
    attention_data: dict,
    input_field_labels: dict[str, str] | None = None,
) -> dict[str, tuple[str, float]]:
    """Compute per-token attention density to discover input→output field mappings.

    Takes serialized attention probe data (from /v1/attention endpoint) and
    identifies which input field each output field attends to most, normalized
    by token count per input region to avoid bias toward long fields.

    Args:
        attention_data: JSON from probe_attention() or /v1/attention endpoint.
            Must have: input_tokens, output_tokens, n_input, content.
        input_field_labels: {name: regex_pattern} for input field labels.
            If None, auto-detects common patterns (Label:, Key=).

    Returns:
        {output_field: (input_field, density_ratio)} where density_ratio > 1.0
        means the field has higher per-token density than the description.
    """
    input_tokens = attention_data.get("input_tokens", [])
    output_tokens = attention_data.get("output_tokens", [])
    n_input = attention_data.get("n_input", len(input_tokens))

    if not input_tokens or not output_tokens:
        return {}

    # Find system prompt boundary: look for the user message start
    # In chat template, user content appears after system content
    # Heuristic: find where input data tokens begin (after template tokens)
    sys_prompt_end = _find_sys_prompt_end(input_tokens, n_input)

    # Map input token positions to field regions
    if input_field_labels is None:
        input_field_labels = _auto_detect_input_fields(input_tokens, sys_prompt_end)

    pos_to_field, field_token_counts = _map_tokens_to_fields(
        input_tokens, sys_prompt_end, input_field_labels
    )

    if not field_token_counts:
        return {}

    # Parse output field boundaries from generated JSON
    output_text = ""
    current_field = None
    field_start = None
    field_token_ranges = {}

    for i, probe in enumerate(output_tokens):
        token = probe["token"] if isinstance(probe, dict) else probe.token
        output_text += token

        for match in re.finditer(r'"(\w+(?:\.\w+)*)":\s*$', output_text):
            if current_field and field_start is not None:
                field_token_ranges[current_field] = (field_start, i)
            current_field = match.group(1)
            field_start = i + 1

    if current_field and field_start is not None:
        field_token_ranges[current_field] = (field_start, len(output_tokens))

    # Compute per-field attention distribution over input field regions
    hints = {}
    for out_field, (start, end) in field_token_ranges.items():
        field_weights = {}
        total_weight = 0

        for idx in range(start, min(end, len(output_tokens))):
            probe = output_tokens[idx]
            top_attended = probe["top_attended"] if isinstance(probe, dict) else probe.top_attended
            for pos, weight in top_attended:
                if isinstance(weight, float) and (weight != weight):  # NaN check
                    continue
                total_weight += weight
                if pos in pos_to_field:
                    f = pos_to_field[pos]
                    field_weights[f] = field_weights.get(f, 0) + weight

        if total_weight == 0:
            continue

        # Normalize to percentages
        field_pcts = {f: w / total_weight for f, w in field_weights.items()}

        # Compute per-token density
        densities = {}
        for f, pct in field_pcts.items():
            n_tokens = field_token_counts.get(f, 1)
            densities[f] = pct / n_tokens

        # Find description density as baseline
        desc_density = densities.get("description", 0)
        if desc_density == 0:
            # No description field — use the longest field as baseline
            longest = max(field_token_counts, key=field_token_counts.get)
            desc_density = densities.get(longest, 0.001)

        # Find non-description field with highest density
        non_desc = {f: d for f, d in densities.items() if f != "description"}
        if non_desc:
            top_field = max(non_desc, key=non_desc.get)
            top_density = non_desc[top_field]
            ratio = top_density / max(desc_density, 0.0001)
            if ratio > 1.0:
                hints[out_field] = (top_field, round(ratio, 2))

    return hints


def _find_sys_prompt_end(input_tokens: list[str], n_input: int) -> int:
    """Estimate where the system prompt ends in the input token sequence."""
    # Look for common chat template markers
    text = ""
    for i, tok in enumerate(input_tokens):
        text += tok
        # After system content, there's usually <|im_end|>\n<|im_start|>user
        if "<|im_start|>user" in text or "<|im_start|>\nuser" in text:
            return i
        if "user\n" in text[-20:] and i > 10:
            return i
    # Fallback: assume system prompt is roughly half the input
    return n_input // 2


def _auto_detect_input_fields(input_tokens: list[str], sys_prompt_end: int) -> dict[str, str]:
    """Auto-detect input field labels from token sequence.

    Only matches labels at the start of lines to avoid false positives
    from content within the description or other free-text fields.
    """
    text = "".join(input_tokens[sys_prompt_end:])
    labels = {}

    # Find "Label:" patterns at line beginnings only
    # Use [\w ] instead of [\w\s] to prevent matching across newlines
    for m in re.finditer(r'^[ \t]*(\w[\w ]*\w|\w)\s*:', text, re.MULTILINE):
        label = m.group(1).strip().lower().replace(" ", "_")
        escaped = re.escape(m.group(1).strip()) + r":"
        labels[label] = escaped

    return labels


def _map_tokens_to_fields(
    input_tokens: list[str],
    sys_prompt_end: int,
    field_labels: dict[str, str],
) -> tuple[dict[int, str], dict[str, int]]:
    """Map input token positions to field regions.

    Returns:
        pos_to_field: {token_position: field_name}
        field_token_counts: {field_name: count}
    """
    text = ""
    token_offsets = []
    for i in range(sys_prompt_end, len(input_tokens)):
        start = len(text)
        text += input_tokens[i]
        end = len(text)
        token_offsets.append((start, end, i))

    # Find field label positions
    field_positions = []
    for field_name, pattern in field_labels.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            field_positions.append((m.start(), field_name))

    field_positions.sort()

    pos_to_field = {}
    field_token_counts = {}
    for idx, (fp_start, field_name) in enumerate(field_positions):
        if idx + 1 < len(field_positions):
            fp_end = field_positions[idx + 1][0]
        else:
            fp_end = len(text)

        count = 0
        for char_start, char_end, token_pos in token_offsets:
            if char_start >= fp_start and char_start < fp_end:
                pos_to_field[token_pos] = field_name
                count += 1

        field_token_counts[field_name] = count

    return pos_to_field, field_token_counts


def _discover_tier2_rules(
    probes: list[tuple[dict, str]],
    tier1_paths: set[str],
    llm_paths: set[str],
    all_flat_fields: list[tuple[str, list]],
    attention_hints: dict[str, tuple[str, float]] | None = None,
) -> list[GeneratedRule]:
    """Discover mapping and conditional rules from multi-probe data.

    For each field marked llm_required by single-probe analysis, check if
    the output is conditionally deterministic when grouped by Tier 1 fields.

    When attention_hints are provided, uses them to:
    - Prioritize the suggested source field for mapping discovery
    - Lower accuracy threshold (60% vs 80%) when attention agrees
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

    # Build reverse map from attention hints: map input field names to
    # Tier 1 paths that extract from that field
    hint_source_paths = {}
    if attention_hints:
        # For each output field with a hint, find the Tier 1 path whose
        # extracted value comes from the hinted input field
        for out_field, (in_field, ratio) in attention_hints.items():
            # Map input field name to possible tier1 paths
            # e.g., input "category" -> tier1 path "ticket.stated_category"
            for t1_path in tier1_paths:
                if in_field in t1_path.lower() or t1_path.lower().endswith(in_field):
                    hint_source_paths[out_field] = (t1_path, ratio)
                    break

    # Only analyze fields that are in llm_paths (high entropy per single probe)
    # but might be conditionally deterministic
    tier2_rules = []

    for path in llm_paths:
        values = field_values.get(path, [])
        if not values or all(v is None for v in values):
            continue

        # Check if attention gives us a hint for this field
        field_name = path.split(".")[-1]
        attention_source = None
        attention_ratio = 0.0
        for hint_key, (source_path, ratio) in hint_source_paths.items():
            if hint_key == field_name or path.endswith(hint_key):
                attention_source = source_path
                attention_ratio = ratio
                break

        # Try each Tier 1 field as a potential source for mapping
        best_rule = _find_best_mapping(
            path, values, tier1_paths, field_values, parsed_outputs,
            preferred_source=attention_source,
            lower_threshold=attention_ratio > 1.0,
        )
        if best_rule:
            tier2_rules.append(best_rule)

    return tier2_rules


def _find_best_mapping(
    target_path: str,
    target_values: list,
    tier1_paths: set[str],
    field_values: dict[str, list],
    parsed_outputs: list[dict],
    preferred_source: str | None = None,
    lower_threshold: bool = False,
) -> GeneratedRule | None:
    """Find the best mapping or conditional rule for a target field.

    Tries single-source mappings first, then multi-source conditionals.
    When preferred_source is set (from attention hints), it's tried first
    and the accuracy threshold is lowered if lower_threshold is True.
    """
    n = len(target_values)
    accuracy_threshold = 0.6 if lower_threshold else 0.8

    # Try single-source mappings (simplest)
    best_source = None
    best_accuracy = 0.0
    best_mapping = None

    # If attention suggests a source, try it first
    source_order = list(tier1_paths)
    if preferred_source and preferred_source in tier1_paths:
        source_order = [preferred_source] + [p for p in source_order if p != preferred_source]

    for source_path in source_order:
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

    # Need sufficient accuracy for a mapping rule
    if best_accuracy >= accuracy_threshold and best_mapping:
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


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her",
    "my", "our", "your", "his", "its", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "and", "or", "but", "if", "when", "while", "as", "at", "by", "for",
    "in", "of", "on", "to", "with", "from", "not", "no", "so", "than",
    "too", "very", "just", "also", "how", "all", "each", "every", "any",
    "few", "more", "most", "other", "some", "such", "only", "own", "same",
    "then", "there", "here", "now", "about", "up", "out", "into", "over",
    "after", "before", "between", "under", "again", "further",
    "hello", "please", "thanks", "thank", "help", "issue", "issues",
    "hey", "dear", "regards", "best", "team", "hi", "support",
}


def _discover_classifier_rules(
    probes: list[tuple[dict, str]],
    tier1_paths: set[str],
    llm_paths: set[str],
    field_values: dict[str, list],
    vocab_size: int = 150,
    confidence_threshold: float = 0.95,
    min_accuracy: float = 0.70,
) -> list[GeneratedRule]:
    """Train classifiers for classification fields from probe data.

    For each llm_required field with < 20 unique values, trains a logistic
    regression classifier on bag-of-words + structured features. Fields
    that pass the accuracy threshold are emitted as classifier rules.

    Requires scikit-learn (optional dependency). Returns empty list if
    unavailable or too few probes.
    """
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except ImportError:
        return []

    if len(probes) < 20:
        return []

    # Parse all LLM outputs
    parsed_outputs = []
    input_texts = []
    for probe_data, input_text in probes:
        parsed = _parse_llm_output(probe_data)
        if parsed:
            parsed_outputs.append(parsed)
            input_texts.append(input_text)

    if len(parsed_outputs) < 20:
        return []

    n = len(parsed_outputs)

    # Find classification candidates: llm_required fields with < 20 unique values
    candidates = []
    for path in llm_paths:
        values = [_get_nested(p, path) for p in parsed_outputs]
        non_none = [v for v in values if v is not None]
        if not non_none:
            continue
        unique = set(str(v) for v in non_none)
        if 2 <= len(unique) <= 20 and len(non_none) >= 20:
            candidates.append((path, values))

    if not candidates:
        return []

    # Build vocabulary from all input texts
    from collections import Counter
    word_counts = Counter()
    for text in input_texts:
        words = set(re.findall(r"[a-z]+", text.lower()))
        word_counts.update(w for w in words if w not in STOPWORDS and len(w) >= 3)

    vocab = [w for w, _ in word_counts.most_common(vocab_size)]
    vocab_index = {w: i for i, w in enumerate(vocab)}

    # Identify structured feature sources from Tier 1 fields
    structured_features = []
    tier1_categories = {}
    for t1_path in sorted(tier1_paths):
        vals = field_values.get(t1_path, [])
        unique_vals = sorted(set(str(v) for v in vals if v is not None))
        if 2 <= len(unique_vals) <= 20:
            structured_features.append({
                "source": t1_path,
                "categories": unique_vals,
            })
            tier1_categories[t1_path] = unique_vals

    # Build feature matrix
    n_sf = sum(len(sf["categories"]) for sf in structured_features)
    n_features = len(vocab) + n_sf
    X = np.zeros((n, n_features), dtype=np.float32)

    for i, text in enumerate(input_texts):
        # Bag-of-words (binary)
        for w in re.findall(r"[a-z]+", text.lower()):
            if w in vocab_index:
                X[i, vocab_index[w]] = 1.0

        # Structured one-hot
        offset = len(vocab)
        for sf in structured_features:
            source_val = str(field_values.get(sf["source"], [None] * n)[i])
            for j, cat in enumerate(sf["categories"]):
                if source_val == cat:
                    X[i, offset + j] = 1.0
            offset += len(sf["categories"])

    # Train classifier for each candidate field
    rules = []
    for path, values in candidates:
        # Build label array, filtering None values
        mask = [v is not None for v in values]
        y = np.array([str(v) for v, m in zip(values, mask) if m])
        X_field = X[mask]

        classes = sorted(set(y))
        if len(classes) < 2:
            continue

        # Filter rare classes (< 3 samples)
        class_counts = Counter(y)
        rare = {c for c, cnt in class_counts.items() if cnt < 3}
        if rare:
            keep = np.array([yi not in rare for yi in y])
            X_field = X_field[keep]
            y = y[keep]
            classes = sorted(set(y))
            if len(classes) < 2 or len(y) < 20:
                continue

        # Train logistic regression with cross-validation
        lr = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
        n_splits = min(5, min(Counter(y).values()))
        if n_splits < 2:
            continue

        cv_scores = cross_val_score(lr, X_field, y, cv=n_splits, scoring="accuracy")
        cv_mean = cv_scores.mean()

        if cv_mean < min_accuracy:
            continue

        # Fit on full data and export weights
        lr.fit(X_field, y)

        rules.append(GeneratedRule(
            path=path,
            value=None,
            rule_type="classifier",
            pattern=None,
            confidence=round(cv_mean, 3),
            entropy=0.0,
            validated=True,
            vocabulary=vocab,
            weights=lr.coef_.tolist(),
            bias=lr.intercept_.tolist(),
            classes=list(lr.classes_),
            confidence_threshold=confidence_threshold,
            structured_features=structured_features if structured_features else None,
        ))

    return rules


def synthesize_rules_multi(
    probes: list[tuple[dict, str]],
    entropy_threshold: float = 0.3,
    attention_data: dict | None = None,
) -> SynthesisResult:
    """Auto-generate hybrid rules from multiple probes.

    Extends single-probe synthesis with Tier 2 discovery:
    1. Run single-probe synthesis on first probe to get Tier 1 rules
    2. Parse outputs from all probes
    3. For fields marked llm_required, check conditional determinism
    4. Emit mapping/conditional rules for Tier 2 fields

    When attention_data is provided (from /v1/attention or probe_attention()),
    per-token density analysis guides Tier 2 discovery by identifying which
    input field drives each output field, lowering accuracy thresholds for
    attention-confirmed mappings.

    Args:
        probes: List of (probe_data, input_text) pairs
        entropy_threshold: Fields below this are rule candidates
        attention_data: Optional attention probe JSON for mapping hints
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

    # Step 2.5: Compute attention hints if attention data is provided
    attention_hints = None
    if attention_data:
        attention_hints = _compute_attention_hints(attention_data)

    # Step 3: Discover Tier 2 rules
    tier2_rules = _discover_tier2_rules(
        probes, tier1_paths, llm_paths, all_flat_fields,
        attention_hints=attention_hints,
    )

    # Step 3.5: Discover Tier 3 classifier rules (needs 20+ probes)
    tier2_paths = {r.path for r in tier2_rules}
    remaining_llm_paths = llm_paths - tier2_paths

    # Collect per-field values for all probes
    all_field_values = {}
    for path, _ in all_flat_fields:
        all_field_values[path] = [
            _get_nested(p, path) for p in
            [_parse_llm_output(pd) for pd, _ in probes]
            if p is not None
        ]

    tier3_rules = _discover_classifier_rules(
        probes, tier1_paths, remaining_llm_paths, all_field_values,
    )

    # Step 4: Merge results — replace llm_required rules with Tier 2/3 where found
    tier23_paths = tier2_paths | {r.path for r in tier3_rules}
    final_rules = [r for r in base_result.rules if r.path not in tier23_paths]
    final_rules.extend(tier2_rules)
    final_rules.extend(tier3_rules)

    # Update llm_fields — remove paths that are now Tier 2 or 3
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
    stats["tier3_discovered"] = len(tier3_rules)
    stats["tier3_fields"] = [r.path for r in tier3_rules]
    if attention_hints:
        stats["attention_hints"] = len(attention_hints)
        stats["attention_fields"] = {k: v[0] for k, v in attention_hints.items()}

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
        elif rule.rule_type == "classifier":
            lines.append(f"    classifier_type: logistic_regression")
            lines.append(f"    classes: {json.dumps(rule.classes)}")
            lines.append(f"    vocabulary: {json.dumps(rule.vocabulary)}")
            lines.append(f"    weights: {json.dumps(rule.weights)}")
            lines.append(f"    bias: {json.dumps(rule.bias)}")
            if rule.structured_features:
                sf_str = pyyaml.dump(rule.structured_features, default_flow_style=True).strip()
                lines.append(f"    structured_features: {sf_str}")
            lines.append(f"    confidence_threshold: {rule.confidence_threshold}")
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
    if "tier3_discovered" in s and s["tier3_discovered"] > 0:
        lines.append(f"  Tier 3 classifiers:       {s['tier3_discovered']}")
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

    # Show Tier 3 rules
    tier3_rules = [r for r in result.rules if r.rule_type == "classifier"]
    if tier3_rules:
        lines.append("  TIER 3 RULES (classifier):")
        lines.append("  " + "-" * 66)
        for r in tier3_rules:
            lines.append(f"  [ OK ] {r.path:<35} classifier CV={r.confidence:.0%}")
            lines.append(f"          classes: {r.classes}")
            lines.append(f"          vocab: {len(r.vocabulary)} words, "
                         f"weights: {len(r.classes)}x{len(r.vocabulary)} matrix")
            lines.append(f"          confidence_threshold: {r.confidence_threshold}")
            if r.structured_features:
                sf_names = [sf["source"] for sf in r.structured_features]
                lines.append(f"          structured: {sf_names}")
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
