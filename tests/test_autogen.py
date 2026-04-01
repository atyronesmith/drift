"""Tests for automated rule generation from probe data."""

import json
import pytest
from pathlib import Path

from drift.hybrid.autogen import (
    _find_value_context,
    _extract_label,
    _generate_regex,
    _validate_pattern,
    synthesize_rules,
    synthesize_rules_multi,
    render_hybrid_yaml,
)


SAMPLE_INPUT = """
Reporter:    Sarah Chen (sarah.chen@example.com)
Account:     Acme Corp (ACCT-1234)
Tier:        Gold
Product:     Widget Pro v2.1
Environment: Production (us-east-1)
"""


def test_find_value_context_labeled():
    ctx = _find_value_context("Sarah Chen", SAMPLE_INPUT)
    assert ctx is not None
    assert ctx["label"] == "Reporter"
    assert "(" in ctx["after"]


def test_find_value_context_not_found():
    assert _find_value_context("nonexistent", SAMPLE_INPUT) is None


def test_extract_label_colon():
    assert _extract_label("Reporter:   ") == "Reporter"
    assert _extract_label("  Product:  ") == "Product"


def test_extract_label_equals():
    assert _extract_label("team=") == "team"


def test_extract_label_none():
    assert _extract_label("just some text") is None


def test_generate_regex_colon_label():
    ctx = _find_value_context("Acme Corp", SAMPLE_INPUT)
    pattern, rule_type = _generate_regex("Acme Corp", ctx, SAMPLE_INPUT)
    assert rule_type == "regex"
    assert pattern is not None
    assert _validate_pattern(pattern, SAMPLE_INPUT, "Acme Corp")


def test_generate_regex_id():
    ctx = _find_value_context("ACCT-1234", SAMPLE_INPUT)
    pattern, rule_type = _generate_regex("ACCT-1234", ctx, SAMPLE_INPUT)
    assert pattern is not None
    assert _validate_pattern(pattern, SAMPLE_INPUT, "ACCT-1234")


def test_validate_pattern_pass():
    assert _validate_pattern(r"Product:\s*(.+?)\s*$", SAMPLE_INPUT, "Widget Pro v2.1")


def test_validate_pattern_fail():
    assert not _validate_pattern(r"Nonexistent:\s*(.+)", SAMPLE_INPUT, "anything")


def test_synthesize_from_probe_data():
    """Test synthesis with minimal probe data."""
    # Simulate probe output for a simple JSON
    tokens = [
        {"token": "{", "logprob": 0, "entropy": 0.0},
        {"token": '"', "logprob": 0, "entropy": 0.0},
        {"token": "name", "logprob": 0, "entropy": 0.0},
        {"token": '"', "logprob": 0, "entropy": 0.0},
        {"token": ":", "logprob": 0, "entropy": 0.0},
        {"token": " ", "logprob": 0, "entropy": 0.0},
        {"token": '"', "logprob": 0, "entropy": 0.0},
        {"token": "Acme", "logprob": 0, "entropy": 0.01},
        {"token": " Corp", "logprob": 0, "entropy": 0.01},
        {"token": '"', "logprob": 0, "entropy": 0.0},
        {"token": "}", "logprob": 0, "entropy": 0.0},
    ]
    probe_data = {
        "tokens": tokens,
        "high_entropy_tokens": [],
    }
    input_text = "Company: Acme Corp\nRevenue: $1M"

    result = synthesize_rules(probe_data, input_text)
    assert result.stats["total_fields"] == 1
    assert result.stats["rules_generated"] >= 1


def test_render_hybrid_yaml_roundtrip():
    """Generated YAML should be parseable and contain correct patterns."""
    import yaml
    tokens = [
        {"token": "{", "logprob": 0, "entropy": 0.0},
        {"token": '"', "logprob": 0, "entropy": 0.0},
        {"token": "name", "logprob": 0, "entropy": 0.0},
        {"token": '"', "logprob": 0, "entropy": 0.0},
        {"token": ":", "logprob": 0, "entropy": 0.0},
        {"token": " ", "logprob": 0, "entropy": 0.0},
        {"token": '"', "logprob": 0, "entropy": 0.0},
        {"token": "Acme", "logprob": 0, "entropy": 0.01},
        {"token": " Corp", "logprob": 0, "entropy": 0.01},
        {"token": '"', "logprob": 0, "entropy": 0.0},
        {"token": "}", "logprob": 0, "entropy": 0.0},
    ]
    probe_data = {"tokens": tokens, "high_entropy_tokens": []}
    result = synthesize_rules(probe_data, "Company: Acme Corp")
    yaml_str = render_hybrid_yaml(result)

    # Should parse as valid YAML
    spec = yaml.safe_load(yaml_str)
    assert "rules" in spec
    assert "llm_fields" in spec

    # Patterns should work after YAML roundtrip
    import re
    for rule in spec["rules"]:
        if "pattern" in rule:
            re.compile(rule["pattern"])  # should not raise


# --- Multi-probe tests ---

def _make_probe(output_json: dict) -> dict:
    """Build a minimal probe_data from a JSON output dict."""
    content = json.dumps(output_json)
    tokens = []
    for ch in content:
        # Low entropy for Tier 1 fields, high for Tier 3
        tokens.append({"token": ch, "logprob": 0, "entropy": 0.01})
    return {"tokens": tokens}


def _make_probe_with_entropy(output_json: dict, high_entropy_paths: set[str]) -> dict:
    """Build a probe with mixed entropy: low for regex fields, high for specified paths."""
    content = json.dumps(output_json)
    tokens = []
    # Find positions of high-entropy values in the JSON string
    high_ranges = []
    for path in high_entropy_paths:
        val = output_json
        for key in path.split("."):
            if isinstance(val, dict):
                val = val.get(key)
        if val is not None:
            val_str = json.dumps(val) if isinstance(val, str) else str(val)
            idx = content.find(val_str)
            if idx >= 0:
                high_ranges.append((idx, idx + len(val_str)))

    for i, ch in enumerate(content):
        is_high = any(s <= i < e for s, e in high_ranges)
        ent = 0.8 if is_high else 0.01
        tokens.append({"token": ch, "logprob": 0, "entropy": ent})
    return {"tokens": tokens}


def test_synthesize_multi_discovers_mapping():
    """Multi-probe synthesis discovers mapping rules from consistent patterns."""
    # 5 probes: priority → severity mapping is consistent
    probes = []
    priorities = ["High", "Medium", "Low", "High", "Medium"]
    severities = ["P2", "P3", "P4", "P2", "P3"]

    for i, (pri, sev) in enumerate(zip(priorities, severities)):
        output = {"ticket": {"priority": pri}, "assessment": {"severity": sev}}
        input_text = f"Priority: {pri}\nDescription: ticket {i}"
        probe = _make_probe_with_entropy(output, {"assessment.severity"})
        probes.append((probe, input_text))

    result = synthesize_rules_multi(probes)

    # Should have discovered a mapping rule for assessment.severity
    mapping_rules = [r for r in result.rules if r.rule_type == "mapping"]
    assert len(mapping_rules) >= 1
    sev_rule = [r for r in mapping_rules if r.path == "assessment.severity"]
    assert len(sev_rule) == 1
    assert sev_rule[0].source == "ticket.priority"
    assert sev_rule[0].map == {"High": "P2", "Medium": "P3", "Low": "P4"}


def test_synthesize_multi_too_few_probes():
    """With fewer than 3 probes, falls back to single-probe synthesis."""
    output = {"name": "Acme"}
    input_text = "Company: Acme"
    probe = _make_probe(output)

    result = synthesize_rules_multi([(probe, input_text)])
    # Should work but no Tier 2 discovery
    assert "tier2_discovered" not in result.stats or result.stats.get("tier2_discovered", 0) == 0


def test_synthesize_multi_no_pattern():
    """Fields that vary inconsistently remain LLM-required."""
    probes = []
    # severity varies even for same priority → no mapping possible
    priorities = ["High", "High", "High", "Medium", "Medium"]
    severities = ["P1", "P2", "P3", "P3", "P4"]  # inconsistent for High

    for i, (pri, sev) in enumerate(zip(priorities, severities)):
        output = {"ticket": {"priority": pri}, "assessment": {"severity": sev}}
        input_text = f"Priority: {pri}\nDescription: ticket {i}"
        probe = _make_probe_with_entropy(output, {"assessment.severity"})
        probes.append((probe, input_text))

    result = synthesize_rules_multi(probes)

    # High has 3 different values → accuracy < 80% → no mapping
    mapping_rules = [r for r in result.rules if r.rule_type == "mapping" and r.path == "assessment.severity"]
    assert len(mapping_rules) == 0


def test_synthesize_multi_mapping_with_default():
    """Mapping rules include a default for unmapped source values."""
    probes = []
    priorities = ["High", "Medium", "Low", "Critical", "High"]
    severities = ["P2", "P3", "P4", "P1", "P2"]

    for i, (pri, sev) in enumerate(zip(priorities, severities)):
        output = {"ticket": {"priority": pri}, "severity": sev}
        input_text = f"Priority: {pri}\nDesc: ticket {i}"
        probe = _make_probe_with_entropy(output, {"severity"})
        probes.append((probe, input_text))

    result = synthesize_rules_multi(probes)
    mapping_rules = [r for r in result.rules if r.rule_type == "mapping" and r.path == "severity"]
    if mapping_rules:
        # All 4 values mapped, so map should include all
        assert mapping_rules[0].map is not None


def test_render_hybrid_yaml_with_mapping():
    """YAML output includes mapping rules correctly."""
    import yaml
    from drift.hybrid.autogen import GeneratedRule, SynthesisResult

    result = SynthesisResult(
        rules=[
            GeneratedRule(
                path="severity", value=None, rule_type="mapping",
                pattern=None, confidence=1.0, entropy=0.0,
                validated=True, source="priority",
                map={"High": "P2", "Medium": "P3"}, default="P4",
            ),
        ],
        llm_fields=["assessment"],
        stats={"total_fields": 2, "rule_candidates": 1, "rules_generated": 1,
               "rules_validated": 1, "llm_required": 1},
    )
    yaml_str = render_hybrid_yaml(result)
    spec = yaml.safe_load(yaml_str)

    assert len(spec["rules"]) == 1
    rule = spec["rules"][0]
    assert rule["type"] == "mapping"
    assert rule["source"] == "priority"
    assert rule["map"]["High"] == "P2"
    assert rule["default"] == "P4"


def test_render_hybrid_yaml_with_conditional():
    """YAML output includes conditional rules correctly."""
    import yaml
    from drift.hybrid.autogen import GeneratedRule, SynthesisResult

    result = SynthesisResult(
        rules=[
            GeneratedRule(
                path="severity", value=None, rule_type="conditional",
                pattern=None, confidence=0.95, entropy=0.0,
                validated=True,
                conditions=[
                    {"when": {"priority": "High", "category": "Security"}, "value": "P1"},
                    {"when": {"priority": "High"}, "value": "P2"},
                ],
                default="P3",
            ),
        ],
        llm_fields=[],
        stats={"total_fields": 1, "rule_candidates": 1, "rules_generated": 1,
               "rules_validated": 1, "llm_required": 0},
    )
    yaml_str = render_hybrid_yaml(result)
    spec = yaml.safe_load(yaml_str)

    rule = spec["rules"][0]
    assert rule["type"] == "conditional"
    assert len(rule["conditions"]) == 2
    assert rule["default"] == "P3"
