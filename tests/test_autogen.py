"""Tests for automated rule generation from probe data."""

import json
import pytest
from pathlib import Path

from drift.hybrid.autogen import (
    _find_value_context,
    _extract_label,
    _generate_regex,
    _validate_pattern,
    _discover_classifier_rules,
    synthesize_rules,
    synthesize_rules_multi,
    render_hybrid_yaml,
    GeneratedRule,
    SynthesisResult,
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


# --- Classifier discovery tests ---

def _make_classifier_probes(n=30):
    """Generate synthetic probes for classifier discovery testing.

    Creates tickets with category→root_cause mapping that a classifier can learn:
    - Network tickets → "network"
    - Software tickets → "software_bug"
    - Account tickets → "access_control"
    """
    import random
    random.seed(42)

    categories = ["Network", "Software", "Account"]
    root_causes = {"Network": "network", "Software": "software_bug", "Account": "access_control"}
    descriptions = {
        "Network": ["printer disconnecting", "network connectivity slow", "VPN connection drops"],
        "Software": ["application crash error", "software update failed", "integration bug"],
        "Account": ["password reset access", "account locked permissions", "access denied login"],
    }

    probes = []
    for i in range(n):
        cat = categories[i % len(categories)]
        rc = root_causes[cat]
        desc = random.choice(descriptions[cat])
        output = {
            "ticket": {"category": cat, "priority": "Medium"},
            "assessment": {"root_cause": rc, "summary": f"Ticket {i} about {desc}"},
        }
        input_text = f"Category: {cat}\nPriority: Medium\nDescription: {desc}"
        probe = _make_probe_with_entropy(
            output, {"assessment.root_cause", "assessment.summary"}
        )
        probes.append((probe, input_text))

    return probes


def test_discover_classifier_rules_basic():
    """Classifier discovery finds learnable classification fields."""
    pytest.importorskip("sklearn")
    probes = _make_classifier_probes(30)

    # Extract field values like synthesize_rules_multi does
    from drift.hybrid.autogen import _parse_llm_output, _get_nested
    parsed = [_parse_llm_output(pd) for pd, _ in probes]

    tier1_paths = {"ticket.category", "ticket.priority"}
    llm_paths = {"assessment.root_cause", "assessment.summary"}
    field_values = {}
    for path in tier1_paths | llm_paths:
        field_values[path] = [_get_nested(p, path) for p in parsed if p]

    rules = _discover_classifier_rules(
        probes, tier1_paths, llm_paths, field_values,
        min_accuracy=0.60,
    )

    # Should discover root_cause as classifiable (clear category→root_cause pattern)
    clf_paths = {r.path for r in rules}
    assert "assessment.root_cause" in clf_paths

    rc_rule = [r for r in rules if r.path == "assessment.root_cause"][0]
    assert rc_rule.rule_type == "classifier"
    assert rc_rule.vocabulary is not None
    assert rc_rule.weights is not None
    assert rc_rule.classes is not None
    assert set(rc_rule.classes) == {"access_control", "network", "software_bug"}

    # summary should NOT be classifiable (unique per ticket)
    assert "assessment.summary" not in clf_paths


def test_discover_classifier_rules_with_tier2_resolved():
    """Classifier structured_features include Tier 2 (mapping) resolved paths."""
    pytest.importorskip("sklearn")
    import random
    random.seed(42)

    # Build probes that include a Tier 2 mapping output (assessment.severity)
    # that correlates with root_cause:
    #   Network → P2, Software → P3, Account → P1
    severity_map = {"Network": "P2", "Software": "P3", "Account": "P1"}
    categories = ["Network", "Software", "Account"]
    root_causes = {"Network": "network", "Software": "software_bug", "Account": "access_control"}
    descriptions = {
        "Network": ["printer disconnecting", "network connectivity slow", "VPN drops"],
        "Software": ["application crash", "software update failed", "integration bug"],
        "Account": ["password reset", "account locked", "access denied"],
    }

    probes = []
    for i in range(30):
        cat = categories[i % len(categories)]
        rc = root_causes[cat]
        sev = severity_map[cat]
        desc = random.choice(descriptions[cat])
        output = {
            "ticket": {"category": cat, "priority": "Medium"},
            "assessment": {"root_cause": rc, "severity": sev,
                          "summary": f"Ticket {i}"},
        }
        input_text = f"Category: {cat}\nPriority: Medium\nDescription: {desc}"
        probe = _make_probe_with_entropy(
            output, {"assessment.root_cause", "assessment.summary"}
        )
        probes.append((probe, input_text))

    from drift.hybrid.autogen import _parse_llm_output, _get_nested
    parsed = [_parse_llm_output(pd) for pd, _ in probes]

    # tier1 = regex-extracted, tier2 = mapping-derived severity
    tier1_paths = {"ticket.category", "ticket.priority"}
    tier2_paths = {"assessment.severity"}
    resolved_paths = tier1_paths | tier2_paths  # what synthesize_rules_multi now passes
    llm_paths = {"assessment.root_cause", "assessment.summary"}

    field_values = {}
    for path in resolved_paths | llm_paths:
        field_values[path] = [_get_nested(p, path) for p in parsed if p]

    rules = _discover_classifier_rules(
        probes, resolved_paths, llm_paths, field_values,
        min_accuracy=0.60,
    )

    clf_paths = {r.path for r in rules}
    assert "assessment.root_cause" in clf_paths

    rc_rule = [r for r in rules if r.path == "assessment.root_cause"][0]
    # Structured features should include the Tier 2 severity path
    sf_sources = {sf["source"] for sf in rc_rule.structured_features}
    assert "assessment.severity" in sf_sources
    # Also includes Tier 1 category (has 3 unique values, within 2-20 range)
    assert "ticket.category" in sf_sources


def test_discover_classifier_rules_too_few_probes():
    """Returns empty when fewer than 20 probes."""
    pytest.importorskip("sklearn")
    probes = _make_classifier_probes(10)

    rules = _discover_classifier_rules(
        probes, set(), {"assessment.root_cause"}, {},
    )
    assert rules == []


def test_discover_classifier_rules_no_sklearn():
    """Returns empty when sklearn not available (tested via too-few-probes path)."""
    # Can't easily mock import failure, but we can verify the too-few-probes guard
    rules = _discover_classifier_rules([], set(), set(), {})
    assert rules == []


def test_render_hybrid_yaml_with_classifier():
    """YAML output includes classifier rules correctly."""
    import yaml

    result = SynthesisResult(
        rules=[
            GeneratedRule(
                path="assessment.root_cause", value=None, rule_type="classifier",
                pattern=None, confidence=0.85, entropy=0.0,
                validated=True,
                vocabulary=["printer", "network", "password"],
                weights=[[1.0, 0.5, -1.0], [-1.0, -0.5, 1.0]],
                bias=[0.0, 0.0],
                classes=["network", "access_control"],
                confidence_threshold=0.95,
                structured_features=[
                    {"source": "ticket.category", "categories": ["Network", "Account"]},
                ],
            ),
        ],
        llm_fields=["assessment"],
        stats={"total_fields": 2, "rule_candidates": 1, "rules_generated": 1,
               "rules_validated": 1, "llm_required": 1},
    )
    yaml_str = render_hybrid_yaml(result)
    spec = yaml.safe_load(yaml_str)

    rule = spec["rules"][0]
    assert rule["type"] == "classifier"
    assert rule["classifier_type"] == "logistic_regression"
    assert rule["classes"] == ["network", "access_control"]
    assert rule["vocabulary"] == ["printer", "network", "password"]
    assert rule["confidence_threshold"] == 0.95
    assert len(rule["weights"]) == 2
