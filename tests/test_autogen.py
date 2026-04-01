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
