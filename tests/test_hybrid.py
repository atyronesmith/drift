"""Tests for hybrid execution framework."""

import json
import pytest
from pathlib import Path

from drift.hybrid.rules import execute_rule, execute_rules, set_nested, RuleResult
from drift.hybrid.executor import deep_merge, build_hybrid_prompt, _parse_json_response


# --- Rule tests ---

INCIDENT_INPUT = Path(__file__).parent.parent / "tasks" / "incident_triage" / "input.md"


@pytest.fixture
def incident_text():
    return INCIDENT_INPUT.read_text()


def test_regex_rule_simple(incident_text):
    rule = {"path": "incident_id", "type": "regex", "pattern": r"(INC-\d{4}-\d{5})"}
    result = execute_rule(rule, incident_text)
    assert result.matched
    assert result.value == "INC-2026-04821"


def test_regex_rule_named_field(incident_text):
    rule = {"path": "reporter.name", "type": "regex", "pattern": r"Reporter:\s*(.+?)\s*\("}
    result = execute_rule(rule, incident_text)
    assert result.matched
    assert result.value == "Sarah Chen"


def test_regex_rule_no_match():
    rule = {"path": "foo", "type": "regex", "pattern": r"NONEXISTENT_(\d+)"}
    result = execute_rule(rule, "nothing here")
    assert not result.matched
    assert result.value is None


def test_literal_rule():
    rule = {"path": "status", "type": "literal", "value": "open"}
    result = execute_rule(rule, "anything")
    assert result.matched
    assert result.value == "open"


def test_regex_all_rule(incident_text):
    rule = {
        "path": "error_indicators",
        "type": "regex_all",
        "pattern": r"(?:Buffer pool exhausted|GC pause \d+ms exceeds threshold)",
    }
    result = execute_rule(rule, incident_text)
    assert result.matched
    assert len(result.value) >= 2  # at least 2 buffer pool matches


def test_set_nested():
    obj = {}
    set_nested(obj, "a.b.c", 42)
    assert obj == {"a": {"b": {"c": 42}}}


def test_set_nested_existing():
    obj = {"a": {"x": 1}}
    set_nested(obj, "a.y", 2)
    assert obj == {"a": {"x": 1, "y": 2}}


def test_execute_rules_incident(incident_text):
    rules = [
        {"path": "incident_id", "type": "regex", "pattern": r"(INC-\d{4}-\d{5})"},
        {"path": "reporter.name", "type": "regex", "pattern": r"Reporter:\s*(.+?)\s*\("},
        {"path": "account.name", "type": "regex", "pattern": r"Account:\s*(.+?)\s*\("},
        {"path": "account.id", "type": "regex", "pattern": r"\((ACCT-\d+)\)"},
    ]
    extracted, results = execute_rules(rules, incident_text)
    assert extracted["incident_id"] == "INC-2026-04821"
    assert extracted["reporter"]["name"] == "Sarah Chen"
    assert extracted["account"]["name"] == "Cascade Manufacturing Ltd."
    assert extracted["account"]["id"] == "ACCT-4827"
    assert all(r.matched for r in results)


# --- Executor tests ---

def test_deep_merge_simple():
    assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


def test_deep_merge_nested():
    base = {"a": {"x": 1, "y": 2}}
    overlay = {"a": {"y": 3, "z": 4}}
    assert deep_merge(base, overlay) == {"a": {"x": 1, "y": 3, "z": 4}}


def test_deep_merge_overlay_wins():
    assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}


def test_build_hybrid_prompt_default():
    prompt = build_hybrid_prompt(
        "some input", {"field1": "val1"}, ["severity", "root_cause"]
    )
    assert "field1" in prompt
    assert "val1" in prompt
    assert "severity" in prompt
    assert "some input" in prompt


def test_build_hybrid_prompt_template():
    tpl = "Extracted: {extracted_fields}\nAnalyze: {llm_fields}\nInput: {input}"
    prompt = build_hybrid_prompt("my input", {"a": 1}, ["x", "y"], tpl)
    assert '"a": 1' in prompt
    assert "x, y" in prompt
    assert "my input" in prompt


def test_parse_json_response():
    assert _parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_response_fenced():
    assert _parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_response_invalid():
    result = _parse_json_response("not json at all")
    assert "_raw" in result


# --- Full hybrid spec validation ---

def test_incident_hybrid_spec_extracts_all(incident_text):
    """Verify the incident_triage hybrid.yaml rules extract all expected fields."""
    import yaml
    spec_path = Path(__file__).parent.parent / "tasks" / "incident_triage" / "hybrid.yaml"
    spec = yaml.safe_load(spec_path.read_text())

    extracted, results = execute_rules(spec["rules"], incident_text)
    matched = [r for r in results if r.matched]

    # Should match most rules
    assert len(matched) >= 15, f"Only {len(matched)}/{len(results)} rules matched"

    # Verify key extractions
    assert extracted["incident_id"] == "INC-2026-04821"
    assert extracted["reporter"]["name"] == "Sarah Chen"
    assert extracted["account"]["name"] == "Cascade Manufacturing Ltd."
    assert "FlowControl" in extracted["affected_system"]["product"]
    assert "13:45" in extracted["timeline"]["incident_start"]
