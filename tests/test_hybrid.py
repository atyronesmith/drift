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


# --- Mapping rule tests ---

def test_mapping_rule_basic():
    rules = [
        {"path": "ticket.priority", "type": "regex", "pattern": r"Priority:\s*(\S+)\s*$"},
        {"path": "assessment.severity", "type": "mapping", "source": "ticket.priority",
         "map": {"High": "P2", "Medium": "P3", "Low": "P4", "Urgent": "P1"}},
    ]
    extracted, results = execute_rules(rules, "Priority:     Medium\nCategory: Network")
    assert extracted["assessment"]["severity"] == "P3"
    assert results[1].matched


def test_mapping_rule_default():
    rules = [
        {"path": "ticket.priority", "type": "literal", "value": "Unknown"},
        {"path": "assessment.severity", "type": "mapping", "source": "ticket.priority",
         "map": {"High": "P2", "Medium": "P3"}, "default": "P4"},
    ]
    extracted, results = execute_rules(rules, "anything")
    assert extracted["assessment"]["severity"] == "P4"


def test_mapping_rule_missing_source():
    rule = {"path": "severity", "type": "mapping", "source": "nonexistent",
            "map": {"a": "b"}, "default": "P3"}
    result = execute_rule(rule, "text", {})
    assert result.value == "P3"
    assert result.matched


def test_mapping_rule_no_default_no_match():
    rule = {"path": "severity", "type": "mapping", "source": "priority",
            "map": {"High": "P2"}}
    result = execute_rule(rule, "text", {"priority": "Unknown"})
    assert not result.matched


# --- Conditional rule tests ---

def test_conditional_rule_first_match_wins():
    rule = {
        "path": "assessment.severity",
        "type": "conditional",
        "conditions": [
            {"when": {"ticket.priority": "High", "ticket.category": "Account"}, "value": "P3"},
            {"when": {"ticket.priority": "High"}, "value": "P2"},
            {"when": {"ticket.priority": "Medium"}, "value": "P3"},
        ],
        "default": "P4",
    }
    # High + Account → P3 (first condition matches)
    extracted = {"ticket": {"priority": "High", "category": "Account"}}
    result = execute_rule(rule, "text", extracted)
    assert result.value == "P3"

    # High + Network → P2 (second condition matches)
    extracted = {"ticket": {"priority": "High", "category": "Network"}}
    result = execute_rule(rule, "text", extracted)
    assert result.value == "P2"

    # Medium + anything → P3 (third condition)
    extracted = {"ticket": {"priority": "Medium", "category": "Security"}}
    result = execute_rule(rule, "text", extracted)
    assert result.value == "P3"


def test_conditional_rule_default():
    rule = {
        "path": "severity",
        "type": "conditional",
        "conditions": [
            {"when": {"priority": "High"}, "value": "P2"},
        ],
        "default": "P4",
    }
    result = execute_rule(rule, "text", {"priority": "Low"})
    assert result.value == "P4"


def test_conditional_rule_no_default_no_match():
    rule = {
        "path": "severity",
        "type": "conditional",
        "conditions": [
            {"when": {"priority": "High"}, "value": "P2"},
        ],
    }
    result = execute_rule(rule, "text", {"priority": "Low"})
    assert not result.matched


def test_conditional_with_regex_pipeline():
    """Conditional rule uses fields extracted by earlier regex rules."""
    rules = [
        {"path": "ticket.priority", "type": "regex", "pattern": r"Priority:\s*(\S+)\s*$"},
        {"path": "ticket.category", "type": "regex", "pattern": r"Category:\s*(\S+)\s*$"},
        {"path": "assessment.severity", "type": "conditional", "conditions": [
            {"when": {"ticket.priority": "High", "ticket.category": "Account"}, "value": "P3"},
            {"when": {"ticket.priority": "High"}, "value": "P2"},
            {"when": {"ticket.priority": "Medium"}, "value": "P3"},
        ], "default": "P4"},
    ]
    text = "Priority:     High\nCategory:     Network\n"
    extracted, results = execute_rules(rules, text)
    assert extracted["assessment"]["severity"] == "P2"

    text = "Priority:     High\nCategory:     Account\n"
    extracted, results = execute_rules(rules, text)
    assert extracted["assessment"]["severity"] == "P3"


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
