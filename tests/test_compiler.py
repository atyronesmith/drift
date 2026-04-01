"""Tests for the engine compiler."""

import json
import re
import textwrap

from drift.hybrid.compiler import compile_engine


SAMPLE_RULES = [
    {"path": "ticket.id", "type": "regex", "pattern": r"ID:\s*(\S+)\s*$"},
    {"path": "ticket.priority", "type": "regex", "pattern": r"Priority:\s*(\S+)\s*$"},
    {"path": "status", "type": "literal", "value": "open"},
    {"path": "severity", "type": "mapping", "source": "ticket.priority",
     "map": {"High": "P2", "Medium": "P3"}, "default": "P4"},
    {"path": "score", "type": "conditional", "conditions": [
        {"when": {"ticket.priority": "High"}, "value": 10},
        {"when": {"ticket.priority": "Medium"}, "value": 5},
    ], "default": 1},
]

SAMPLE_INPUT = "ID:       ABC-123\nPriority: Medium\nDesc: something broke"


def test_compile_produces_valid_python():
    code = compile_engine(SAMPLE_RULES, ["assessment"], "You are a bot.", "test_engine")
    compile(code, "<test>", "exec")  # should not raise


def test_compile_process_function_exists():
    code = compile_engine(SAMPLE_RULES, ["assessment"], "You are a bot.", "test_engine")
    ns = {}
    exec(code, ns)
    assert "process" in ns
    assert callable(ns["process"])


def test_compiled_engine_extracts_fields():
    code = compile_engine(SAMPLE_RULES, ["assessment"], "You are a bot.", "test_engine")
    ns = {}
    exec(code, ns)
    result = ns["process"](SAMPLE_INPUT)

    assert result["ticket"]["id"] == "ABC-123"
    assert result["ticket"]["priority"] == "Medium"
    assert result["status"] == "open"
    assert result["severity"] == "P3"
    assert result["score"] == 5


def test_compiled_engine_conditional_high():
    code = compile_engine(SAMPLE_RULES, [], "You are a bot.", "test_engine")
    ns = {}
    exec(code, ns)
    result = ns["process"]("ID: XYZ-999\nPriority: High\n")

    assert result["severity"] == "P2"
    assert result["score"] == 10


def test_compiled_engine_default_values():
    code = compile_engine(SAMPLE_RULES, [], "You are a bot.", "test_engine")
    ns = {}
    exec(code, ns)
    result = ns["process"]("ID: XYZ-999\nPriority: Low\n")

    assert result["severity"] == "P4"  # mapping default
    assert result["score"] == 1  # conditional default


def test_compiled_engine_no_llm_skips_tier3():
    code = compile_engine(SAMPLE_RULES, ["assessment"], "You are a bot.", "test_engine")
    ns = {}
    exec(code, ns)
    result = ns["process"](SAMPLE_INPUT)  # no llm arg

    # Should have Tier 1+2 fields but no assessment from LLM
    assert "ticket" in result
    assert result["severity"] == "P3"
    assert "assessment" not in result or "summary" not in result.get("assessment", {})


def test_compiled_engine_has_main_block():
    code = compile_engine(SAMPLE_RULES, [], "You are a bot.", "test_engine")
    assert 'if __name__ == "__main__":' in code


def test_compiled_engine_has_docstring():
    code = compile_engine(SAMPLE_RULES, ["x"], "prompt", "my_engine")
    assert "Auto-generated processing engine: my_engine" in code
    assert "Tier 1 (regex/literal): 3 fields" in code
    assert "Tier 2 (mapping/conditional): 2 fields" in code
    assert "Tier 3 (LLM): 1 field groups" in code


def test_compiled_engine_no_llm_fields():
    """Engine with no LLM fields should not include LLM helpers."""
    code = compile_engine(SAMPLE_RULES, [], "prompt", "test")
    assert "_call_llm" not in code
    assert "_parse_json" not in code
    assert "_merge" not in code
