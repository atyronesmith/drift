"""Tests for the engine compiler."""

import json
import re
import textwrap
from pathlib import Path

from click.testing import CliRunner

from drift.cli import cli
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


def test_validate_command(tmp_path):
    """drift validate runs an engine against input files and reports extraction rates."""
    # Generate engine
    code = compile_engine(SAMPLE_RULES, [], "You are a bot.", "test_engine")
    engine_file = tmp_path / "engine.py"
    engine_file.write_text(code)

    # Create input files
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "t1.txt").write_text("ID: ABC-123\nPriority: Medium\nDesc: something")
    (input_dir / "t2.txt").write_text("ID: XYZ-999\nPriority: High\nDesc: urgent")

    runner = CliRunner()
    result = runner.invoke(cli, ["validate", str(engine_file), "--input-dir", str(input_dir)])

    assert result.exit_code == 0
    assert "Processed 2/2 inputs successfully" in result.output
    assert "100%" in result.output


def test_validate_with_expected(tmp_path):
    """drift validate compares against expected outputs when provided."""
    code = compile_engine(SAMPLE_RULES, [], "You are a bot.", "test_engine")
    engine_file = tmp_path / "engine.py"
    engine_file.write_text(code)

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "t1.txt").write_text("ID: ABC-123\nPriority: Medium\nDesc: something")

    expected_dir = tmp_path / "expected"
    expected_dir.mkdir()
    (expected_dir / "t1.json").write_text(json.dumps({
        "ticket": {"id": "ABC-123", "priority": "Medium"},
        "status": "open",
        "severity": "P3",
        "score": 5,
    }))

    runner = CliRunner()
    result = runner.invoke(cli, [
        "validate", str(engine_file),
        "--input-dir", str(input_dir),
        "--expected-dir", str(expected_dir),
    ])

    assert result.exit_code == 0
    assert "Accuracy" in result.output


def test_validate_field_filter(tmp_path):
    """drift validate --fields filters which fields are reported."""
    code = compile_engine(SAMPLE_RULES, [], "You are a bot.", "test_engine")
    engine_file = tmp_path / "engine.py"
    engine_file.write_text(code)

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "t1.txt").write_text("ID: ABC-123\nPriority: High\nDesc: x")

    runner = CliRunner()
    result = runner.invoke(cli, [
        "validate", str(engine_file),
        "--input-dir", str(input_dir),
        "--fields", "ticket.id,severity",
    ])

    assert result.exit_code == 0
    assert "ticket.id" in result.output
    assert "severity" in result.output
    # score and status should not appear since they weren't in --fields
    assert "score" not in result.output
