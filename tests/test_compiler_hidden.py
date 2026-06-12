"""Tests for hidden_probe rule compilation into standalone engines."""

import pytest

from drift.hybrid.compiler import compile_engine

HIDDEN_RULE = {
    "path": "assessment.root_cause_category",
    "type": "hidden_probe",
    "layer": 2,
    "position": "last_input",
    "classes": ["hardware", "software"],
    "weights": [[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]],
    "bias": [0.0, 0.0],
    "confidence_threshold": 0.6,
    "probe_model": "test-model",
}

REGEX_RULE = {
    "path": "ticket.id",
    "type": "regex",
    "pattern": r"Ticket ID:\s*(\S+)",
}


def _build_ns(rules, llm_fields=None):
    code = compile_engine(rules, llm_fields or [], "You are a bot.", "test_engine")
    compile(code, "<test>", "exec")  # valid python
    ns = {}
    exec(code, ns)
    return ns


def test_hidden_probe_engine_compiles():
    ns = _build_ns([REGEX_RULE, HIDDEN_RULE])
    assert "process" in ns
    assert "_fetch_hidden" in ns


def test_hidden_probe_engine_classifies_with_fake_fetch():
    ns = _build_ns([REGEX_RULE, HIDDEN_RULE])
    ns["_fetch_hidden"] = lambda layers, text: {2: [2.0, 0.0, 0.1, 0.1]}
    result = ns["process"]("Ticket ID: T-42\nbroken printer")
    assert result["ticket"]["id"] == "T-42"
    assert result["assessment"]["root_cause_category"] == "hardware"


def test_hidden_probe_engine_escalates_without_server():
    ns = _build_ns([REGEX_RULE, HIDDEN_RULE])

    def broken_fetch(layers, text):
        raise RuntimeError("no server")

    ns["_fetch_hidden"] = broken_fetch
    result = ns["process"]("Ticket ID: T-42\nbroken printer")
    # Regex still works; probe field absent (escalated), no crash
    assert result["ticket"]["id"] == "T-42"
    assert "root_cause_category" not in result.get("assessment", {})


def test_hidden_probe_single_fetch_for_multiple_probes():
    rule2 = dict(HIDDEN_RULE)
    rule2["path"] = "assessment.assignment_group"
    rule2["classes"] = ["FieldOps", "DeskOps"]
    ns = _build_ns([HIDDEN_RULE, rule2])

    calls = []

    def counting_fetch(layers, text):
        calls.append(layers)
        return {2: [2.0, 0.0, 0.1, 0.1]}

    ns["_fetch_hidden"] = counting_fetch
    result = ns["process"]("anything")
    assert len(calls) == 1
    assert calls[0] == [2]
    assert result["assessment"]["root_cause_category"] == "hardware"
    assert result["assessment"]["assignment_group"] == "FieldOps"
