"""Tests for hidden-state fetching in the hybrid executor."""

import asyncio
from dataclasses import dataclass

import pytest

import drift.hybrid.executor as executor_mod
from drift.hybrid.executor import run_hybrid


@dataclass
class FakeResponse:
    content: str = '{"assessment": {"summary": "ok"}}'
    total_ms: float = 1.0


class FakeProvider:
    async def complete(self, system, prompt, **kwargs):
        return FakeResponse()


class FakeHiddenResult:
    def __init__(self, layers, vec):
        self.layers = layers
        self.positions = [9]
        self._vec = vec

    def vector(self, layer, position):
        return self._vec


PROBE_RULE = {
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

REGEX_RULE = {"path": "ticket.id", "type": "regex", "pattern": r"Ticket ID:\s*(\S+)"}


def test_single_hidden_fetch_for_multiple_probes(monkeypatch):
    calls = []

    def fake_remote(system, prompt, base_url=None, layers=None, positions="last_input",
                    prefill_only=True, **kwargs):
        calls.append({"layers": layers})
        return FakeHiddenResult(layers, [2.0, 0.0, 0.1, 0.1])

    monkeypatch.setattr(executor_mod, "_fetch_hidden_states_remote", fake_remote)

    probe_rule_2 = dict(PROBE_RULE)
    probe_rule_2["path"] = "assessment.assignment_group"
    probe_rule_2["classes"] = ["FieldOps", "DeskOps"]

    result = asyncio.run(run_hybrid(
        provider=FakeProvider(),
        system_prompt="triage",
        input_text="Ticket ID: T-42\nthe printer is broken",
        rules=[REGEX_RULE, PROBE_RULE, probe_rule_2],
        llm_fields=["assessment"],
    ))

    assert len(calls) == 1                      # one prefill for both probes
    assert calls[0]["layers"] == [2]            # union of required layers
    assert result.extracted["ticket"]["id"] == "T-42"
    assert result.extracted["assessment"]["root_cause_category"] == "hardware"
    assert result.extracted["assessment"]["assignment_group"] == "FieldOps"


def test_no_probe_rules_no_fetch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        executor_mod, "_fetch_hidden_states_remote",
        lambda *a, **k: calls.append(1),
    )
    result = asyncio.run(run_hybrid(
        provider=FakeProvider(),
        system_prompt="triage",
        input_text="Ticket ID: T-42",
        rules=[REGEX_RULE],
        llm_fields=["assessment"],
    ))
    assert calls == []
    assert result.extracted["ticket"]["id"] == "T-42"


def test_hidden_fetch_failure_degrades_to_escalation(monkeypatch):
    def broken_remote(*a, **k):
        raise RuntimeError("server down")

    monkeypatch.setattr(executor_mod, "_fetch_hidden_states_remote", broken_remote)

    result = asyncio.run(run_hybrid(
        provider=FakeProvider(),
        system_prompt="triage",
        input_text="Ticket ID: T-42",
        rules=[REGEX_RULE, PROBE_RULE],
        llm_fields=["assessment"],
    ))
    # Probe escalated (no crash), regex still extracted
    assert result.extracted["ticket"]["id"] == "T-42"
    assert "root_cause_category" not in result.extracted.get("assessment", {})
