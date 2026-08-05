"""Tests for the Tier 4 hidden_probe rule type (logistic probe on hidden states)."""

import math

import pytest

from drift.hybrid.rules import RuleResult, execute_rule, execute_rules


def _make_probe_rule(threshold=0.6):
    # 2 classes, 4-dim hidden state. Weights chosen so the class is obvious.
    return {
        "path": "assessment.root_cause_category",
        "type": "hidden_probe",
        "layer": 2,
        "position": "last_input",
        "classes": ["hardware", "software"],
        "weights": [
            [4.0, 0.0, 0.0, 0.0],   # hardware: fires on dim 0
            [0.0, 4.0, 0.0, 0.0],   # software: fires on dim 1
        ],
        "bias": [0.0, 0.0],
        "confidence_threshold": threshold,
        "probe_model": "test-model",
    }


HARDWARE_VEC = [2.0, 0.0, 0.1, 0.1]   # clearly hardware
AMBIGUOUS_VEC = [0.1, 0.1, 0.0, 0.0]  # near-uniform logits


class TestHiddenProbeRule:
    def test_confident_prediction_matches(self):
        rule = _make_probe_rule(threshold=0.6)
        result = execute_rule(rule, "ignored input text", {},
                              hidden_states={2: HARDWARE_VEC})
        assert result.matched
        assert result.value == "hardware"

    def test_distribution_returned_and_normalized(self):
        rule = _make_probe_rule(threshold=0.6)
        result = execute_rule(rule, "", {}, hidden_states={2: HARDWARE_VEC})
        assert result.distribution is not None
        assert set(result.distribution) == {"hardware", "software"}
        assert math.isclose(sum(result.distribution.values()), 1.0, rel_tol=1e-6)
        assert result.distribution["hardware"] > result.distribution["software"]

    def test_below_threshold_escalates_with_hint(self):
        rule = _make_probe_rule(threshold=0.99)
        result = execute_rule(rule, "", {}, hidden_states={2: AMBIGUOUS_VEC})
        assert not result.matched
        assert result.value is None
        assert result.escalation_hint in ("hardware", "software")
        assert result.escalation_confidence is not None
        assert result.escalation_confidence < 0.99
        # distribution still reported for monitoring even on escalation
        assert result.distribution is not None

    def test_missing_hidden_states_escalates(self):
        rule = _make_probe_rule()
        result = execute_rule(rule, "", {}, hidden_states=None)
        assert not result.matched
        assert result.value is None

    def test_missing_layer_escalates(self):
        rule = _make_probe_rule()
        result = execute_rule(rule, "", {}, hidden_states={0: HARDWARE_VEC})
        assert not result.matched

    def test_dimension_mismatch_escalates(self):
        rule = _make_probe_rule()
        result = execute_rule(rule, "", {}, hidden_states={2: [1.0, 2.0]})
        assert not result.matched


class TestExecuteRulesThreading:
    def test_mixed_rules_share_hidden_states(self):
        rules = [
            {"path": "ticket.id", "type": "regex", "pattern": r"Ticket ID:\s*(\S+)"},
            _make_probe_rule(threshold=0.6),
        ]
        extracted, results = execute_rules(
            rules, "Ticket ID: T-42\nthe printer is broken",
            hidden_states={2: HARDWARE_VEC},
        )
        assert extracted["ticket"]["id"] == "T-42"
        assert extracted["assessment"]["root_cause_category"] == "hardware"

    def test_no_hidden_states_keeps_other_rules_working(self):
        rules = [
            {"path": "ticket.id", "type": "regex", "pattern": r"Ticket ID:\s*(\S+)"},
            _make_probe_rule(),
        ]
        extracted, results = execute_rules(rules, "Ticket ID: T-42")
        assert extracted["ticket"]["id"] == "T-42"
        assert "assessment" not in extracted  # probe escalated, no crash
