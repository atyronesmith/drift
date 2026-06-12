"""Tests for concept-bottleneck programs (drift.hybrid.concepts)."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from drift.hybrid.concepts import eval_concept_program, fit_concept_program
from drift.hybrid.rules import execute_rule


def _synthetic_concepts(n=120, seed=0):
    """Two binary concepts that jointly determine a 3-class label."""
    rng = np.random.default_rng(seed)
    p_lockout = rng.uniform(0, 1, n)
    p_hardware = rng.uniform(0, 1, n)
    y = np.where(p_lockout > 0.5, "IdentityOps",
                 np.where(p_hardware > 0.5, "FieldOps", "DeskOps"))
    X = np.stack([p_lockout, p_hardware], axis=1)
    return X, ["is_lockout", "is_hardware"], y


class TestFitConceptProgram:
    def test_tree_export_roundtrip(self):
        X, names, y = _synthetic_concepts()
        program = fit_concept_program(X, names, y, max_depth=3)
        assert program["accuracy"] > 0.95
        # Exported tree reproduces predictions sample by sample
        for i in range(len(y)):
            concept_values = dict(zip(names, X[i]))
            value, conf = eval_concept_program(program["tree"], concept_values)
            assert value == program["train_predictions"][i]
            assert 0.0 < conf <= 1.0

    def test_tree_is_shallow(self):
        X, names, y = _synthetic_concepts()
        program = fit_concept_program(X, names, y, max_depth=2)

        def depth(node):
            if "value" in node:
                return 0
            return 1 + max(depth(node["then"]), depth(node["else"]))

        assert depth(program["tree"]) <= 2


class TestEvalConceptProgram:
    def test_hand_built_tree(self):
        tree = {
            "if": {"concept": "is_lockout", "op": ">", "threshold": 0.5},
            "then": {"value": "IdentityOps", "confidence": 0.97, "samples": 40},
            "else": {
                "if": {"concept": "is_hardware", "op": ">", "threshold": 0.5},
                "then": {"value": "FieldOps", "confidence": 0.9, "samples": 30},
                "else": {"value": "DeskOps", "confidence": 0.85, "samples": 50},
            },
        }
        assert eval_concept_program(tree, {"is_lockout": 0.9, "is_hardware": 0.1}) == \
            ("IdentityOps", 0.97)
        assert eval_concept_program(tree, {"is_lockout": 0.2, "is_hardware": 0.8}) == \
            ("FieldOps", 0.9)
        assert eval_concept_program(tree, {"is_lockout": 0.2, "is_hardware": 0.2}) == \
            ("DeskOps", 0.85)


class TestConceptProgramRule:
    RULE = {
        "path": "assessment.assignment_group",
        "type": "concept_program",
        "confidence_threshold": 0.8,
        "concepts": [
            # binary probes: p = sigmoid(w @ h + b)
            {"name": "is_lockout", "layer": 2,
             "weights": [4.0, 0.0, 0.0, 0.0], "bias": 0.0},
            {"name": "is_hardware", "layer": 2,
             "weights": [0.0, 4.0, 0.0, 0.0], "bias": 0.0},
        ],
        "program": {
            "if": {"concept": "is_lockout", "op": ">", "threshold": 0.5},
            "then": {"value": "IdentityOps", "confidence": 0.97, "samples": 40},
            "else": {"value": "DeskOps", "confidence": 0.6, "samples": 50},
        },
    }

    def test_confident_branch_matches(self):
        # h dim 0 large -> is_lockout ~ 1 -> IdentityOps @ 0.97 >= 0.8
        result = execute_rule(self.RULE, "", {}, hidden_states={2: [2.0, 0.0, 0.0, 0.0]})
        assert result.matched
        assert result.value == "IdentityOps"

    def test_low_purity_leaf_escalates(self):
        # h dim 0 negative -> is_lockout ~ 0 -> DeskOps @ 0.6 < 0.8
        result = execute_rule(self.RULE, "", {}, hidden_states={2: [-2.0, 0.0, 0.0, 0.0]})
        assert not result.matched
        assert result.escalation_hint == "DeskOps"
        assert result.escalation_confidence == pytest.approx(0.6)

    def test_missing_hidden_states_escalates(self):
        result = execute_rule(self.RULE, "", {}, hidden_states=None)
        assert not result.matched
        assert result.value is None
