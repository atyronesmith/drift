"""Tests for drift compare functionality."""

import pytest
from drift.cli import _build_comparison, _format_comparison


class TestBuildComparison:
    def test_baseline_vs_baseline(self):
        a = {
            "type": "baseline",
            "exact_match_rate": 1.0,
            "num_unique": 1,
            "mean_edit_distance": 0.0,
            "length_cv": 0.0,
            "model": "qwen2.5:32b",
            "mean_latency_ms": 5000,
        }
        b = {
            "type": "baseline",
            "exact_match_rate": 0.9,
            "num_unique": 2,
            "mean_edit_distance": 0.006,
            "length_cv": 0.001,
            "model": "qwen3:30b",
            "mean_latency_ms": 8000,
        }
        result = _build_comparison(a, b, "run_a", "run_b")
        assert result["a"]["label"] == "run_a"
        assert result["b"]["label"] == "run_b"
        assert result["a"]["model"] == "qwen2.5:32b"
        assert result["b"]["model"] == "qwen3:30b"
        assert result["metrics"]["exact_match_rate"]["a"] == 1.0
        assert result["metrics"]["exact_match_rate"]["b"] == 0.9

    def test_with_functional(self):
        a = {
            "type": "baseline",
            "exact_match_rate": 0.5,
            "functional": {
                "functional_equivalence_rate": 1.0,
                "variants_with_bugs": 0,
            },
        }
        b = {
            "type": "baseline",
            "exact_match_rate": 0.2,
            "functional": {
                "functional_equivalence_rate": 0.0,
                "variants_with_bugs": 5,
            },
        }
        result = _build_comparison(a, b, "a", "b")
        assert result["functional"]["a"] == 1.0
        assert result["functional"]["b"] == 0.0
        assert result["functional"]["a_bugs"] == 0
        assert result["functional"]["b_bugs"] == 5

    def test_no_functional(self):
        a = {"type": "baseline", "exact_match_rate": 1.0}
        b = {"type": "baseline", "exact_match_rate": 0.9}
        result = _build_comparison(a, b, "a", "b")
        assert "functional" not in result

    def test_pipeline_uses_final_for_metrics(self):
        a = {
            "type": "pipeline",
            "final": {
                "exact_match_rate": 0.1,
                "num_unique": 10,
                "mean_edit_distance": 0.7,
                "length_cv": 0.15,
            },
            "per_step": [
                {"name": "s1", "model": "m1"},
                {"name": "s2", "model": "m2"},
            ],
        }
        b = {"type": "baseline", "exact_match_rate": 1.0, "model": "m3"}
        result = _build_comparison(a, b, "a", "b")
        assert result["metrics"]["exact_match_rate"]["a"] == 0.1
        assert result["a"]["model"] == "m1 → m2"


class TestFormatComparison:
    def test_produces_string_with_header(self):
        comp = {
            "a": {"label": "/path/to/run_a", "model": "model_a"},
            "b": {"label": "/path/to/run_b", "model": "model_b"},
            "metrics": {
                "exact_match_rate": {"label": "Exact match", "a": 1.0, "b": 0.9},
                "num_unique": {"label": "Unique outputs", "a": 1, "b": 2},
            },
        }
        text = _format_comparison(comp)
        assert "DRIFT COMPARISON" in text
        assert "run_a" in text
        assert "run_b" in text
        assert "100%" in text
        assert "90%" in text

    def test_handles_none_values(self):
        comp = {
            "a": {"label": "a"},
            "b": {"label": "b"},
            "metrics": {
                "exact_match_rate": {"label": "Exact match", "a": 0.5, "b": None},
            },
        }
        text = _format_comparison(comp)
        assert "50%" in text

    def test_includes_functional(self):
        comp = {
            "a": {"label": "a"},
            "b": {"label": "b"},
            "metrics": {},
            "functional": {"a": 1.0, "b": 0.0, "a_bugs": 0, "b_bugs": 5},
        }
        text = _format_comparison(comp)
        assert "FUNCTIONAL EQUIVALENCE" in text
        assert "100%" in text
        assert "0%" in text

    def test_shows_delta(self):
        comp = {
            "a": {"label": "a"},
            "b": {"label": "b"},
            "metrics": {
                "exact_match_rate": {"label": "Exact match", "a": 0.5, "b": 0.8},
            },
        }
        text = _format_comparison(comp)
        assert "+30%" in text
