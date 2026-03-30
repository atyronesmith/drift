"""Tests for drift.analysis module."""

import pytest
from drift.analysis import analyze_metrics, format_report, _classify_equivalence


class TestClassifyEquivalence:
    def test_distill_band(self):
        result = _classify_equivalence(1.0)
        assert result["band"] == "distill"

    def test_distill_at_threshold(self):
        result = _classify_equivalence(0.99)
        assert result["band"] == "distill"

    def test_distill_with_fallback(self):
        result = _classify_equivalence(0.967)
        assert result["band"] == "distill_with_fallback"

    def test_keep_with_gates(self):
        result = _classify_equivalence(0.90)
        assert result["band"] == "keep_with_gates"

    def test_unreliable(self):
        result = _classify_equivalence(0.50)
        assert result["band"] == "unreliable"

    def test_zero(self):
        result = _classify_equivalence(0.0)
        assert result["band"] == "unreliable"


class TestAnalyzeMetrics:
    def test_baseline_with_functional(self):
        metrics = {
            "type": "baseline",
            "exact_match_rate": 0.833,
            "num_unique": 3,
            "num_outputs": 30,
            "mean_edit_distance": 0.0008,
            "length_cv": 0.001,
            "functional": {
                "functional_equivalence_rate": 1.0,
                "variants_all_pass": 30,
                "variants_with_bugs": 0,
                "num_variants": 30,
                "num_tests": 12,
                "per_test": {},
                "failures": [],
            },
        }
        report = analyze_metrics(metrics)
        assert report["verdict"]["band"] == "distill"
        assert report["verdict"]["confidence"] == "high"
        assert report["functional"]["available"] is True

    def test_baseline_without_functional(self):
        metrics = {
            "type": "baseline",
            "exact_match_rate": 0.5,
            "num_unique": 5,
            "num_outputs": 10,
            "mean_edit_distance": 0.3,
            "length_cv": 0.05,
        }
        report = analyze_metrics(metrics)
        assert report["verdict"]["band"] == "needs_functional_tests"
        assert report["verdict"]["confidence"] == "low"

    def test_deterministic_without_functional(self):
        metrics = {
            "type": "baseline",
            "exact_match_rate": 1.0,
            "num_unique": 1,
            "num_outputs": 30,
            "mean_edit_distance": 0.0,
            "length_cv": 0.0,
        }
        report = analyze_metrics(metrics)
        assert report["verdict"]["band"] == "likely_distillable"

    def test_weak_tests_produce_warning(self):
        metrics = {
            "type": "baseline",
            "exact_match_rate": 0.1,
            "num_unique": 10,
            "num_outputs": 10,
            "mean_edit_distance": 0.6,
            "length_cv": 0.05,
            "functional": {
                "functional_equivalence_rate": 0.90,
                "variants_all_pass": 9,
                "variants_with_bugs": 1,
                "num_variants": 10,
                "num_tests": 5,
                "per_test": {
                    "test_a": {"pass": 9, "fail": 1, "error": 0, "skip": 0, "pass_rate": 0.9},
                },
                "failures": [],
            },
        }
        report = analyze_metrics(metrics)
        assert report["verdict"]["band"] == "keep_with_gates"
        assert "warning" in report["verdict"]
        assert "test_a" in report["verdict"]["warning"]

    def test_pipeline_with_drift_amplification(self):
        metrics = {
            "type": "pipeline",
            "num_steps": 3,
            "num_runs": 10,
            "per_step": [
                {"name": "step1", "exact_match_rate": 0.5, "num_unique": 5, "mean_edit_distance": 0.3},
                {"name": "step2", "exact_match_rate": 0.2, "num_unique": 8, "mean_edit_distance": 0.5},
                {"name": "step3", "exact_match_rate": 0.1, "num_unique": 10, "mean_edit_distance": 0.7},
            ],
            "final": {
                "exact_match_rate": 0.1,
                "num_unique": 10,
                "num_outputs": 10,
                "mean_edit_distance": 0.7,
                "length_cv": 0.05,
            },
        }
        report = analyze_metrics(metrics)
        steps = report["pipeline_steps"]
        observations = [s for s in steps if s["type"] == "observation"]
        assert len(observations) == 1
        assert "amplifies" in observations[0]["detail"]

    def test_confidence_by_sample_size(self):
        for n, expected in [(30, "high"), (10, "medium"), (3, "low")]:
            metrics = {
                "type": "baseline",
                "exact_match_rate": 1.0,
                "functional": {
                    "functional_equivalence_rate": 1.0,
                    "variants_all_pass": n,
                    "variants_with_bugs": 0,
                    "num_variants": n,
                    "num_tests": 1,
                    "per_test": {},
                    "failures": [],
                },
            }
            report = analyze_metrics(metrics)
            assert report["verdict"]["confidence"] == expected, f"n={n}: expected {expected}"


class TestFormatReport:
    def test_produces_string(self):
        metrics = {
            "type": "baseline",
            "exact_match_rate": 1.0,
            "num_unique": 1,
            "num_outputs": 30,
            "mean_edit_distance": 0.0,
            "length_cv": 0.0,
            "functional": {
                "functional_equivalence_rate": 1.0,
                "variants_all_pass": 30,
                "variants_with_bugs": 0,
                "num_variants": 30,
                "num_tests": 5,
                "per_test": {},
                "failures": [],
            },
        }
        report = analyze_metrics(metrics)
        text = format_report(report)
        assert "DRIFT ANALYSIS REPORT" in text
        assert "distill" in text
        assert "100.0%" in text

    def test_length_variance_reported(self):
        metrics = {
            "type": "baseline",
            "exact_match_rate": 0.5,
            "num_unique": 5,
            "num_outputs": 10,
            "mean_edit_distance": 0.3,
            "length_cv": 0.25,
        }
        report = analyze_metrics(metrics)
        text = format_report(report)
        assert "length_variance" in text
