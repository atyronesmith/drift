"""Tests for drift.metrics.stats module."""

import pytest
from drift.metrics.stats import bootstrap_ci, permutation_test


class TestBootstrapCI:
    def test_all_same(self):
        result = bootstrap_ci([1.0, 1.0, 1.0, 1.0, 1.0])
        assert result["estimate"] == 1.0
        assert result["ci_lower"] == 1.0
        assert result["ci_upper"] == 1.0

    def test_binary_values(self):
        result = bootstrap_ci([True, True, True, False, False])
        assert result["estimate"] == pytest.approx(0.6)
        assert 0.1 < result["ci_lower"] < 0.6
        assert 0.6 < result["ci_upper"] <= 1.0

    def test_single_value(self):
        result = bootstrap_ci([0.5])
        assert result["estimate"] == 0.5
        assert result["ci_lower"] is None

    def test_empty(self):
        result = bootstrap_ci([])
        assert result["estimate"] is None

    def test_ci_level(self):
        result = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], ci=0.90)
        assert result["ci_level"] == 0.90


class TestPermutationTest:
    def test_identical_groups(self):
        a = [1.0, 1.0, 1.0, 1.0, 1.0]
        b = [1.0, 1.0, 1.0, 1.0, 1.0]
        result = permutation_test(a, b)
        assert result["observed_diff"] == 0.0
        assert result["p_value"] > 0.05
        assert result["significant"] is False

    def test_different_groups(self):
        a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        b = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        result = permutation_test(a, b)
        assert result["observed_diff"] == 1.0
        assert result["p_value"] < 0.05
        assert result["significant"] is True

    def test_overlapping_groups(self):
        a = [0.1, 0.2, 0.3, 0.4, 0.5]
        b = [0.3, 0.4, 0.5, 0.6, 0.7]
        result = permutation_test(a, b)
        assert result["observed_diff"] == pytest.approx(0.2)
        # With overlapping groups and small N, may or may not be significant
        assert 0.0 < result["p_value"] <= 1.0
