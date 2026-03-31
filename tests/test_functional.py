"""Tests for drift.metrics.functional module."""

import pytest
import tempfile
from pathlib import Path

from drift.metrics.functional import (
    load_test_module,
    _run_variant,
    compute_functional_metrics,
)


@pytest.fixture
def simple_test_module(tmp_path):
    """Create a minimal functional test module."""
    module_file = tmp_path / "test_mod.py"
    module_file.write_text('''
def extract(code):
    """Parse an integer from the output."""
    return int(code.strip())

def test_positive(val):
    assert val > 0, f"Expected positive, got {val}"

def test_even(val):
    assert val % 2 == 0, f"Expected even, got {val}"

def test_skip_if_big(val):
    if val > 100:
        return "skip"
    assert val < 50, f"Expected < 50, got {val}"

tests = {
    "positive": test_positive,
    "even": test_even,
    "skip_if_big": test_skip_if_big,
}
''')
    return module_file


class TestLoadTestModule:
    def test_loads_valid_module(self, simple_test_module):
        mod = load_test_module(simple_test_module)
        assert hasattr(mod, "extract")
        assert hasattr(mod, "tests")
        assert len(mod.tests) == 3

    def test_missing_extract_raises(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("tests = {}")
        with pytest.raises(ValueError, match="extract"):
            load_test_module(f)

    def test_missing_tests_raises(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def extract(c): return c")
        with pytest.raises(ValueError, match="tests"):
            load_test_module(f)


class TestRunVariant:
    def test_all_pass(self, simple_test_module):
        mod = load_test_module(simple_test_module)
        results = _run_variant(mod.extract, mod.tests, "42")
        assert results["positive"] == ("pass", None)
        assert results["even"] == ("pass", None)
        assert results["skip_if_big"] == ("pass", None)

    def test_failure(self, simple_test_module):
        mod = load_test_module(simple_test_module)
        results = _run_variant(mod.extract, mod.tests, "3")
        assert results["positive"] == ("pass", None)
        assert results["even"][0] == "fail"
        assert "Expected even" in results["even"][1]

    def test_skip(self, simple_test_module):
        mod = load_test_module(simple_test_module)
        results = _run_variant(mod.extract, mod.tests, "200")
        assert results["skip_if_big"] == ("skip", None)

    def test_extract_error(self, simple_test_module):
        mod = load_test_module(simple_test_module)
        results = _run_variant(mod.extract, mod.tests, "not_a_number")
        for name in mod.tests:
            assert results[name][0] == "error"
            assert "extract failed" in results[name][1]

    def test_suppresses_stdout(self, tmp_path):
        """Variant code that prints should not leak to stdout."""
        f = tmp_path / "noisy.py"
        f.write_text('''
def extract(code):
    return code

def test_noisy(val):
    print("NOISY OUTPUT")
    assert True

tests = {"noisy": test_noisy}
''')
        mod = load_test_module(f)
        import io, sys
        captured = io.StringIO()
        old = sys.stdout
        sys.stdout = captured
        _run_variant(mod.extract, mod.tests, "hello")
        sys.stdout = old
        assert "NOISY" not in captured.getvalue()


class TestComputeFunctionalMetrics:
    def test_all_pass(self, simple_test_module):
        outputs = ["2", "4", "6", "8", "10"]
        metrics = compute_functional_metrics(outputs, simple_test_module)
        assert metrics["functional_equivalence_rate"] == 1.0
        assert metrics["variants_all_pass"] == 5
        assert metrics["variants_with_bugs"] == 0
        assert metrics["num_variants"] == 5
        assert metrics["num_tests"] == 3
        assert metrics["failures"] == []

    def test_partial_failures(self, simple_test_module):
        outputs = ["2", "3", "4"]  # 3 is odd
        metrics = compute_functional_metrics(outputs, simple_test_module)
        assert metrics["functional_equivalence_rate"] == pytest.approx(0.6667, abs=0.001)
        assert metrics["variants_all_pass"] == 2
        assert metrics["variants_with_bugs"] == 1
        assert len(metrics["failures"]) == 1
        assert metrics["failures"][0]["variant"] == 1
        assert "even" in metrics["failures"][0]["failed_tests"]

    def test_all_fail(self, simple_test_module):
        outputs = ["-1", "-3", "-5"]  # negative and odd
        metrics = compute_functional_metrics(outputs, simple_test_module)
        assert metrics["functional_equivalence_rate"] == 0.0
        assert metrics["variants_with_bugs"] == 3

    def test_skip_counts_as_pass(self, simple_test_module):
        outputs = ["200"]  # > 100, so skip_if_big skips
        metrics = compute_functional_metrics(outputs, simple_test_module)
        assert metrics["per_test"]["skip_if_big"]["skip"] == 1
        assert metrics["per_test"]["skip_if_big"]["pass_rate"] is None  # 0 evaluated
        assert metrics["variants_all_pass"] == 1  # skip doesn't fail

    def test_extract_errors_counted(self, simple_test_module):
        outputs = ["not_a_number"]
        metrics = compute_functional_metrics(outputs, simple_test_module)
        assert metrics["variants_with_bugs"] == 1
        for test_info in metrics["per_test"].values():
            assert test_info["error"] == 1

    def test_empty_outputs(self, simple_test_module):
        metrics = compute_functional_metrics([], simple_test_module)
        assert metrics["functional_equivalence_rate"] is None
        assert metrics["num_variants"] == 0
