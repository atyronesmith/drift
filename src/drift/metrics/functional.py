"""
Layer 4: Functional equivalence metrics.

Loads a task-specific test module, extracts testable functions from LLM outputs,
and runs them against the test suite. Reports pass/fail/skip per test per variant
and aggregate equivalence rates.

Test module interface (e.g., tasks/code_review/functional_tests.py):

    def extract(code: str) -> callable
        Extract the testable function from raw LLM output text.
        Should handle markdown fences, imports, etc.
        Raise on failure.

    tests: dict[str, callable]
        Maps test names to test functions. Each test function takes
        the extracted callable and raises AssertionError on failure.
        Return "skip" to skip the test.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path


def load_test_module(module_path: Path):
    """Load a functional test module from a file path."""
    spec = importlib.util.spec_from_file_location("_functional_tests", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "extract"):
        raise ValueError(f"{module_path} must define an extract(code) function")
    if not hasattr(mod, "tests"):
        raise ValueError(f"{module_path} must define a tests dict")
    return mod


def _run_variant(extract_fn, tests: dict, output: str) -> dict:
    """Run all tests on a single output variant. Returns per-test results."""
    try:
        fn = extract_fn(output)
    except Exception as e:
        return {name: ("error", f"extract failed: {e}") for name in tests}

    results = {}
    for name, test_fn in tests.items():
        # Suppress stdout/stderr from variant code (print statements, tracebacks)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            ret = test_fn(fn)
            if ret == "skip":
                results[name] = ("skip", None)
            else:
                results[name] = ("pass", None)
        except AssertionError as e:
            results[name] = ("fail", str(e))
        except Exception as e:
            results[name] = ("error", str(e))
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
    return results


def compute_functional_metrics(outputs: list[str], test_module_path: Path) -> dict:
    """Run functional tests on a list of code outputs.

    Returns a dict with aggregate metrics and per-variant detail.
    """
    mod = load_test_module(test_module_path)
    test_names = list(mod.tests.keys())

    per_variant = []
    for i, output in enumerate(outputs):
        results = _run_variant(mod.extract, mod.tests, output)
        per_variant.append(results)

    # Aggregate per-test pass rates
    per_test = {}
    for name in test_names:
        statuses = [v[name][0] for v in per_variant]
        n_pass = statuses.count("pass")
        n_fail = statuses.count("fail")
        n_error = statuses.count("error")
        n_skip = statuses.count("skip")
        n_evaluated = n_pass + n_fail + n_error
        per_test[name] = {
            "pass": n_pass,
            "fail": n_fail,
            "error": n_error,
            "skip": n_skip,
            "pass_rate": round(n_pass / n_evaluated, 4) if n_evaluated > 0 else None,
        }

    # Aggregate per-variant
    n_variants = len(outputs)
    variants_all_pass = sum(
        1 for v in per_variant
        if all(v[name][0] in ("pass", "skip") for name in test_names)
    )
    variants_with_bugs = sum(
        1 for v in per_variant
        if any(v[name][0] in ("fail", "error") for name in test_names)
    )

    # Failures detail
    failures = []
    for i, v in enumerate(per_variant):
        failed = [name for name in test_names if v[name][0] in ("fail", "error")]
        if failed:
            failures.append({
                "variant": i,
                "failed_tests": {name: v[name][1] for name in failed},
            })

    return {
        "functional_equivalence_rate": round(variants_all_pass / n_variants, 4) if n_variants > 0 else None,
        "variants_all_pass": variants_all_pass,
        "variants_with_bugs": variants_with_bugs,
        "num_variants": n_variants,
        "num_tests": len(test_names),
        "per_test": per_test,
        "failures": failures,
    }
