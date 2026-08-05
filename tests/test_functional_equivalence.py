"""
Functional equivalence tests for pipeline-generated code variants.

Extracts the implementer step output from each pipeline run, loads the
process_users function, and runs it against a test database to determine
whether textual drift corresponds to functional drift.

Requires saved experiment artifacts under ``results/`` (gitignored). If those
directories are missing, the module is skipped.

Harness logic lives in ``functional_equivalence_lib`` — shared with
``report_functional_equivalence.py``.

Expected non-pass outcomes (see lib module docstring and docs/RESULTS.md):

- ``pipeline-mixed/run_005``: 3 core test **failures** — duplicate dict key bug
  in generated code (Experiment 3 finding; not a drift harness bug).
- ``pipeline-mixed/run_005``: ``works_without_row_factory`` may **fail** for the
  same reason (wrong ``id`` values even without row_factory).
- ``pipeline-structured/run_001``, ``run_005``: ``works_without_row_factory``
  **skips** — variant assumes caller set ``sqlite3.Row`` factory.
"""

import pytest

from functional_equivalence_lib import collect_variants, load_fn, run_test

VARIANTS = list(collect_variants())

if not VARIANTS:
    pytest.skip(
        "results/ pipeline artifacts not found (see docs/RESULTS.md Experiment 3)",
        allow_module_level=True,
    )


@pytest.fixture(params=VARIANTS, ids=[f"{t}/run_{i:03d}" for t, i, _ in VARIANTS])
def variant(request):
    trial, run_idx, code = request.param
    fn, err = load_fn(code)
    if fn is None:
        pytest.fail(f"Code failed to load: {err}")
    return trial, run_idx, fn


def _assert_test(fn, test_name: str):
    status, reason = run_test(fn, test_name)
    if status == "skip":
        pytest.skip(reason or "skipped")
    assert status == "pass", reason


def test_returns_correct_users(variant):
    _assert_test(variant[2], "correct_users")


def test_returns_correct_data(variant):
    _assert_test(variant[2], "correct_data")


def test_uses_parameterized_queries(variant):
    _assert_test(variant[2], "parameterized_queries")


def test_sql_injection_blocked(variant):
    _assert_test(variant[2], "sql_injection_blocked")


def test_no_password_in_result(variant):
    _assert_test(variant[2], "no_password_in_result")


def test_password_value_not_in_result(variant):
    _assert_test(variant[2], "password_value_not_leaked")


def test_batch_query(variant):
    _assert_test(variant[2], "batch_query")


def test_empty_input(variant):
    _assert_test(variant[2], "empty_input")


def test_nonexistent_user(variant):
    _assert_test(variant[2], "nonexistent_user")


def test_mixed_existing_nonexistent(variant):
    _assert_test(variant[2], "mixed_existing_nonexistent")


def test_works_without_row_factory(variant):
    _assert_test(variant[2], "works_without_row_factory")
