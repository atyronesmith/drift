"""
Generate a summary report of functional equivalence test results.

Run after: pytest tests/test_functional_equivalence.py
"""

import json
import os
import re
import sqlite3
import types
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

TRIALS = {
    "pipeline-homogeneous": "20260306T180616Z",
    "pipeline-mixed": "20260306T180622Z",
    "pipeline-structured": "20260306T180626Z",
}

TESTS = [
    "correct_users",
    "correct_data",
    "parameterized_queries",
    "sql_injection_blocked",
    "no_password_in_result",
    "password_value_not_leaked",
    "batch_query",
    "empty_input",
    "nonexistent_user",
    "mixed_existing_nonexistent",
    "works_without_row_factory",
]


def extract_code(content: str) -> str:
    match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1)
    return content


def load_fn(code: str):
    module = types.ModuleType("variant")
    module.__dict__["sqlite3"] = sqlite3
    module.__dict__["Connection"] = sqlite3.Connection
    try:
        exec(compile(code, "<variant>", "exec"), module.__dict__)
        return module.__dict__["process_users"], None
    except Exception as e:
        return None, e


def make_db(with_row_factory=True):
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, password TEXT)"
    )
    conn.executemany(
        "INSERT INTO users VALUES (?, ?, ?, ?)",
        [
            (1, "Alice", "alice@example.com", "s3cret_alice"),
            (2, "Bob", "bob@example.com", "s3cret_bob"),
            (3, "Charlie", "charlie@example.com", "s3cret_charlie"),
        ],
    )
    conn.commit()
    if with_row_factory:
        conn.row_factory = sqlite3.Row
    return conn


class QueryTracker:
    def __init__(self, conn):
        self._conn = conn
        self.queries = []
        self.row_factory = conn.row_factory

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def execute(self, sql, params=None):
        self.queries.append((sql, params))
        if params is not None:
            return self._conn.execute(sql, params)
        return self._conn.execute(sql)


def run_test(fn, test_name):
    """Run a single test, return (pass, skip, fail_reason)."""
    try:
        if test_name == "correct_users":
            result = fn(make_db(), [1, 2])
            ids = {u["id"] for u in result}
            assert ids == {1, 2}, f"got {ids}"
            for u in result:
                assert "name" in u, "missing 'name' key"
                assert "email" in u, "missing 'email' key"

        elif test_name == "correct_data":
            result = fn(make_db(), [1])
            u = result[0]
            assert u["id"] == 1 and u["name"] == "Alice" and u["email"] == "alice@example.com"

        elif test_name == "parameterized_queries":
            tracker = QueryTracker(make_db())
            fn(tracker, [1, 2])
            for sql, params in tracker.queries:
                assert "?" in sql or "%s" in sql, f"not parameterized: {sql}"
                assert params is not None

        elif test_name == "sql_injection_blocked":
            try:
                result = fn(make_db(), ["1 OR 1=1"])
                assert len(result) <= 1, f"injection returned {len(result)} users"
            except (ValueError, TypeError, sqlite3.OperationalError):
                pass

        elif test_name == "no_password_in_result":
            result = fn(make_db(), [1, 2, 3])
            for u in result:
                assert "password" not in u

        elif test_name == "password_value_not_leaked":
            result = fn(make_db(), [1])
            for v in result[0].values():
                assert "s3cret" not in str(v)

        elif test_name == "batch_query":
            tracker = QueryTracker(make_db())
            fn(tracker, [1, 2, 3])
            assert len(tracker.queries) <= 1, f"{len(tracker.queries)} queries"

        elif test_name == "empty_input":
            assert fn(make_db(), []) == []

        elif test_name == "nonexistent_user":
            assert fn(make_db(), [999]) == []

        elif test_name == "mixed_existing_nonexistent":
            result = fn(make_db(), [1, 999])
            ids = {u["id"] for u in result}
            assert ids == {1}, f"got {ids}"

        elif test_name == "works_without_row_factory":
            try:
                result = fn(make_db(with_row_factory=False), [1])
                assert len(result) == 1
                assert result[0]["id"] == 1
                assert result[0]["name"] == "Alice"
            except (TypeError, KeyError, RuntimeError):
                return "skip", "requires row_factory"

        return "pass", None
    except Exception as e:
        return "fail", str(e)


def main():
    print("# Functional Equivalence Report")
    print()

    all_results = {}

    for trial, timestamp in TRIALS.items():
        run_dir = RESULTS_DIR / trial / timestamp
        if not run_dir.exists():
            continue

        trial_results = {}
        for i in range(10):
            run_file = run_dir / f"run_{i:03d}.json"
            if not run_file.exists():
                continue
            data = json.loads(run_file.read_text())
            content = data["steps"][1]["response"]["content"]
            code = extract_code(content)
            fn, err = load_fn(code)

            run_id = f"run_{i:03d}"
            if fn is None:
                trial_results[run_id] = {t: ("fail", f"load error: {err}") for t in TESTS}
                continue

            trial_results[run_id] = {}
            for test in TESTS:
                trial_results[run_id][test] = run_test(fn, test)

        all_results[trial] = trial_results

    # Summary per trial
    for trial, runs in all_results.items():
        print(f"## {trial}")
        print()

        # Count pass/fail/skip per test
        test_summary = {t: {"pass": 0, "fail": 0, "skip": 0} for t in TESTS}
        for run_id, tests in runs.items():
            for test, (status, _) in tests.items():
                test_summary[test][status] += 1

        n = len(runs)
        print(f"| Test | Pass | Fail | Skip | Rate |")
        print(f"|------|------|------|------|------|")
        for test in TESTS:
            s = test_summary[test]
            rate = s["pass"] / (s["pass"] + s["fail"]) * 100 if (s["pass"] + s["fail"]) > 0 else 100
            print(f"| {test} | {s['pass']}/{n} | {s['fail']}/{n} | {s['skip']}/{n} | {rate:.0f}% |")
        print()

        # Show failures
        failures = []
        for run_id, tests in runs.items():
            for test, (status, reason) in tests.items():
                if status == "fail":
                    failures.append((run_id, test, reason))

        if failures:
            print("### Failures")
            print()
            for run_id, test, reason in failures:
                print(f"- **{run_id}** / `{test}`: {reason}")
            print()

    # Overall summary
    print("## Overall Summary")
    print()
    total_variants = sum(len(runs) for runs in all_results.values())
    total_tests = total_variants * len(TESTS)
    total_pass = sum(
        1
        for runs in all_results.values()
        for tests in runs.values()
        for status, _ in tests.values()
        if status == "pass"
    )
    total_fail = sum(
        1
        for runs in all_results.values()
        for tests in runs.values()
        for status, _ in tests.values()
        if status == "fail"
    )
    total_skip = sum(
        1
        for runs in all_results.values()
        for tests in runs.values()
        for status, _ in tests.values()
        if status == "skip"
    )

    print(f"- **{total_variants} code variants** tested across 3 pipeline configurations")
    print(f"- **{total_pass}/{total_tests}** test assertions passed ({total_pass/total_tests*100:.1f}%)")
    print(f"- **{total_fail}/{total_tests}** test assertions failed ({total_fail/total_tests*100:.1f}%)")
    print(f"- **{total_skip}/{total_tests}** tests skipped (row_factory dependency)")
    print()

    # Per-variant pass rate
    print("### Per-Variant Functional Pass Rate")
    print()
    core_tests = [t for t in TESTS if t != "works_without_row_factory"]
    print(f"| Trial | Variant | Core Tests ({len(core_tests)}) | Robustness | Bug Found |")
    print(f"|-------|---------|------------|------------|-----------|")
    for trial, runs in all_results.items():
        short = trial.replace("pipeline-", "")
        for run_id, tests in runs.items():
            core_pass = sum(1 for t in core_tests if tests[t][0] == "pass")
            robust = tests["works_without_row_factory"][0]
            has_bug = any(tests[t][0] == "fail" for t in core_tests)
            bug_str = ""
            if has_bug:
                bug_str = ", ".join(t for t in core_tests if tests[t][0] == "fail")
            print(f"| {short} | {run_id} | {core_pass}/{len(core_tests)} | {robust} | {bug_str} |")
    print()


if __name__ == "__main__":
    main()
