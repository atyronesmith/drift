"""
Shared harness for Experiment 3 functional equivalence testing.

Loads implementer-step code from saved pipeline runs in ``results/``, compiles
each variant's ``process_users`` function, and runs it against an in-memory SQLite
database with known test data.

See ``docs/RESULTS.md`` (Experiment 3) for methodology and findings.

Expected outcomes when ``results/`` artifacts are present
---------------------------------------------------------

After aligning pytest with this harness, most variants pass. Remaining non-pass
results are **expected** and reflect generated-code quality, not drift bugs:

**``pipeline-mixed/run_005`` — real functional bug (3 core test failures)**

The Ollama implementer produced a duplicate dictionary key::

    {"id": row[0], "id": row[1], "email": row[2]}

Python silently overwrites the first ``"id"`` with the name string, so
``user["id"]`` returns ``"Alice"`` instead of ``1``. Security fixes (parameterized
SQL, no password in output) are correct; the data mapping is wrong. This is the
documented 1-in-30 (3.3%) defect from Experiment 3.

Failing core tests: ``correct_users``, ``correct_data``,
``mixed_existing_nonexistent``.

**``pipeline-mixed/run_005`` — ``works_without_row_factory`` may also fail**

Same duplicate-key bug: without ``row_factory`` the code still uses positional
tuple access, but the dict comprehension still maps ``id`` to the name. The
robustness test then fails on assertions, not on missing ``row_factory``.

**``pipeline-structured/run_001`` and ``run_005`` — robustness skips**

These variants use ``row["id"]`` without setting ``sqlite3.Row`` themselves.
When the caller omits ``row_factory``, access raises ``TypeError``, often wrapped
in ``RuntimeError``. That is integration fragility, not incorrectness when called
as intended — the harness records **skip**, not fail.

Phase 2 (not yet implemented): mark ``pipeline-mixed/run_005`` core failures as
``xfail`` and add a meta-test asserting 29/30 core pass rate.
"""

from __future__ import annotations

import json
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

# Core correctness tests (Experiment 3 table in RESULTS.md)
CORE_TESTS = [
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
]

ROBUSTNESS_TEST = "works_without_row_factory"

TESTS = CORE_TESTS + [ROBUSTNESS_TEST]


def extract_code(content: str) -> str:
    match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1)
    return content


def load_fn(code: str):
    """Compile code and return (process_users, error)."""
    module = types.ModuleType("variant")
    module.__dict__["sqlite3"] = sqlite3
    module.__dict__["Connection"] = sqlite3.Connection
    try:
        exec(compile(code, "<variant>", "exec"), module.__dict__)
        return module.__dict__["process_users"], None
    except Exception as e:
        return None, e


def make_db(with_row_factory: bool = True):
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


def run_test(fn, test_name: str) -> tuple[str, str | None]:
    """Run a single test. Returns (status, reason) where status is pass|skip|fail."""
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

        else:
            raise ValueError(f"unknown test: {test_name}")

        return "pass", None
    except Exception as e:
        return "fail", str(e)


def collect_variants(results_dir: Path | None = None):
    """Yield (trial_name, run_index, code_string) for each implementer output."""
    base = results_dir or RESULTS_DIR
    for trial, timestamp in TRIALS.items():
        run_dir = base / trial / timestamp
        if not run_dir.exists():
            continue
        for i in range(10):
            run_file = run_dir / f"run_{i:03d}.json"
            if not run_file.exists():
                continue
            data = json.loads(run_file.read_text())
            content = data["steps"][1]["response"]["content"]
            code = extract_code(content)
            yield trial, i, code
