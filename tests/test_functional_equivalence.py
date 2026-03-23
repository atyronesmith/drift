"""
Functional equivalence tests for pipeline-generated code variants.

Extracts the implementer step output from each pipeline run, loads the
process_users function, and runs it against a test database to determine
whether textual drift corresponds to functional drift.

Tests map to the 5 planted bugs in the original code:
1. SQL injection via f-string interpolation
2. Password exposed in return data
3. N+1 query pattern
4. No error handling for DB failures
5. Hardcoded column indices (fragile)
"""

import json
import os
import re
import sqlite3
import textwrap
import types
from pathlib import Path

import pytest

RESULTS_DIR = Path(__file__).parent.parent / "results"

TRIALS = {
    "pipeline-homogeneous": "20260306T180616Z",
    "pipeline-mixed": "20260306T180622Z",
    "pipeline-structured": "20260306T180626Z",
}


def extract_code(content: str) -> str:
    match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1)
    return content


def load_process_users(code: str):
    """Compile code and return the process_users function, or None if it fails to load."""
    module = types.ModuleType("variant")
    module.__dict__["sqlite3"] = sqlite3
    # Provide common type aliases that variants may reference without importing
    module.__dict__["Connection"] = sqlite3.Connection
    try:
        exec(compile(code, "<variant>", "exec"), module.__dict__)
    except Exception as e:
        return None, e
    return module.__dict__["process_users"], None


def make_db():
    """Create an in-memory SQLite database with test data."""
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
    return conn


def make_db_with_row_factory():
    conn = make_db()
    conn.row_factory = sqlite3.Row
    return conn


def collect_variants():
    """Yield (trial_name, run_index, code_string) for each implementer output."""
    for trial, timestamp in TRIALS.items():
        run_dir = RESULTS_DIR / trial / timestamp
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


VARIANTS = list(collect_variants())


def variant_id(val):
    if isinstance(val, tuple) and len(val) == 3:
        return f"{val[0]}/run_{val[1]:03d}"
    return str(val)


@pytest.fixture(params=VARIANTS, ids=[f"{t}/run_{i:03d}" for t, i, _ in VARIANTS])
def variant(request):
    trial, run_idx, code = request.param
    fn, err = load_process_users(code)
    if fn is None:
        pytest.fail(f"Code failed to load: {err}")
    return trial, run_idx, code, fn


# --- Test 1: Basic correctness ---


def test_returns_correct_users(variant):
    _, _, code, fn = variant
    db = make_db_with_row_factory()
    result = fn(db, [1, 2])
    ids = {u["id"] for u in result}
    assert ids == {1, 2}, f"Expected users 1,2 but got {ids}"
    for u in result:
        assert "id" in u
        assert "name" in u
        assert "email" in u


def test_returns_correct_data(variant):
    _, _, code, fn = variant
    db = make_db_with_row_factory()
    result = fn(db, [1])
    assert len(result) == 1
    user = result[0]
    assert user["id"] == 1
    assert user["name"] == "Alice"
    assert user["email"] == "alice@example.com"


# --- Test 2: SQL injection prevention ---


class QueryTracker:
    """Wraps a connection to record executed SQL."""

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


def test_uses_parameterized_queries(variant):
    _, _, code, fn = variant
    conn = make_db_with_row_factory()
    tracker = QueryTracker(conn)
    fn(tracker, [1, 2])
    for sql, params in tracker.queries:
        assert "?" in sql or "%s" in sql, f"Query is not parameterized: {sql}"
        assert params is not None, f"No params passed for query: {sql}"


def test_sql_injection_blocked(variant):
    _, _, code, fn = variant
    db = make_db_with_row_factory()
    # A classic SQL injection payload as a user_id.
    # If the code interpolates directly, this would cause an error or unexpected results.
    # With parameterized queries, it should safely return no results or raise a type error.
    try:
        result = fn(db, ["1 OR 1=1"])
        # If it runs, it should not return all users
        assert len(result) <= 1, (
            f"SQL injection succeeded: got {len(result)} users from injection payload"
        )
    except (ValueError, TypeError, sqlite3.OperationalError):
        pass  # Acceptable: code rejected the malicious input


# --- Test 3: Password not exposed ---


def test_no_password_in_result(variant):
    _, _, code, fn = variant
    db = make_db_with_row_factory()
    result = fn(db, [1, 2, 3])
    for user in result:
        assert "password" not in user, (
            f"Password field exposed for user {user.get('id')}"
        )


def test_password_value_not_in_result(variant):
    _, _, code, fn = variant
    db = make_db_with_row_factory()
    result = fn(db, [1])
    user = result[0]
    for v in user.values():
        assert "s3cret" not in str(v), f"Password value leaked: {v}"


# --- Test 4: Batch query (N+1 fix) ---


def test_batch_query(variant):
    _, _, code, fn = variant
    conn = make_db_with_row_factory()
    tracker = QueryTracker(conn)
    fn(tracker, [1, 2, 3])
    # Should use 1 query (IN clause), not 3 separate queries
    assert len(tracker.queries) <= 1, (
        f"N+1 pattern detected: {len(tracker.queries)} queries for 3 users"
    )


# --- Test 5: Empty and edge case handling ---


def test_empty_input(variant):
    _, _, code, fn = variant
    db = make_db_with_row_factory()
    result = fn(db, [])
    assert result == [], f"Expected empty list for empty input, got {result}"


def test_nonexistent_user(variant):
    _, _, code, fn = variant
    db = make_db_with_row_factory()
    result = fn(db, [999])
    assert result == [], f"Expected empty list for nonexistent user, got {result}"


def test_mixed_existing_nonexistent(variant):
    _, _, code, fn = variant
    db = make_db_with_row_factory()
    result = fn(db, [1, 999])
    ids = {u["id"] for u in result}
    assert ids == {1}, f"Expected only user 1, got {ids}"


# --- Test 6: Works without row_factory (robustness) ---


def test_works_without_row_factory(variant):
    """Some variants set row_factory themselves; others require it pre-set.
    This test reveals which variants are self-contained vs. depend on caller setup."""
    _, _, code, fn = variant
    db = make_db()  # No row_factory set
    try:
        result = fn(db, [1])
        assert len(result) == 1
        assert result[0]["id"] == 1
        assert result[0]["name"] == "Alice"
    except (TypeError, KeyError) as e:
        pytest.skip(f"Variant requires row_factory to be pre-set: {e}")
