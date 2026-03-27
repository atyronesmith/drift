"""
Functional equivalence tests for the code_review task.

Tests the implementer step's output against the 5 planted bugs
in the original process_users function.
"""

import re
import sqlite3
import types


def extract(code: str):
    """Extract the process_users function from LLM output."""
    # Strip markdown fences
    match = re.search(r"```python\n(.*?)```", code, re.DOTALL)
    if match:
        code = match.group(1)

    module = types.ModuleType("variant")
    module.__dict__["sqlite3"] = sqlite3
    module.__dict__["Connection"] = sqlite3.Connection
    exec(compile(code, "<variant>", "exec"), module.__dict__)
    return module.__dict__["process_users"]


# --- Test fixtures ---

def _make_db(with_row_factory=True):
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


class _QueryTracker:
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


# --- Tests ---

def test_correct_users(fn):
    result = fn(_make_db(), [1, 2])
    ids = {u["id"] for u in result}
    assert ids == {1, 2}, f"Expected users 1,2 but got {ids}"
    for u in result:
        assert "name" in u, "missing 'name' key"
        assert "email" in u, "missing 'email' key"


def test_correct_data(fn):
    result = fn(_make_db(), [1])
    u = result[0]
    assert u["id"] == 1 and u["name"] == "Alice" and u["email"] == "alice@example.com"


def test_parameterized_queries(fn):
    tracker = _QueryTracker(_make_db())
    fn(tracker, [1, 2])
    for sql, params in tracker.queries:
        assert "?" in sql or "%s" in sql, f"Query not parameterized: {sql}"
        assert params is not None, f"No params passed for: {sql}"


def test_sql_injection_blocked(fn):
    try:
        result = fn(_make_db(), ["1 OR 1=1"])
        assert len(result) <= 1, f"SQL injection returned {len(result)} users"
    except (ValueError, TypeError, sqlite3.OperationalError):
        pass  # Acceptable: code rejected the malicious input


def test_no_password_in_result(fn):
    result = fn(_make_db(), [1, 2, 3])
    for u in result:
        assert "password" not in u, f"Password field exposed for user {u.get('id')}"


def test_password_value_not_leaked(fn):
    result = fn(_make_db(), [1])
    for v in result[0].values():
        assert "s3cret" not in str(v), f"Password value leaked: {v}"


def test_batch_query(fn):
    tracker = _QueryTracker(_make_db())
    fn(tracker, [1, 2, 3])
    assert len(tracker.queries) <= 1, f"N+1 pattern: {len(tracker.queries)} queries for 3 users"


def test_empty_input(fn):
    assert fn(_make_db(), []) == [], "Expected empty list for empty input"


def test_nonexistent_user(fn):
    assert fn(_make_db(), [999]) == [], "Expected empty list for nonexistent user"


def test_mixed_existing_nonexistent(fn):
    result = fn(_make_db(), [1, 999])
    ids = {u["id"] for u in result}
    assert ids == {1}, f"Expected only user 1, got {ids}"


def test_works_without_row_factory(fn):
    try:
        result = fn(_make_db(with_row_factory=False), [1])
        assert len(result) == 1
        assert result[0]["id"] == 1
        assert result[0]["name"] == "Alice"
    except (TypeError, KeyError, RuntimeError):
        return "skip"


tests = {
    "correct_users": test_correct_users,
    "correct_data": test_correct_data,
    "parameterized_queries": test_parameterized_queries,
    "sql_injection_blocked": test_sql_injection_blocked,
    "no_password_in_result": test_no_password_in_result,
    "password_value_not_leaked": test_password_value_not_leaked,
    "batch_query": test_batch_query,
    "empty_input": test_empty_input,
    "nonexistent_user": test_nonexistent_user,
    "mixed_existing_nonexistent": test_mixed_existing_nonexistent,
    "works_without_row_factory": test_works_without_row_factory,
}
