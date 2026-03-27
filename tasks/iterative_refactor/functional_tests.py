"""
Functional equivalence tests for the iterative_refactor task.

Tests the implementer step's refactored UserService against a test
SQLite database. Validates the 4 core security fixes:
1. No hardcoded password
2. Parameterized queries (no SQL injection)
3. Password excluded from authenticate response
4. Specific exception handling (no bare except)
"""

import re
import sqlite3
import types
import os


def extract(code: str):
    """Extract the UserService class from LLM output.

    Returns the class with a _source attribute containing the raw code.
    """
    # Strip markdown fences
    match = re.search(r"```python\n(.*?)```", code, re.DOTALL)
    if match:
        code = match.group(1)

    module = types.ModuleType("variant")
    module.__dict__["sqlite3"] = sqlite3
    module.__dict__["os"] = os
    exec(compile(code, "<variant>", "exec"), module.__dict__)

    cls = module.__dict__["UserService"]
    cls._source = code
    return cls


# --- Test fixtures ---

def _make_db():
    """Create a test database with users and orders tables."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, "
        "email TEXT, password TEXT, role TEXT)"
    )
    conn.executemany(
        "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
        [
            (1, "alice", "alice@example.com", "pass123", "admin"),
            (2, "bob", "bob@example.com", "pass456", "user"),
            (3, "charlie", "charlie@example.com", "pass789", "user"),
        ],
    )
    conn.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, item TEXT)"
    )
    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?)",
        [(1, 1, "widget"), (2, 1, "gadget"), (3, 2, "thing")],
    )
    conn.commit()
    return conn


class _ServiceHelper:
    def __init__(self, cls):
        self.cls = cls

    def make(self, db=None):
        if db is None:
            db = _make_db()
        svc = object.__new__(self.cls)
        svc.conn = db
        return svc, db


def _run_with_class(cls):
    helper = _ServiceHelper(cls)
    return helper


# --- Tests ---

def test_authenticate_returns_user(cls):
    """Basic: authenticate with correct credentials returns user dict."""
    h = _run_with_class(cls)
    svc, db = h.make()
    result = svc.authenticate("alice", "pass123")
    assert result is not None, "authenticate returned None for valid credentials"
    assert result.get("username") == "alice", f"Expected username 'alice', got {result.get('username')}"


def test_no_password_in_response(cls):
    """Security fix: authenticate must not return password or db_password."""
    h = _run_with_class(cls)
    svc, db = h.make()
    result = svc.authenticate("alice", "pass123")
    assert result is not None, "authenticate returned None"
    assert "password" not in result, f"Password field exposed: {list(result.keys())}"
    assert "db_password" not in result, f"db_password field exposed: {list(result.keys())}"
    for v in result.values():
        assert "pass123" not in str(v), f"Password value leaked: {v}"
        assert "admin123" not in str(v), f"Hardcoded password leaked: {v}"


def test_no_hardcoded_password(cls):
    """Security fix: source code must not contain hardcoded password."""
    assert "admin123" not in cls._source, "Hardcoded password 'admin123' still in source"


def test_parameterized_authenticate(cls):
    """Security fix: authenticate uses parameterized queries."""
    h = _run_with_class(cls)
    svc, db = h.make()
    # SQL injection attempt — should return None, not all users
    result = svc.authenticate("' OR 1=1 --", "anything")
    assert result is None, f"SQL injection succeeded in authenticate: {result}"


def test_parameterized_update_email(cls):
    """Security fix: update_email uses parameterized queries."""
    h = _run_with_class(cls)
    svc, db = h.make()
    # Normal operation
    svc.update_email(1, "new@example.com")
    row = db.execute("SELECT email FROM users WHERE id = 1").fetchone()
    assert row[0] == "new@example.com", f"Email not updated: {row[0]}"
    # SQL injection attempt
    svc.update_email(2, "'; DROP TABLE users; --")
    # Table should still exist
    count = db.execute("SELECT count(*) FROM users").fetchone()[0]
    assert count >= 2, f"SQL injection in update_email: only {count} users remain"


def test_parameterized_search(cls):
    """Security fix: search_users uses parameterized queries."""
    h = _run_with_class(cls)
    svc, db = h.make()
    # Normal search
    results = svc.search_users("ali")
    assert len(results) == 1, f"Expected 1 result for 'ali', got {len(results)}"
    # SQL injection attempt
    try:
        results = svc.search_users("' OR 1=1 --")
        # Should return 0 or at most match literally
        assert len(results) <= 1, f"SQL injection in search_users returned {len(results)} results"
    except (sqlite3.OperationalError, ValueError):
        pass  # Acceptable: rejected malicious input


def test_parameterized_delete(cls):
    """Security fix: delete_user uses parameterized queries."""
    h = _run_with_class(cls)
    svc, db = h.make()
    svc.delete_user(3)
    row = db.execute("SELECT count(*) FROM users WHERE id = 3").fetchone()
    assert row[0] == 0, f"User 3 not deleted"


def test_specific_exception_handling(cls):
    """Security fix: delete_user does not use bare except."""
    # Check there's no bare "except:" in the source
    assert not re.search(r"except\s*:", cls._source), "Bare 'except:' found in delete_user"


def test_authenticate_wrong_password(cls):
    """Basic: authenticate with wrong password returns None."""
    h = _run_with_class(cls)
    svc, db = h.make()
    result = svc.authenticate("alice", "wrongpassword")
    assert result is None, f"authenticate returned non-None for wrong password: {result}"


def test_search_no_results(cls):
    """Basic: search for nonexistent user returns empty list."""
    h = _run_with_class(cls)
    svc, db = h.make()
    results = svc.search_users("zzzznonexistent")
    assert len(results) == 0, f"Expected 0 results, got {len(results)}"


tests = {
    "authenticate_returns_user": test_authenticate_returns_user,
    "no_password_in_response": test_no_password_in_response,
    "no_hardcoded_password": test_no_hardcoded_password,
    "parameterized_authenticate": test_parameterized_authenticate,
    "parameterized_update_email": test_parameterized_update_email,
    "parameterized_search": test_parameterized_search,
    "parameterized_delete": test_parameterized_delete,
    "specific_exception_handling": test_specific_exception_handling,
    "authenticate_wrong_password": test_authenticate_wrong_password,
    "search_no_results": test_search_no_results,
}
