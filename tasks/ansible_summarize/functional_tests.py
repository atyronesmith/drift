"""
Functional equivalence tests for the ansible_summarize task.

Tests split into two categories:
- EXTRACTABLE: fields that can be pulled directly from log text (rule candidates)
- REASONING: fields that require LLM analysis (must stay as LLM calls)

The summarizer produces JSON: {"error_type": "...", "description": "..."}
"""

import json
import re


def extract(code: str):
    """Parse JSON from LLM output."""
    text = code.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)


# Known error types from the upstream prompt examples and corpus analysis
KNOWN_ERROR_TYPES = {
    "variable error", "ssh error", "aws api error", "repository error",
    "dns error", "certificate error", "permission error", "resource limit error",
    "timeout error", "package error", "configuration error", "kubernetes error",
    "network error", "authentication error", "empty log",
}


# --- EXTRACTABLE tests (rule candidates) ---

def test_valid_json(data):
    """Extract: output is valid JSON (already parsed by extract)."""
    assert isinstance(data, dict), f"Expected dict, got {type(data).__name__}"


def test_has_error_type(data):
    """Extract: output contains error_type field."""
    assert "error_type" in data, f"Missing 'error_type' key. Keys: {list(data.keys())}"
    assert isinstance(data["error_type"], str), f"error_type must be string"
    assert len(data["error_type"]) > 0, "error_type must not be empty"


def test_has_description(data):
    """Extract: output contains description field."""
    assert "description" in data, f"Missing 'description' key. Keys: {list(data.keys())}"
    assert isinstance(data["description"], str), f"description must be string"
    assert len(data["description"]) > 0, "description must not be empty"


def test_error_type_known(data):
    """Extract: error_type is from the known set."""
    error_type = data.get("error_type", "").lower().strip()
    assert error_type in KNOWN_ERROR_TYPES, \
        f"Unknown error_type '{data.get('error_type')}'. Expected one of: {sorted(KNOWN_ERROR_TYPES)}"


def test_description_length(data):
    """Extract: description is concise (1-3 sentences, not a dump of the log)."""
    desc = data.get("description", "")
    # Should be at least 10 chars and at most ~500 chars
    assert len(desc) >= 10, f"Description too short ({len(desc)} chars): '{desc}'"
    assert len(desc) <= 500, f"Description too long ({len(desc)} chars) — should be a brief summary, not a log dump"


def test_no_extra_fields(data):
    """Extract: output has only the expected fields."""
    expected = {"error_type", "description"}
    extra = set(data.keys()) - expected
    assert not extra, f"Unexpected fields: {extra}. Expected only: {expected}"


# --- REASONING tests (LLM required) ---

def test_description_references_input(data):
    """Reason: description should reference specifics from the input log."""
    desc = data.get("description", "").lower()
    # Must contain some specific detail, not just a generic "an error occurred"
    generic_only = all(
        term not in desc
        for term in [
            "forbidden", "403", "permission", "rbac", "showroom",
            "serviceaccount", "project", "namespace", "openshift",
            # Also accept terms from other log types
            "variable", "undefined", "ssh", "aws", "ec2",
            "dns", "certificate", "timeout", "package", "repository",
            "subscription", "cluster", "operator", "argocd",
        ]
    )
    assert not generic_only, \
        f"Description appears too generic — should reference specific error details from input"


def test_error_type_matches_content(data):
    """Reason: error_type should match the actual error in the log."""
    # For the default input (job_1462656.txt), the error is a 403 Forbidden
    # on an OpenShift project — should be Permission Error or Kubernetes Error
    error_type = data.get("error_type", "").lower()
    desc = data.get("description", "").lower()

    # Check consistency: if description mentions permission/forbidden/RBAC,
    # error_type should reflect that
    if any(t in desc for t in ["forbidden", "403", "rbac", "cannot get resource"]):
        assert any(t in error_type for t in ["permission", "kubernetes", "authentication"]), \
            f"Description mentions permission/RBAC issue but error_type is '{data.get('error_type')}'"

    # If description mentions variable/undefined, error_type should match
    if any(t in desc for t in ["undefined variable", "undefined"]):
        assert "variable" in error_type, \
            f"Description mentions undefined variable but error_type is '{data.get('error_type')}'"


def test_captures_host_or_component(data):
    """Reason: description should identify the affected host or component when present."""
    desc = data.get("description", "").lower()
    # A good summary identifies what failed, not just that something failed.
    # At minimum, should mention a role, host, task, component, or service.
    has_specific = any(
        term in desc
        for term in [
            "localhost", "bastion", "showroom", "serviceaccount",
            "node", "host", "role", "task", "playbook",
            # Terms from other common logs
            "ec2", "instance", "cluster", "pod", "operator",
            "subscription", "repository", "dns", "certificate",
        ]
    )
    assert has_specific, \
        f"Description should identify the affected component/host/service"


def test_root_cause_identified(data):
    """Reason: description should identify the root cause, not just symptoms."""
    desc = data.get("description", "").lower()
    # Should not just say "task failed" — should explain why
    assert "failed" not in desc or len(desc) > 30, \
        f"Description should explain the root cause, not just state 'failed'"


tests = {
    # Extractable (rule candidates)
    "valid_json": test_valid_json,
    "has_error_type": test_has_error_type,
    "has_description": test_has_description,
    "error_type_known": test_error_type_known,
    "description_length": test_description_length,
    "no_extra_fields": test_no_extra_fields,
    # Reasoning (LLM required)
    "description_references_input": test_description_references_input,
    "error_type_matches_content": test_error_type_matches_content,
    "captures_host_or_component": test_captures_host_or_component,
    "root_cause_identified": test_root_cause_identified,
}
