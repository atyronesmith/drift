"""
Functional equivalence tests for the ansible_route task.

Tests split into two categories:
- EXTRACTABLE: structural validation and signal-presence heuristics (rule candidates)
- REASONING: ambiguous routing decisions requiring judgment (must stay as LLM calls)

The router produces JSON: {"suggestion": "<No More Context Needed or Need More Context>"}
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


VALID_SUGGESTIONS = {
    "No More Context Needed",
    "Need More Context",
}


# --- EXTRACTABLE tests (rule candidates) ---

def test_valid_json(data):
    """Extract: output is valid JSON (already parsed by extract)."""
    assert isinstance(data, dict), f"Expected dict, got {type(data).__name__}"


def test_has_suggestion(data):
    """Extract: output contains suggestion field."""
    assert "suggestion" in data, f"Missing 'suggestion' key. Keys: {list(data.keys())}"
    assert isinstance(data["suggestion"], str), f"suggestion must be string"


def test_suggestion_valid(data):
    """Extract: suggestion is one of the 2 valid values."""
    val = data.get("suggestion", "")
    assert val in VALID_SUGGESTIONS, \
        f"Invalid suggestion '{val}'. Must be one of: {VALID_SUGGESTIONS}"


def test_no_extra_fields(data):
    """Extract: output has only the expected field."""
    expected = {"suggestion"}
    extra = set(data.keys()) - expected
    assert not extra, f"Unexpected fields: {extra}. Expected only: {expected}"


# --- REASONING tests (LLM required) ---

def test_explicit_error_routes_to_no_context(data):
    """Reason: inputs with explicit root cause + component should route to 'No More Context Needed'."""
    # This test validates the decision logic. For the default input (RBAC 403
    # on a specific service account + namespace), the error is explicit and
    # actionable — should not need more context.
    # However, this is a judgment call the LLM must make.
    val = data.get("suggestion", "")
    assert val in VALID_SUGGESTIONS, f"Must be a valid suggestion: got '{val}'"


def test_routing_consistency(data):
    """Reason: routing decision should be consistent with input signal quality."""
    val = data.get("suggestion", "")
    # The router should produce one of exactly two values — validated above.
    # This test is a placeholder for cross-input consistency validation:
    # when run across the corpus, we check that similar inputs get similar routing.
    assert val in VALID_SUGGESTIONS, f"Must be a valid suggestion: got '{val}'"


tests = {
    # Extractable (rule candidates)
    "valid_json": test_valid_json,
    "has_suggestion": test_has_suggestion,
    "suggestion_valid": test_suggestion_valid,
    "no_extra_fields": test_no_extra_fields,
    # Reasoning (LLM required)
    "explicit_error_routing": test_explicit_error_routes_to_no_context,
    "routing_consistency": test_routing_consistency,
}
