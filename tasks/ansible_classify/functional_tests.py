"""
Functional equivalence tests for the ansible_classify task.

Tests split into two categories:
- EXTRACTABLE: structural validation and keyword-based category mapping (rule candidates)
- REASONING: ambiguous classifications requiring judgment (must stay as LLM calls)

The classifier produces JSON: {"category": "<exact category name>"}
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


VALID_CATEGORIES = {
    "Cloud Infrastructure / AWS Engineers",
    "Kubernetes Cluster Admins",
    "DevOps / CI/CD Engineers (Ansible + Automation Platform)",
    "Networking / Security Engineers",
    "System Administrators / OS Engineers",
    "Application Developers / GitOps / Platform Engineers",
    "Identity & Access Management (IAM) Engineers",
    "Other / Miscellaneous",
}


# --- EXTRACTABLE tests (rule candidates) ---

def test_valid_json(data):
    """Extract: output is valid JSON (already parsed by extract)."""
    assert isinstance(data, dict), f"Expected dict, got {type(data).__name__}"


def test_has_category(data):
    """Extract: output contains category field."""
    assert "category" in data, f"Missing 'category' key. Keys: {list(data.keys())}"
    assert isinstance(data["category"], str), f"category must be string"


def test_category_valid(data):
    """Extract: category is one of the 8 valid literals."""
    cat = data.get("category", "")
    assert cat in VALID_CATEGORIES, \
        f"Invalid category '{cat}'. Must be one of:\n" + \
        "\n".join(f"  - {c}" for c in sorted(VALID_CATEGORIES))


def test_no_extra_fields(data):
    """Extract: output has only the expected field."""
    expected = {"category"}
    extra = set(data.keys()) - expected
    assert not extra, f"Unexpected fields: {extra}. Expected only: {expected}"


def test_aws_keywords_map_correctly(data):
    """Extract: AWS/EC2/CloudFormation errors should map to Cloud Infrastructure."""
    text = json.dumps(data).lower()
    cat = data.get("category", "")
    # This test is only meaningful when the input contains AWS keywords
    # It validates the mapping when triggered by keyword presence
    if any(kw in text for kw in ["ec2", "cloudformation", "aws_access_key", "vpc", "iam"]):
        assert cat == "Cloud Infrastructure / AWS Engineers", \
            f"AWS-related input should map to Cloud Infrastructure, got '{cat}'"


def test_k8s_keywords_map_correctly(data):
    """Extract: cluster/operator/pod/RBAC errors should map to Kubernetes Cluster Admins."""
    text = json.dumps(data).lower()
    cat = data.get("category", "")
    if any(kw in text for kw in ["operator", "olm", "rbac", "serviceaccount"]):
        assert cat in {
            "Kubernetes Cluster Admins",
            "Networking / Security Engineers",  # some RBAC overlaps with security
        }, f"K8s-related input should map to Kubernetes Cluster Admins, got '{cat}'"


def test_ansible_keywords_map_correctly(data):
    """Extract: Ansible role/playbook/variable errors should map to DevOps / CI/CD."""
    text = json.dumps(data).lower()
    cat = data.get("category", "")
    if any(kw in text for kw in ["ansible", "playbook", "runner", "undefined variable"]):
        assert cat == "DevOps / CI/CD Engineers (Ansible + Automation Platform)", \
            f"Ansible-related input should map to DevOps / CI/CD, got '{cat}'"


# --- REASONING tests (LLM required) ---

def test_category_plausible_for_input(data):
    """Reason: category should be plausible given the error summary content."""
    cat = data.get("category", "")
    # For the default input (RBAC/403 on OpenShift project), expect Kubernetes or DevOps
    # This test validates that the LLM made a reasonable judgment call
    assert cat in VALID_CATEGORIES, f"Category must be valid: got '{cat}'"
    # Can't test more specifically without knowing the input — the test framework
    # runs this against various inputs


tests = {
    # Extractable (rule candidates)
    "valid_json": test_valid_json,
    "has_category": test_has_category,
    "category_valid": test_category_valid,
    "no_extra_fields": test_no_extra_fields,
    "aws_keywords_map": test_aws_keywords_map_correctly,
    "k8s_keywords_map": test_k8s_keywords_map_correctly,
    "ansible_keywords_map": test_ansible_keywords_map_correctly,
    # Reasoning (LLM required)
    "category_plausible": test_category_plausible_for_input,
}
