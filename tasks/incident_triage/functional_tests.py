"""
Functional equivalence tests for the incident_triage task.

Tests split into two categories:
- EXTRACTABLE: fields that can be pulled directly from input text (rule candidates)
- REASONING: fields that require LLM analysis (must stay as LLM calls)

This split lets us measure the hybrid boundary: which tests pass with
rules alone vs. which require LLM reasoning.
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


# --- Helper ---

def _deep_get(obj, path):
    """Get a value from a nested dict using dot notation."""
    for key in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return None
    return obj


def _find_field(data, candidates):
    """Find a value trying multiple possible field paths."""
    for path in candidates:
        val = _deep_get(data, path)
        if val is not None:
            return val
    return None


# --- EXTRACTABLE tests (rule candidates) ---

def test_incident_id(data):
    """Extract: incident ID from report header."""
    text = json.dumps(data)
    assert "INC-2026-04821" in text or "04821" in text or "INC-2026" in text, \
        f"Expected incident ID (INC-2026-04821) somewhere in output"


def test_reporter(data):
    """Extract: reporter name from report."""
    text = json.dumps(data)
    assert "Sarah Chen" in text, f"Expected Sarah Chen somewhere in output"


def test_account(data):
    """Extract: customer account info."""
    text = json.dumps(data)
    assert "Cascade" in text, f"Expected Cascade Manufacturing in output"


def test_product(data):
    """Extract: affected product."""
    text = json.dumps(data)
    assert "FlowControl" in text, f"Expected FlowControl Pro in output"


def test_environment(data):
    """Extract: environment and region."""
    text = json.dumps(data).lower()
    assert "prod" in text and "us-west" in text, f"Expected production/us-west-2 in output"


def test_error_codes(data):
    """Extract: specific error indicators from logs."""
    text = json.dumps(data).lower()
    assert "buffer pool" in text or "gc pause" in text or "heap" in text, \
        "Expected memory/GC-related error indicators in output"


def test_sla(data):
    """Extract: SLA terms from report header."""
    text = json.dumps(data)
    assert "4" in text, "Expected 4-hour response SLA mentioned"


def test_timeline(data):
    """Extract: key timestamps from report."""
    text = json.dumps(data)
    assert "13:45" in text or "1345" in text, "Expected incident start time 13:45"


# --- REASONING tests (LLM required) ---

def test_severity_classification(data):
    """Reason: correct severity based on revenue impact."""
    val = _find_field(data, [
        "severity", "priority", "classification.severity",
        "classification.priority", "triage.severity", "triage.priority",
    ])
    assert val is not None, "Missing severity classification"
    sev = str(val).upper()
    # $340K batch at risk + SLA customer = P1 or Critical
    assert "P1" in sev or "CRITICAL" in sev or "1" in sev, \
        f"Expected P1/Critical severity (revenue impact: $340K batch), got {val}"


def test_root_cause_analysis(data):
    """Reason: identifies memory/GC as root cause from log evidence."""
    text = json.dumps(data).lower()
    # Must mention memory-related root cause somewhere
    has_root_cause = any(term in text for term in ["root_cause", "root cause", "diagnosis", "cause"])
    assert has_root_cause, "Missing root cause analysis section"
    assert any(term in text for term in ["memory", "heap", "gc", "garbage", "jvm", "buffer"]), \
        f"Expected memory/heap/GC identified as root cause"


def test_recommends_heap_increase(data):
    """Reason: recommends JVM heap increase based on prior incident pattern."""
    text = json.dumps(data).lower()
    assert any(term in text for term in ["heap", "jvm", "memory", "8gb", "16gb"]), \
        f"Expected recommendation to increase heap/memory"


def test_identifies_batch_risk(data):
    """Reason: flags the batch certification deadline risk."""
    text = json.dumps(data).lower()
    assert any(term in text for term in ["batch", "qc-7814", "340", "certif", "scrap", "quality"]), \
        f"Expected batch risk / QC-7814 / $340K identified"


def test_recommends_immediate_action(data):
    """Reason: provides actionable immediate steps."""
    val = _find_field(data, [
        "recommended_actions", "actions", "recommendations",
        "immediate_actions", "triage.actions",
    ])
    assert val is not None, "Missing recommended actions"
    text = json.dumps(val).lower()
    # Should recommend something actionable, not just "investigate"
    assert any(term in text for term in ["heap", "restart", "memory", "jvm", "increase", "scale"]), \
        f"Expected actionable recommendation (heap increase, restart, etc.), got brief: {str(val)[:100]}"


tests = {
    # Extractable (rule candidates)
    "incident_id": test_incident_id,
    "reporter": test_reporter,
    "account": test_account,
    "product": test_product,
    "environment": test_environment,
    "error_codes": test_error_codes,
    "sla": test_sla,
    "timeline": test_timeline,
    # Reasoning (LLM required)
    "severity_classification": test_severity_classification,
    "root_cause_analysis": test_root_cause_analysis,
    "recommends_heap_increase": test_recommends_heap_increase,
    "identifies_batch_risk": test_identifies_batch_risk,
    "recommends_immediate_action": test_recommends_immediate_action,
}
