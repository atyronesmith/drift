"""
Functional equivalence tests for the metric_summary task.

Tests split into two categories:
- EXTRACTABLE: fields that can be pulled directly from input (rule candidates)
- REASONING: fields that require LLM analysis (must stay as LLM calls)

Based on the ai-observability-summarizer's generate_llm_summary flow:
question + Thanos metric data → structured JSON summary
"""

import json
import re


def extract(code: str):
    """Parse JSON from LLM output."""
    text = code.strip()
    # Remove markdown fencing if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    # Try to find JSON object in the text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


# --- EXTRACTABLE tests (rule candidates) ---

def test_has_current_status(data):
    """Extract: response contains a current_status field."""
    assert "current_status" in data, \
        f"Missing 'current_status' field. Got keys: {list(data.keys())}"


def test_has_requests_running(data):
    """Extract: response contains requests_running field."""
    assert "requests_running" in data, \
        f"Missing 'requests_running' field. Got keys: {list(data.keys())}"


def test_requests_running_value(data):
    """Extract: requests_running matches the input value of 3."""
    val = data.get("requests_running")
    assert val is not None and float(val) == 3.0, \
        f"Expected requests_running=3, got {val}"


def test_has_requests_waiting(data):
    """Extract: response contains requests_waiting field."""
    assert "requests_waiting" in data, \
        f"Missing 'requests_waiting' field. Got keys: {list(data.keys())}"


def test_requests_waiting_zero(data):
    """Extract: requests_waiting matches the input value of 0."""
    val = data.get("requests_waiting")
    assert val is not None and float(val) == 0.0, \
        f"Expected requests_waiting=0, got {val}"


def test_has_p95_latency(data):
    """Extract: response contains p95_latency_seconds field."""
    assert "p95_latency_seconds" in data, \
        f"Missing 'p95_latency_seconds' field. Got keys: {list(data.keys())}"


def test_p95_latency_value(data):
    """Extract: p95_latency matches the input value of ~2.847."""
    val = data.get("p95_latency_seconds")
    assert val is not None and 2.5 <= float(val) <= 3.5, \
        f"Expected p95_latency ~2.847, got {val}"


def test_has_gpu_utilization(data):
    """Extract: response contains gpu_utilization_percent field."""
    assert "gpu_utilization_percent" in data, \
        f"Missing 'gpu_utilization_percent' field. Got keys: {list(data.keys())}"


def test_gpu_utilization_value(data):
    """Extract: gpu_utilization matches the input value of ~67.4."""
    val = data.get("gpu_utilization_percent")
    assert val is not None and 60 <= float(val) <= 75, \
        f"Expected gpu_utilization ~67.4, got {val}"


def test_has_pod_count(data):
    """Extract: response contains pod_count field."""
    assert "pod_count" in data, \
        f"Missing 'pod_count' field. Got keys: {list(data.keys())}"


def test_pod_count_value(data):
    """Extract: pod_count matches the input value of 4."""
    val = data.get("pod_count")
    assert val is not None and float(val) == 4.0, \
        f"Expected pod_count=4, got {val}"


# --- REASONING tests (LLM required) ---

def test_identifies_healthy_status(data):
    """Reason: classifies overall status as healthy/normal."""
    status = str(data.get("current_status", "")).lower()
    assert status in ["healthy", "normal", "operational", "stable", "good"], \
        f"Expected healthy status classification, got '{status}'"


def test_concern_is_none_or_benign(data):
    """Reason: does not raise false alarms when metrics are normal."""
    concern = str(data.get("immediate_concern", "")).lower()
    # Should be None or benign — not a critical false alarm
    critical_terms = ["outage", "down", "crash", "failure", "emergency"]
    for term in critical_terms:
        assert term not in concern, \
            f"False alarm: flagged '{term}' when metrics are normal"


def test_meaning_mentions_service(data):
    """Reason: meaning references the inference service."""
    meaning = str(data.get("meaning", "")).lower()
    assert any(term in meaning for term in ["vllm", "inference", "service",
                                              "model", "llm"]), \
        f"Expected inference service mentioned in meaning"


tests = {
    # Extractable (rule candidates)
    "has_current_status": test_has_current_status,
    "has_requests_running": test_has_requests_running,
    "requests_running_value": test_requests_running_value,
    "has_requests_waiting": test_has_requests_waiting,
    "requests_waiting_zero": test_requests_waiting_zero,
    "has_p95_latency": test_has_p95_latency,
    "p95_latency_value": test_p95_latency_value,
    "has_gpu_utilization": test_has_gpu_utilization,
    "gpu_utilization_value": test_gpu_utilization_value,
    "has_pod_count": test_has_pod_count,
    "pod_count_value": test_pod_count_value,
    # Reasoning (LLM required)
    "identifies_healthy_status": test_identifies_healthy_status,
    "concern_is_none_or_benign": test_concern_is_none_or_benign,
    "meaning_mentions_service": test_meaning_mentions_service,
}
