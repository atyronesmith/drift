"""
Functional equivalence tests for the pod_log_triage task.

Tests split into two categories:
- EXTRACTABLE: fields that can be pulled directly from input (rule candidates)
- REASONING: fields that require LLM analysis (must stay as LLM calls)

Input: noisy pod logs (~167 lines) with 3 error clusters buried in
health checks, leader election, informer sync, and YAML noise.
"""

import re


def extract(code: str):
    """Parse triage sections from LLM output."""
    text = code.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    sections = {}
    current_section = None
    current_lines = []
    for line in text.split("\n"):
        if line.strip().startswith("### "):
            if current_section:
                sections[current_section] = "\n".join(current_lines)
            current_section = line.strip()[4:].strip()
            current_lines = []
        elif current_section:
            current_lines.append(line)
    if current_section:
        sections[current_section] = "\n".join(current_lines)
    return {"_raw": text, "_sections": sections}


# --- EXTRACTABLE tests (rule candidates) ---

def test_identifies_three_issues(data):
    """Extract: output contains sections for all 3 error clusters."""
    sections = data.get("_sections", {})
    assert len(sections) >= 3, \
        f"Expected 3+ issue sections, got {len(sections)}: {list(sections.keys())}"


def test_mentions_oom_killed(data):
    """Extract: identifies the OOMKilled event."""
    raw = data.get("_raw", "").lower()
    assert "oom" in raw, "Expected OOMKilled referenced in output"


def test_mentions_crashloopbackoff(data):
    """Extract: identifies the CrashLoopBackOff state."""
    raw = data.get("_raw", "").lower()
    assert "crash" in raw or "crashloop" in raw, \
        "Expected CrashLoopBackOff referenced in output"


def test_mentions_gpu_memory(data):
    """Extract: identifies GPU memory / CUDA OOM issue."""
    raw = data.get("_raw", "").lower()
    assert "gpu" in raw or "cuda" in raw, \
        "Expected GPU memory or CUDA issue referenced in output"


def test_mentions_node_not_ready(data):
    """Extract: identifies the node NotReady event."""
    raw = data.get("_raw", "").lower()
    assert "node" in raw and ("notready" in raw or "not ready" in raw), \
        "Expected node NotReady condition referenced in output"


def test_mentions_namespace_demo_ns(data):
    """Extract: references the demo-ns namespace."""
    raw = data.get("_raw", "")
    assert "demo-ns" in raw, "Expected namespace 'demo-ns' referenced in output"


def test_mentions_pod_name(data):
    """Extract: references the predictor pod name."""
    raw = data.get("_raw", "")
    assert "llama-3-1-8b-instruct-predictor" in raw or "kserve-container" in raw, \
        "Expected pod or container name in output"


def test_mentions_node_3(data):
    """Extract: references the affected node."""
    raw = data.get("_raw", "")
    assert "node-3" in raw, "Expected 'node-3' referenced in output"


# --- REASONING tests (LLM required) ---

def test_oom_severity_critical(data):
    """Reason: OOMKilled should be flagged as critical."""
    raw = data.get("_raw", "").lower()
    assert "critical" in raw, \
        "Expected OOMKilled flagged as critical severity"


def test_memory_limit_analysis(data):
    """Reason: explains that 16Gi limit was exceeded."""
    raw = data.get("_raw", "")
    assert "16" in raw, \
        "Expected 16Gi memory limit mentioned in analysis"


def test_gpu_fragmentation_analysis(data):
    """Reason: identifies GPU memory fragmentation as root cause."""
    raw = data.get("_raw", "").lower()
    assert any(term in raw for term in ["fragment", "contiguous", "allocat"]), \
        "Expected analysis of GPU memory fragmentation"


def test_eviction_consequence(data):
    """Reason: notes pod eviction as consequence of node failure."""
    raw = data.get("_raw", "").lower()
    assert "evict" in raw, \
        "Expected pod eviction mentioned as consequence of node NotReady"


def test_actionable_recommendations(data):
    """Reason: provides actionable remediation steps."""
    raw = data.get("_raw", "").lower()
    action_terms = ["increase", "check", "investigate", "restart", "scale",
                    "adjust", "review", "reduce", "examine", "inspect",
                    "request", "limit", "allocat"]
    assert any(term in raw for term in action_terms), \
        "Expected actionable recommendations in analysis"


tests = {
    # Extractable (rule candidates)
    "identifies_three_issues": test_identifies_three_issues,
    "mentions_oom_killed": test_mentions_oom_killed,
    "mentions_crashloopbackoff": test_mentions_crashloopbackoff,
    "mentions_gpu_memory": test_mentions_gpu_memory,
    "mentions_node_not_ready": test_mentions_node_not_ready,
    "mentions_namespace_demo_ns": test_mentions_namespace_demo_ns,
    "mentions_pod_name": test_mentions_pod_name,
    "mentions_node_3": test_mentions_node_3,
    # Reasoning (LLM required)
    "oom_severity_critical": test_oom_severity_critical,
    "memory_limit_analysis": test_memory_limit_analysis,
    "gpu_fragmentation_analysis": test_gpu_fragmentation_analysis,
    "eviction_consequence": test_eviction_consequence,
    "actionable_recommendations": test_actionable_recommendations,
}
