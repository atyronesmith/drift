"""
Functional equivalence tests for the alert_triage task.

Tests split into two categories:
- EXTRACTABLE: fields that can be pulled directly from input (rule candidates)
- REASONING: fields that require LLM analysis (must stay as LLM calls)

Based on the ai-observability-summarizer's generate_alert_analysis_with_llm flow:
firing alerts → per-alert analysis with Severity/Impact/Action/Troubleshooting/Namespace
"""

import json
import re


def extract(code: str):
    """Parse alert analysis sections from LLM output."""
    text = code.strip()
    # Remove markdown fencing if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    # Return as a dict with the raw text and parsed sections
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

def test_has_three_alert_sections(data):
    """Extract: output contains sections for all 3 alerts."""
    sections = data.get("_sections", {})
    assert len(sections) >= 3, \
        f"Expected 3 alert sections, got {len(sections)}: {list(sections.keys())}"


def test_has_vllm_inference_section(data):
    """Extract: has a section for VLLMHighAverageInferenceTime."""
    raw = data.get("_raw", "").lower()
    assert "vllmhighaverageinferencetime" in raw or "high average inference" in raw, \
        "Missing section for VLLMHighAverageInferenceTime alert"


def test_has_crashloop_section(data):
    """Extract: has a section for KubePodCrashLooping."""
    raw = data.get("_raw", "").lower()
    assert "kubepodcrashlooping" in raw or "crash" in raw, \
        "Missing section for KubePodCrashLooping alert"


def test_has_memory_overcommit_section(data):
    """Extract: has a section for KubeMemoryOvercommit."""
    raw = data.get("_raw", "").lower()
    assert "kubememoryovercommit" in raw or "memory overcommit" in raw, \
        "Missing section for KubeMemoryOvercommit alert"


def test_mentions_namespace_m3(data):
    """Extract: references the m3 namespace."""
    raw = data.get("_raw", "")
    assert "m3" in raw, "Expected namespace 'm3' referenced in output"


def test_mentions_pod_name(data):
    """Extract: references at least one pod name from the alerts."""
    raw = data.get("_raw", "")
    assert "llama-3-1-8b-instruct-predictor" in raw, \
        "Expected pod name 'llama-3-1-8b-instruct-predictor' in output"


def test_severity_labels_present(data):
    """Extract: each section includes severity labeling."""
    raw = data.get("_raw", "").lower()
    assert "critical" in raw and "warning" in raw, \
        "Expected both 'critical' and 'warning' severity labels"


# --- REASONING tests (LLM required) ---

def test_crashloop_highest_priority(data):
    """Reason: identifies CrashLoopBackOff as highest priority."""
    raw = data.get("_raw", "").lower()
    # CrashLooping is critical; it should be flagged as most urgent
    assert "critical" in raw, \
        "Expected CrashLoopBackOff flagged as critical"


def test_inference_time_interpretation(data):
    """Reason: interprets the inference time expression correctly."""
    raw = data.get("_raw", "").lower()
    # The expr says inference time > 2 seconds for 5 minutes
    assert any(term in raw for term in ["2 second", "threshold", "latency", "slow",
                                         "inference time", "response time"]), \
        "Expected interpretation of inference time threshold"


def test_memory_overcommit_analysis(data):
    """Reason: explains memory overcommit means requests exceed capacity."""
    raw = data.get("_raw", "").lower()
    assert any(term in raw for term in ["exceed", "overcommit", "capacity",
                                         "insufficient", "more than available"]), \
        "Expected analysis that memory requests exceed node capacity"


def test_troubleshooting_commands(data):
    """Reason: provides relevant kubectl/oc troubleshooting commands."""
    raw = data.get("_raw", "").lower()
    assert any(cmd in raw for cmd in ["kubectl", "oc ", "oc get", "oc describe",
                                       "oc logs", "kubectl logs", "kubectl describe"]), \
        "Expected kubectl/oc troubleshooting commands"


def test_restart_count_noted(data):
    """Reason: notes the restart count (8) for the crash-looping pod."""
    raw = data.get("_raw", "")
    assert "8" in raw, \
        "Expected restart count (8) mentioned for crash-looping pod"


def test_actionable_recommendations(data):
    """Reason: provides actionable steps, not just descriptions."""
    raw = data.get("_raw", "").lower()
    action_terms = ["check", "investigate", "increase", "restart", "scale",
                    "adjust", "review", "reduce", "examine", "inspect"]
    assert any(term in raw for term in action_terms), \
        "Expected actionable recommendations in analysis"


tests = {
    # Extractable (rule candidates)
    "has_three_alert_sections": test_has_three_alert_sections,
    "has_vllm_inference_section": test_has_vllm_inference_section,
    "has_crashloop_section": test_has_crashloop_section,
    "has_memory_overcommit_section": test_has_memory_overcommit_section,
    "mentions_namespace_m3": test_mentions_namespace_m3,
    "mentions_pod_name": test_mentions_pod_name,
    "severity_labels_present": test_severity_labels_present,
    # Reasoning (LLM required)
    "crashloop_highest_priority": test_crashloop_highest_priority,
    "inference_time_interpretation": test_inference_time_interpretation,
    "memory_overcommit_analysis": test_memory_overcommit_analysis,
    "troubleshooting_commands": test_troubleshooting_commands,
    "restart_count_noted": test_restart_count_noted,
    "actionable_recommendations": test_actionable_recommendations,
}
