"""
Functional equivalence tests for the promql_generation task.

Tests split into two categories:
- EXTRACTABLE: fields that can be pulled directly from input (rule candidates)
- REASONING: fields that require LLM analysis (must stay as LLM calls)

Based on the ai-observability-summarizer's build_openshift_chat_prompt flow:
user question + metrics context → {"promql": "...", "summary": "..."}
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

def test_has_promql_field(data):
    """Extract: response contains a promql field."""
    assert "promql" in data, "Missing 'promql' field in response"


def test_has_summary_field(data):
    """Extract: response contains a summary field."""
    assert "summary" in data, "Missing 'summary' field in response"


def test_promql_contains_cpu_metric(data):
    """Extract: PromQL references a CPU-related metric."""
    promql = data.get("promql", "")
    cpu_metrics = ["node_cpu_seconds_total", "cpu_usage", "container_cpu"]
    assert any(m in promql for m in cpu_metrics), \
        f"Expected CPU metric in PromQL, got: {promql}"


def test_promql_has_rate_function(data):
    """Extract: PromQL uses rate() for counter metrics."""
    promql = data.get("promql", "")
    assert "rate(" in promql, \
        f"Expected rate() function in PromQL for CPU counter metric, got: {promql}"


def test_promql_has_time_range(data):
    """Extract: PromQL includes a time range bracket."""
    promql = data.get("promql", "")
    assert re.search(r'\[\d+[smhd]\]', promql), \
        f"Expected time range (e.g., [2h], [5m]) in PromQL, got: {promql}"


def test_promql_valid_syntax(data):
    """Extract: PromQL has balanced brackets and parentheses."""
    promql = data.get("promql", "")
    assert promql.count('(') == promql.count(')'), \
        f"Unbalanced parentheses in PromQL: {promql}"
    assert promql.count('{') == promql.count('}'), \
        f"Unbalanced braces in PromQL: {promql}"
    assert promql.count('[') == promql.count(']'), \
        f"Unbalanced brackets in PromQL: {promql}"


# --- REASONING tests (LLM required) ---

def test_identifies_node3_hot(data):
    """Reason: identifies node-3 as running hot."""
    summary = data.get("summary", "")
    assert "node-3" in summary or "node 3" in summary.lower(), \
        f"Expected node-3 identified as problematic in summary"


def test_quantifies_cpu_usage(data):
    """Reason: quantifies CPU usage with specific values."""
    summary = data.get("summary", "").lower()
    # Accept percentages, idle fractions, or core counts
    has_numbers = re.search(r'\d+\.?\d*', summary)
    has_cpu_context = any(term in summary for term in [
        "%", "percent", "idle", "cores", "utilization", "usage", "0.5", "0.6", "0.7",
        "47", "49", "12.4",
    ])
    assert has_numbers and has_cpu_context, \
        f"Expected quantified CPU usage in summary"


def test_mentions_eviction_or_throttling(data):
    """Reason: connects log evidence (eviction/throttling) to CPU pressure."""
    summary = data.get("summary", "").lower()
    assert any(term in summary for term in ["evict", "throttl", "pressure", "reclaim"]), \
        f"Expected eviction/throttling/pressure mentioned from log correlation"


def test_mentions_vllm_workload(data):
    """Reason: identifies vllm as top CPU consumer."""
    summary = data.get("summary", "").lower()
    assert "vllm" in summary, \
        f"Expected vllm identified as top CPU consumer"


def test_provides_recommendation(data):
    """Reason: includes actionable recommendation."""
    summary = data.get("summary", "").lower()
    action_terms = ["recommend", "consider", "should", "suggest", "action",
                    "scale", "increase", "move", "redistribute", "rebalance"]
    assert any(term in summary for term in action_terms), \
        f"Expected actionable recommendation in summary"


tests = {
    # Extractable (rule candidates)
    "has_promql_field": test_has_promql_field,
    "has_summary_field": test_has_summary_field,
    "promql_contains_cpu_metric": test_promql_contains_cpu_metric,
    "promql_has_rate_function": test_promql_has_rate_function,
    "promql_has_time_range": test_promql_has_time_range,
    "promql_valid_syntax": test_promql_valid_syntax,
    # Reasoning (LLM required)
    "identifies_node3_hot": test_identifies_node3_hot,
    "quantifies_cpu_usage": test_quantifies_cpu_usage,
    "mentions_eviction_or_throttling": test_mentions_eviction_or_throttling,
    "mentions_vllm_workload": test_mentions_vllm_workload,
    "provides_recommendation": test_provides_recommendation,
}
