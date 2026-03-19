Analyze this Python code and identify all issues. For each issue, assign a unique ID (ISS-001, ISS-002, etc.), severity (critical, high, medium, low), the affected line number, a category (security, performance, reliability, style), a one-sentence description, and a one-sentence fix strategy.

Return JSON with this exact structure:
{
  "iteration": 1,
  "issues": [{"id": "ISS-001", "severity": "critical", "line": 14, "category": "security", "description": "...", "fix_strategy": "..."}],
  "issue_count": {"critical": 0, "high": 0, "medium": 0, "low": 0},
  "priority_fix_order": ["ISS-001", "ISS-002"]
}

{input}
