Analyze the current state of this code. Focus on remaining unfixed issues and any new issues introduced by recent changes. Assign new IDs to newly discovered issues (NEW-001, NEW-002, etc.) and retain original IDs for issues that persist.

Return JSON with this exact structure:
{
  "iteration": 1,
  "issues": [{"id": "ISS-001", "severity": "critical", "line": 14, "category": "security", "description": "...", "fix_strategy": "..."}],
  "issue_count": {"critical": 0, "high": 0, "medium": 0, "low": 0},
  "priority_fix_order": ["ISS-001", "ISS-002"]
}

Current code:
{loop_step_implementer}

Verifier feedback:
{loop_step_verifier}
