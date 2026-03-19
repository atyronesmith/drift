Verify whether the revised code fixes the issues listed in the analysis. Check for any new issues introduced by the changes.

Return JSON with this exact structure:
{
  "iteration": 1,
  "status": "pass or fail",
  "fixed_issues": ["ISS-001"],
  "remaining_issues": [{"id": "ISS-002", "severity": "high", "description": "..."}],
  "new_issues": [{"id": "NEW-001", "severity": "medium", "description": "..."}],
  "remaining_critical_high": 0,
  "recommendation": "pass or iterate or abort",
  "summary": "one sentence summary"
}

Set recommendation to "pass" if no critical or high severity issues remain. Set to "iterate" if critical or high issues remain. Set to "abort" only if the code is worse than before.

Original code:
{input}

Revised code:
{loop_step_implementer}

Issues that were supposed to be fixed:
{loop_input}
