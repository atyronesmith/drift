"""Rule types for hybrid execution: extract deterministic fields from input."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RuleResult:
    path: str
    value: str | int | float | list | None
    matched: bool


def execute_rule(rule: dict, input_text: str) -> RuleResult:
    """Execute a single rule against input text."""
    path = rule["path"]
    rule_type = rule["type"]

    if rule_type == "regex":
        return _regex_rule(path, rule["pattern"], input_text, rule.get("group", 1))
    elif rule_type == "literal":
        return RuleResult(path=path, value=rule["value"], matched=True)
    elif rule_type == "regex_all":
        return _regex_all_rule(path, rule["pattern"], input_text)
    else:
        return RuleResult(path=path, value=None, matched=False)


def _regex_rule(path: str, pattern: str, text: str, group: int = 1) -> RuleResult:
    """Extract a single value with a regex."""
    m = re.search(pattern, text, re.MULTILINE)
    if m:
        return RuleResult(path=path, value=m.group(group).strip(), matched=True)
    return RuleResult(path=path, value=None, matched=False)


def _regex_all_rule(path: str, pattern: str, text: str) -> RuleResult:
    """Extract all matches as a list."""
    matches = re.findall(pattern, text, re.MULTILINE)
    if matches:
        return RuleResult(path=path, value=matches, matched=True)
    return RuleResult(path=path, value=None, matched=False)


def set_nested(obj: dict, path: str, value) -> None:
    """Set a value in a nested dict using dot notation."""
    keys = path.split(".")
    for key in keys[:-1]:
        obj = obj.setdefault(key, {})
    obj[keys[-1]] = value


def execute_rules(rules: list[dict], input_text: str) -> tuple[dict, list[RuleResult]]:
    """Execute all rules and return (extracted_dict, results)."""
    extracted = {}
    results = []
    for rule in rules:
        result = execute_rule(rule, input_text)
        results.append(result)
        if result.matched and result.value is not None:
            set_nested(extracted, result.path, result.value)
    return extracted, results
