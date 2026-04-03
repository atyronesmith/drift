"""Rule types for hybrid execution: extract deterministic fields from input."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RuleResult:
    path: str
    value: str | int | float | list | None
    matched: bool


def execute_rule(rule: dict, input_text: str, extracted: dict | None = None) -> RuleResult:
    """Execute a single rule against input text.

    Args:
        rule: Rule definition dict
        input_text: Original input text
        extracted: Already-extracted fields (for mapping/conditional rules)
    """
    path = rule["path"]
    rule_type = rule["type"]

    if rule_type == "regex":
        return _regex_rule(path, rule["pattern"], input_text, rule.get("group", 1))
    elif rule_type == "literal":
        return RuleResult(path=path, value=rule["value"], matched=True)
    elif rule_type == "regex_all":
        return _regex_all_rule(path, rule["pattern"], input_text)
    elif rule_type == "mapping":
        return _mapping_rule(path, rule, extracted or {})
    elif rule_type == "conditional":
        return _conditional_rule(path, rule, extracted or {})
    elif rule_type == "keyword":
        return _keyword_rule(path, rule, input_text)
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


def _mapping_rule(path: str, rule: dict, extracted: dict) -> RuleResult:
    """Look up a value from another extracted field.

    YAML example:
        - path: assessment.severity
          type: mapping
          source: ticket.stated_priority
          map:
            High: P2
            Medium: P3
            Low: P4
            Urgent: P1
          default: P3
    """
    source_path = rule["source"]
    source_value = _get_nested(extracted, source_path)
    if source_value is None:
        return RuleResult(path=path, value=rule.get("default"), matched=rule.get("default") is not None)

    mapping = rule["map"]
    value = mapping.get(str(source_value), rule.get("default"))
    return RuleResult(path=path, value=value, matched=value is not None)


def _conditional_rule(path: str, rule: dict, extracted: dict) -> RuleResult:
    """Evaluate ordered conditions against extracted fields. First match wins.

    YAML example:
        - path: assessment.severity
          type: conditional
          conditions:
            - when:
                ticket.stated_priority: High
                ticket.stated_category: Account
              value: P3
            - when:
                ticket.stated_priority: High
              value: P2
            - when:
                ticket.stated_priority: Medium
              value: P3
          default: P3
    """
    for condition in rule.get("conditions", []):
        when = condition.get("when", {})
        if _match_conditions(when, extracted):
            return RuleResult(path=path, value=condition["value"], matched=True)

    default = rule.get("default")
    return RuleResult(path=path, value=default, matched=default is not None)


def _keyword_rule(path: str, rule: dict, text: str) -> RuleResult:
    """Classify based on keyword presence in input text.

    YAML example:
        - path: assessment.root_cause_category
          type: keyword
          field: description
          keywords:
            network: ["printer", "disconnecting", "connectivity"]
            software_bug: ["error", "crash", "update"]
          default: user_error
    """
    text_lower = text.lower()
    scores = {}
    for category, words in rule["keywords"].items():
        scores[category] = sum(1 for w in words if w.lower() in text_lower)

    if scores:
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return RuleResult(path=path, value=best, matched=True)

    default = rule.get("default")
    return RuleResult(path=path, value=default, matched=default is not None)


def _match_conditions(when: dict, extracted: dict) -> bool:
    """Check if all conditions in a 'when' clause match extracted values."""
    for field_path, expected in when.items():
        actual = _get_nested(extracted, field_path)
        if str(actual) != str(expected):
            return False
    return True


def _get_nested(obj: dict, path: str):
    """Get a value from a nested dict using dot notation."""
    for key in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return None
    return obj


def set_nested(obj: dict, path: str, value) -> None:
    """Set a value in a nested dict using dot notation."""
    keys = path.split(".")
    for key in keys[:-1]:
        obj = obj.setdefault(key, {})
    obj[keys[-1]] = value


def execute_rules(rules: list[dict], input_text: str) -> tuple[dict, list[RuleResult]]:
    """Execute all rules and return (extracted_dict, results).

    Rules are executed in order. Mapping and conditional rules can
    reference fields extracted by earlier rules.
    """
    extracted = {}
    results = []
    for rule in rules:
        result = execute_rule(rule, input_text, extracted)
        results.append(result)
        if result.matched and result.value is not None:
            set_nested(extracted, result.path, result.value)
    return extracted, results
