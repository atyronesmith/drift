"""Rule types for hybrid execution: extract deterministic fields from input."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass
class RuleResult:
    path: str
    value: str | int | float | list | None
    matched: bool
    escalation_hint: str | None = None
    escalation_confidence: float | None = None
    distribution: dict[str, float] | None = None


def execute_rule(rule: dict, input_text: str, extracted: dict | None = None,
                 hidden_states: dict[int, list[float]] | None = None) -> RuleResult:
    """Execute a single rule against input text.

    Args:
        rule: Rule definition dict
        input_text: Original input text
        extracted: Already-extracted fields (for mapping/conditional rules)
        hidden_states: {layer: hidden_vector} from a prefill pass (for
            hidden_probe rules). All probe rules share one capture.
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
    elif rule_type == "classifier":
        return _classifier_rule(path, rule, input_text, extracted or {})
    elif rule_type == "hidden_probe":
        return _hidden_probe_rule(path, rule, hidden_states)
    elif rule_type == "concept_program":
        return _concept_program_rule(path, rule, hidden_states)
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


def _classifier_rule(path: str, rule: dict, text: str, extracted: dict) -> RuleResult:
    """Classify using an inlined logistic regression model.

    Builds a feature vector from bag-of-words (binary) on text plus one-hot
    structured features from already-extracted fields. Computes logits via
    dot product, applies softmax, and returns the top class if confidence
    exceeds the threshold. Below-threshold predictions return matched=False
    so the field escalates to LLM.

    YAML example:
        - path: assessment.root_cause_category
          type: classifier
          classifier_type: logistic_regression
          vocabulary: ["access", "account", ...]
          weights: [[0.12, -0.34, ...], ...]
          bias: [0.1, -0.2, ...]
          classes: ["access_control", "configuration", ...]
          structured_features:
            - source: ticket.stated_priority
              categories: [High, Medium, Low, Urgent]
            - source: ticket.stated_category
              categories: [Network, Software, Account, ...]
          confidence_threshold: 0.95
    """
    vocab = rule["vocabulary"]
    weights = rule["weights"]
    bias = rule["bias"]
    classes = rule["classes"]
    threshold = rule.get("confidence_threshold", 0.95)

    # Tokenize input text
    words = set(re.findall(r"[a-z]+", text.lower()))

    # Build feature vector: bag-of-words (binary)
    n_vocab = len(vocab)
    features = [0.0] * (n_vocab + sum(
        len(sf["categories"]) for sf in rule.get("structured_features", [])
    ))

    for i, w in enumerate(vocab):
        if w in words:
            features[i] = 1.0

    # One-hot structured features
    offset = n_vocab
    for sf in rule.get("structured_features", []):
        source_value = _get_nested(extracted, sf["source"])
        for j, cat in enumerate(sf["categories"]):
            if str(source_value) == str(cat):
                features[offset + j] = 1.0
        offset += len(sf["categories"])

    # Logits = W @ features + b
    n_classes = len(classes)
    logits = [0.0] * n_classes
    for c in range(n_classes):
        s = bias[c]
        for j, fval in enumerate(features):
            if fval != 0:
                s += weights[c][j] * fval
        logits[c] = s

    # Softmax with numerical stability
    max_logit = max(logits)
    exp_logits = [math.exp(l - max_logit) for l in logits]
    total = sum(exp_logits)
    probs = [e / total for e in exp_logits]

    best_idx = max(range(n_classes), key=lambda i: probs[i])
    confidence = probs[best_idx]

    if confidence >= threshold:
        return RuleResult(path=path, value=classes[best_idx], matched=True)

    # Below threshold — escalate to LLM with tentative prediction
    return RuleResult(
        path=path, value=None, matched=False,
        escalation_hint=classes[best_idx],
        escalation_confidence=confidence,
    )


def _hidden_probe_rule(path: str, rule: dict,
                       hidden_states: dict[int, list[float]] | None) -> RuleResult:
    """Classify using a logistic probe on a hidden-state vector (Tier 4).

    The feature vector is the model's residual stream at a fixed layer and
    position, captured by one prefill pass shared across all probe rules.
    Returns the full class distribution; below-threshold predictions
    escalate to LLM with a hint, identical contract to classifier rules.

    YAML example:
        - path: assessment.root_cause_category
          type: hidden_probe
          layer: 38
          position: last_input
          classes: ["access_control", "hardware", ...]
          weights: [[...], ...]        # n_classes x d_model (or weights_ref: probes/field.npz)
          bias: [...]
          confidence_threshold: 0.95
          probe_model: mlx-community/Qwen2.5-32B-Instruct-4bit
    """
    layer = rule["layer"]
    if not hidden_states or layer not in hidden_states:
        return RuleResult(path=path, value=None, matched=False)

    h = hidden_states[layer]
    weights, bias = _load_probe_weights(rule)
    classes = rule["classes"]
    threshold = rule.get("confidence_threshold", 0.95)

    if not weights or len(weights[0]) != len(h):
        return RuleResult(path=path, value=None, matched=False)

    logits = []
    for c in range(len(classes)):
        s = bias[c]
        wc = weights[c]
        for j, hv in enumerate(h):
            s += wc[j] * hv
        logits.append(s)

    max_logit = max(logits)
    exp_logits = [math.exp(l - max_logit) for l in logits]
    total = sum(exp_logits)
    probs = [e / total for e in exp_logits]
    distribution = {classes[i]: probs[i] for i in range(len(classes))}

    best_idx = max(range(len(classes)), key=lambda i: probs[i])
    confidence = probs[best_idx]

    if confidence >= threshold:
        return RuleResult(path=path, value=classes[best_idx], matched=True,
                          distribution=distribution)

    return RuleResult(
        path=path, value=None, matched=False,
        escalation_hint=classes[best_idx],
        escalation_confidence=confidence,
        distribution=distribution,
    )


def _concept_program_rule(path: str, rule: dict,
                          hidden_states: dict[int, list[float]] | None) -> RuleResult:
    """Evaluate a concept-bottleneck program (auditable Tier 4 variant).

    Each concept is a binary probe (sigmoid of a dot product on a hidden
    state); the program is a shallow decision tree over the probed concept
    probabilities. Confidence is the leaf purity from training; below the
    threshold the field escalates with the tentative value as hint.

    YAML example:
        - path: assessment.assignment_group
          type: concept_program
          confidence_threshold: 0.9
          concepts:
            - name: is_lockout
              layer: 38
              weights: [...]   # d_model floats
              bias: 0.0
          program:
            if: {concept: is_lockout, op: ">", threshold: 0.5}
            then: {value: IdentityOps, confidence: 0.97, samples: 40}
            else: {value: DeskOps, confidence: 0.85, samples: 50}
    """
    from drift.hybrid.concepts import eval_concept_program

    if not hidden_states:
        return RuleResult(path=path, value=None, matched=False)

    concept_values = {}
    for concept in rule["concepts"]:
        h = hidden_states.get(concept["layer"])
        if h is None or len(concept["weights"]) != len(h):
            return RuleResult(path=path, value=None, matched=False)
        z = concept["bias"] + sum(w * hv for w, hv in zip(concept["weights"], h))
        concept_values[concept["name"]] = 1.0 / (1.0 + math.exp(-z))

    value, confidence = eval_concept_program(rule["program"], concept_values)
    threshold = rule.get("confidence_threshold", 0.95)

    if confidence >= threshold:
        return RuleResult(path=path, value=value, matched=True)

    return RuleResult(
        path=path, value=None, matched=False,
        escalation_hint=value, escalation_confidence=confidence,
    )


_PROBE_WEIGHT_CACHE: dict[str, tuple[list, list]] = {}


def _load_probe_weights(rule: dict) -> tuple[list, list]:
    """Return (weights, bias) for a hidden_probe rule.

    Inline weights take precedence; otherwise weights_ref names an .npz
    sidecar with arrays 'weights' and 'bias' (cached per path).
    """
    if rule.get("weights") is not None:
        return rule["weights"], rule["bias"]

    ref = rule.get("weights_ref")
    if not ref:
        return [], []
    if ref in _PROBE_WEIGHT_CACHE:
        return _PROBE_WEIGHT_CACHE[ref]

    import numpy as np
    data = np.load(ref)
    weights = data["weights"].tolist()
    bias = data["bias"].tolist()
    _PROBE_WEIGHT_CACHE[ref] = (weights, bias)
    return weights, bias


def _match_conditions(when: dict, extracted: dict) -> bool:
    """Check if all conditions in a 'when' clause match extracted values."""
    for field_path, expected in when.items():
        actual = _get_nested(extracted, field_path)
        if str(actual) != str(expected):
            return False
    return True


def _get_nested(obj: dict, path: str):
    """Get a value from a nested dict/list using dot notation with array indices."""
    for key in _parse_path_segments(path):
        if isinstance(key, int):
            if isinstance(obj, list) and key < len(obj):
                obj = obj[key]
            else:
                return None
        elif isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return None
    return obj


def _parse_path_segments(path: str) -> list[str | int]:
    """Parse a dotted path with array indices into segments.

    "issues[0].severity" -> ["issues", 0, "severity"]
    "ticket.stated_priority" -> ["ticket", "stated_priority"]
    """
    import re as _re
    segments = []
    for part in path.split("."):
        m = _re.match(r"^(\w+)\[(\d+)\]$", part)
        if m:
            segments.append(m.group(1))
            segments.append(int(m.group(2)))
        else:
            segments.append(part)
    return segments


def set_nested(obj: dict, path: str, value) -> None:
    """Set a value in a nested dict/list using dot notation with array indices.

    Supports paths like "issues[0].severity" by creating lists as needed.
    """
    segments = _parse_path_segments(path)
    for i, key in enumerate(segments[:-1]):
        next_key = segments[i + 1]
        if isinstance(key, int):
            while len(obj) <= key:
                obj.append({} if isinstance(next_key, str) else [])
            obj = obj[key]
        else:
            if isinstance(next_key, int):
                obj = obj.setdefault(key, [])
            else:
                obj = obj.setdefault(key, {})

    last = segments[-1]
    if isinstance(last, int):
        while len(obj) <= last:
            obj.append(None)
        obj[last] = value
    else:
        obj[last] = value


def execute_rules(rules: list[dict], input_text: str,
                  hidden_states: dict[int, list[float]] | None = None
                  ) -> tuple[dict, list[RuleResult]]:
    """Execute all rules and return (extracted_dict, results).

    Rules are executed in order. Mapping and conditional rules can
    reference fields extracted by earlier rules. hidden_states (one
    prefill capture) is shared by all hidden_probe rules.
    """
    extracted = {}
    results = []
    for rule in rules:
        result = execute_rule(rule, input_text, extracted, hidden_states=hidden_states)
        results.append(result)
        if result.matched and result.value is not None:
            set_nested(extracted, result.path, result.value)
    return extracted, results
