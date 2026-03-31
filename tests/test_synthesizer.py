"""Tests for drift.probing.synthesizer module."""

import json
from drift.probing.synthesizer import analyze_fields, format_rules, _flatten_json


class TestFlattenJson:
    def test_simple(self):
        result = _flatten_json({"a": 1, "b": "hello"})
        assert len(result) == 2
        assert result[0] == ("a", "1", 1)
        assert result[1] == ("b", "hello", "hello")

    def test_nested(self):
        result = _flatten_json({"x": {"y": 42}})
        assert result == [("x.y", "42", 42)]

    def test_array(self):
        result = _flatten_json({"items": [10, 20]})
        assert result[0] == ("items[0]", "10", 10)
        assert result[1] == ("items[1]", "20", 20)


class TestAnalyzeFields:
    def _make_probe(self, content_json, entropies=None):
        content = json.dumps(content_json)
        tokens = []
        for i, ch in enumerate(content):
            e = 0.0
            if entropies and i < len(entropies):
                e = entropies[i]
            tokens.append({"token": ch, "logprob": 0.0, "entropy": e})
        return {
            "tokens": tokens,
            "high_entropy_tokens": [
                {"index": i, "token": t["token"], "entropy": t["entropy"], "alternatives": []}
                for i, t in enumerate(tokens) if t["entropy"] > 0.5
            ],
        }

    def test_all_deterministic(self):
        probe = self._make_probe({"name": "Alice", "age": 30})
        rules = analyze_fields(probe, input_text="Alice is 30")
        assert len(rules) == 2
        assert all(r.rule_type in ("extracted", "literal", "computed") for r in rules)

    def test_format_rules_produces_output(self):
        probe = self._make_probe({"x": 1})
        rules = analyze_fields(probe)
        text = format_rules(rules)
        assert "RULE EXTRACTION ANALYSIS" in text
        assert "Total fields:" in text

    def test_empty_json(self):
        rules = analyze_fields({"tokens": [], "high_entropy_tokens": []})
        assert rules == []
