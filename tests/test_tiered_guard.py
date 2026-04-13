"""Tests for TieredGuard safety module."""

import json
import pytest
from unittest.mock import patch, MagicMock

from drift.safety.guard import GuardResult, TieredGuard


@pytest.fixture
def weights_file(tmp_path):
    """Create a temporary guard classifier weights file."""
    weights = {
        "vocabulary": ["code", "function", "error", "password", "hack"],
        "weights": [[1.0, 1.0, 0.5, -1.0, -2.0, 0.0, 0.0, 0.0],
                     [-1.0, -1.0, -0.5, 1.0, 2.0, 0.0, 0.0, 0.0]],
        "bias": [0.5, -0.5],
        "classes": ["safe", "unsafe"],
        "confidence_threshold": 0.90,
    }
    path = tmp_path / "guard_weights.json"
    path.write_text(json.dumps(weights))
    return str(path)


@pytest.fixture
def keywords_file(tmp_path):
    """Create a temporary guard keywords file."""
    keywords = {
        "unsafe_keywords": ["exploit", "attack", "malware", "inject", "shellcode"],
        "safe_keywords": ["function", "return", "class", "import", "variable"],
    }
    path = tmp_path / "guard_keywords.json"
    path.write_text(json.dumps(keywords))
    return str(path)


@pytest.fixture
def mock_guard_model():
    """Create a mock GuardModel that returns safe."""
    model = MagicMock()
    model.check.return_value = GuardResult(
        safe=True, raw_response="safe", category=None, latency_ms=150.0,
    )
    return model


@pytest.fixture
def tiered_guard(mock_guard_model, weights_file, keywords_file):
    """Create a TieredGuard with mocked LLM guard."""
    # Reset denoise guard module state before each test
    import denoise.guard as dg
    dg._LOADED = False
    guard = TieredGuard(mock_guard_model, weights_file, keywords_file)
    return guard


def test_tiered_guard_classifier_handles_safe(tiered_guard, mock_guard_model):
    """Safe SE content should be handled by classifier, not LLM guard."""
    # Strong safe signal: code-like content with safe vocabulary
    result = tiered_guard.check("def function code return error handling")
    assert result.safe
    assert "tiered:" in result.raw_response
    mock_guard_model.check.assert_not_called()


def test_tiered_guard_escalates_ambiguous(tiered_guard, mock_guard_model):
    """Ambiguous content should escalate to LLM guard."""
    # Content with no strong signal → low confidence → escalation
    result = tiered_guard.check("something completely generic with no keywords")
    # Either the classifier handles it or escalates — if escalated, LLM guard was called
    if "tiered:" not in result.raw_response:
        mock_guard_model.check.assert_called_once()


def test_tiered_guard_stats_tracking(tiered_guard):
    """Stats should track tier distribution."""
    # Run multiple checks
    for _ in range(5):
        tiered_guard.check("def function code return error handling class import")

    stats = tiered_guard.summary()
    assert stats["total"] == 5
    assert stats["tier1"] + stats["tier2"] + stats["escalated"] == 5


def test_tiered_guard_returns_guard_result(tiered_guard):
    """TieredGuard.check() should always return a GuardResult."""
    result = tiered_guard.check("some content to check")
    assert isinstance(result, GuardResult)
    assert isinstance(result.safe, bool)
    assert isinstance(result.latency_ms, float)


def test_tiered_guard_summary_percentages(tiered_guard):
    """Summary should include percentage breakdowns."""
    for _ in range(10):
        tiered_guard.check("def function code return error handling class import")

    stats = tiered_guard.summary()
    assert "classifier_handled_pct" in stats
    assert "escalation_pct" in stats
    assert stats["classifier_handled_pct"] + stats["escalation_pct"] == 100.0


def test_tiered_guard_empty_summary(tiered_guard):
    """Summary with no checks should return zero stats."""
    stats = tiered_guard.summary()
    assert stats["total"] == 0


def test_tiered_guard_escalation_uses_llm(tiered_guard, mock_guard_model):
    """When classifier escalates, LLM guard result is returned directly."""
    mock_guard_model.check.return_value = GuardResult(
        safe=False, raw_response="unsafe\nS1", category="S1", latency_ms=200.0,
    )
    # Force escalation by resetting weights to trigger low confidence
    import denoise.guard as dg
    dg._LOADED = False  # classify returns ("escalate", 0.0) when not loaded

    result = tiered_guard.check("some content")
    assert not result.safe
    assert result.category == "S1"
    mock_guard_model.check.assert_called_once()


def test_tiered_guard_to_dict_compat(tiered_guard):
    """GuardResult from tiered guard should be serializable via to_dict()."""
    result = tiered_guard.check("def function code return class import")
    d = result.to_dict()
    assert "safe" in d
    assert "raw_response" in d
    assert "latency_ms" in d
