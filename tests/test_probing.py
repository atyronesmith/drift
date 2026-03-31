"""Tests for drift.probing.logprobs module."""

import math
from drift.probing.logprobs import TokenProbe, ProbeResult


class TestTokenProbe:
    def test_entropy_deterministic(self):
        t = TokenProbe(token="x", logprob=0.0, prob=1.0, top_alternatives=[])
        assert t.entropy == 0.0

    def test_entropy_with_alternatives(self):
        t = TokenProbe(
            token="x", logprob=math.log(0.5), prob=0.5,
            top_alternatives=[{"token": "y", "logprob": math.log(0.5)}],
        )
        assert abs(t.entropy - 1.0) < 0.01  # 1 bit for 50/50 split

    def test_entropy_skewed(self):
        t = TokenProbe(
            token="x", logprob=math.log(0.9), prob=0.9,
            top_alternatives=[{"token": "y", "logprob": math.log(0.1)}],
        )
        assert t.entropy < 0.6  # low entropy for 90/10 split


class TestProbeResult:
    def _make_result(self, entropies):
        tokens = []
        for i, e in enumerate(entropies):
            if e == 0:
                tokens.append(TokenProbe(f"t{i}", 0.0, 1.0, []))
            else:
                # Create a token with approximately the target entropy
                tokens.append(TokenProbe(
                    f"t{i}", math.log(0.5), 0.5,
                    [{"token": f"alt{i}", "logprob": math.log(0.5)}],
                ))
        return ProbeResult(tokens=tokens, content="".join(t.token for t in tokens), model="test")

    def test_entropy_stats(self):
        result = self._make_result([0, 0, 0, 0, 1.0])
        stats = result.entropy_stats()
        assert stats["num_tokens"] == 5
        assert stats["pct_deterministic"] == 0.8

    def test_high_entropy_tokens(self):
        result = self._make_result([0, 1.0, 0, 1.0])
        high = result.high_entropy_tokens(threshold=0.5)
        assert len(high) == 2
        assert high[0][0] == 1
        assert high[1][0] == 3

    def test_low_entropy_tokens(self):
        result = self._make_result([0, 1.0, 0])
        low = result.low_entropy_tokens(threshold=0.1)
        assert len(low) == 2

    def test_segment_by_entropy(self):
        result = self._make_result([0, 0, 1.0, 1.0, 0])
        segments = result.segment_by_entropy(threshold=0.5)
        assert len(segments) == 3
        assert segments[0]["type"] == "deterministic"
        assert segments[1]["type"] == "stochastic"
        assert segments[2]["type"] == "deterministic"

    def test_empty_result(self):
        result = ProbeResult(tokens=[], content="", model="test")
        assert result.entropy_stats() == {}
        assert result.high_entropy_tokens() == []
        assert result.segment_by_entropy() == []
