"""Logprob probing via Ollama API.

Extracts per-token log probabilities from model output to identify
which parts of the generation are deterministic (high confidence)
vs. stochastic (low confidence / high entropy).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass


@dataclass
class TokenProbe:
    token: str
    logprob: float
    prob: float
    top_alternatives: list[dict]  # [{"token": str, "logprob": float}, ...]

    @property
    def entropy(self) -> float:
        """Shannon entropy across top alternatives (bits). 0 = deterministic."""
        probs = [math.exp(a["logprob"]) for a in self.top_alternatives]
        probs.append(self.prob)
        total = sum(probs)
        if total == 0:
            return 0.0
        probs = [p / total for p in probs]
        return -sum(p * math.log2(p) for p in probs if p > 0)


@dataclass
class ProbeResult:
    tokens: list[TokenProbe]
    content: str
    model: str

    def entropy_map(self) -> list[tuple[str, float]]:
        """Return (token, entropy) pairs."""
        return [(t.token, t.entropy) for t in self.tokens]

    def high_entropy_tokens(self, threshold: float = 0.5) -> list[tuple[int, str, float]]:
        """Tokens with entropy above threshold — candidates for drift."""
        return [
            (i, t.token, t.entropy)
            for i, t in enumerate(self.tokens)
            if t.entropy > threshold
        ]

    def low_entropy_tokens(self, threshold: float = 0.1) -> list[tuple[int, str, float]]:
        """Tokens with entropy below threshold — deterministic, rule-extractable."""
        return [
            (i, t.token, t.entropy)
            for i, t in enumerate(self.tokens)
            if t.entropy <= threshold
        ]

    def entropy_stats(self) -> dict:
        """Aggregate entropy statistics."""
        entropies = [t.entropy for t in self.tokens]
        if not entropies:
            return {}
        return {
            "mean_entropy": sum(entropies) / len(entropies),
            "max_entropy": max(entropies),
            "min_entropy": min(entropies),
            "num_tokens": len(entropies),
            "pct_deterministic": sum(1 for e in entropies if e < 0.1) / len(entropies),
            "pct_high_entropy": sum(1 for e in entropies if e > 0.5) / len(entropies),
        }

    def segment_by_entropy(self, threshold: float = 0.5) -> list[dict]:
        """Segment output into deterministic and stochastic runs."""
        segments = []
        current = {"type": "deterministic", "tokens": [], "start": 0}

        for i, t in enumerate(self.tokens):
            is_high = t.entropy > threshold
            segment_type = "stochastic" if is_high else "deterministic"

            if segment_type != current["type"]:
                if current["tokens"]:
                    current["text"] = "".join(current["tokens"])
                    current["end"] = i
                    segments.append(current)
                current = {"type": segment_type, "tokens": [], "start": i}

            current["tokens"].append(t.token)

        if current["tokens"]:
            current["text"] = "".join(current["tokens"])
            current["end"] = len(self.tokens)
            segments.append(current)

        return segments


def probe_with_logprobs(
    system: str,
    prompt: str,
    model: str = "qwen2.5:32b",
    base_url: str | None = None,
    top_logprobs: int = 5,
    temperature: float = 0,
) -> ProbeResult:
    """Send a completion request with logprobs enabled and parse the results."""
    base = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
    url = f"{base}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "stream": False,
    }

    result = subprocess.run(
        ["curl", "-s", "-X", "POST", url,
         "-H", "Content-Type: application/json",
         "-d", json.dumps(payload)],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Request failed (curl exit {result.returncode}): {result.stderr}")

    data = json.loads(result.stdout)
    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")

    choice = data["choices"][0]
    content = choice["message"]["content"]
    logprobs_data = choice.get("logprobs", {}).get("content", [])

    tokens = []
    for entry in logprobs_data:
        top_alts = [
            {"token": alt["token"], "logprob": alt["logprob"]}
            for alt in entry.get("top_logprobs", [])
            if alt["token"] != entry["token"]
        ]
        tokens.append(TokenProbe(
            token=entry["token"],
            logprob=entry["logprob"],
            prob=math.exp(entry["logprob"]),
            top_alternatives=top_alts,
        ))

    return ProbeResult(
        tokens=tokens,
        content=content,
        model=data.get("model", model),
    )
