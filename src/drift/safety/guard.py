"""Safety guard module for auditing LLM outputs via guard models.

Uses Llama Guard (or compatible) on Ollama to classify LLM outputs
as safe/unsafe. Supports both batch auditing of saved results and
inline checking during pipeline execution.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass


@dataclass
class GuardResult:
    """Result of a safety guard check."""
    safe: bool
    raw_response: str
    category: str | None  # e.g. "S1" for violent crimes
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


class GuardModel:
    """Wraps a guard model on Ollama for safety classification."""

    def __init__(
        self,
        model: str = "llama-guard3:1b",
        base_url: str | None = None,
        exclude_categories: list[str] | None = None,
    ):
        self.model = model
        self.base_url = (
            base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self.exclude_categories = set(exclude_categories or [])

    def check(self, content: str, prompt: str = "") -> GuardResult:
        """Check if content is safe according to the guard model.

        Args:
            content: The LLM output to check.
            prompt: Optional original prompt (for context).

        Returns:
            GuardResult with safe/unsafe classification.
        """
        # Build the conversation for the guard model.
        # Llama Guard expects a conversation format:
        #   User: <the prompt>
        #   Assistant: <the response to check>
        messages = []
        if prompt:
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": content})
        else:
            # If no prompt, just check the content as a user message
            # that the model "would have produced"
            messages.append({"role": "user", "content": "Analyze this output."})
            messages.append({"role": "assistant", "content": content})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "stream": False,
        }

        url = f"{self.base_url}/v1/chat/completions"
        start = time.perf_counter()

        result = subprocess.run(
            ["curl", "-s", "-X", "POST", url,
             "-H", "Content-Type: application/json",
             "-d", json.dumps(payload)],
            capture_output=True, text=True, timeout=60,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        if result.returncode != 0:
            raise RuntimeError(
                f"Guard check failed (curl exit {result.returncode}): {result.stderr}"
            )

        data = json.loads(result.stdout)
        if "error" in data:
            raise RuntimeError(f"Guard API error: {data['error']}")

        response_text = data["choices"][0]["message"]["content"].strip()
        return self._parse_response(response_text, latency_ms)

    def _parse_response(self, response: str, latency_ms: float) -> GuardResult:
        """Parse Llama Guard response into structured result.

        Llama Guard responds with either:
          "safe"
        or:
          "unsafe\nS1" (with category code)

        If exclude_categories is set, flags in those categories are
        treated as safe (the guard fired but we don't care).
        """
        lines = response.strip().split("\n")
        verdict = lines[0].strip().lower()

        if verdict == "safe":
            return GuardResult(
                safe=True,
                raw_response=response,
                category=None,
                latency_ms=round(latency_ms, 1),
            )
        else:
            category = lines[1].strip() if len(lines) > 1 else None
            excluded = category in self.exclude_categories if category else False
            return GuardResult(
                safe=excluded,  # treat excluded categories as safe
                raw_response=response,
                category=category,
                latency_ms=round(latency_ms, 1),
            )


class TieredGuard:
    """Tiered safety guard: classifier first, LLM guard on escalation.

    Uses drift.safety.classifier.classify() for fast Tier 1/2 classification.
    Only escalates to the full LLM guard (GuardModel) when the classifier
    confidence is below threshold. Tracks per-tier statistics.
    """

    def __init__(
        self,
        guard_model: GuardModel,
        weights_path: str,
        keywords_path: str | None = None,
    ):
        self.guard_model = guard_model
        self.stats = {"tier1": 0, "tier2": 0, "escalated": 0, "total": 0}

        from drift.safety.classifier import load_weights
        load_weights(weights_path, keywords_path)

    def check(self, content: str, prompt: str = "") -> GuardResult:
        """Check content safety using tiered classification.

        Tier 1/2 (drift.safety.classifier, <1ms) handles most inputs.
        Tier 3 (LLM guard) only called on escalation.

        Returns:
            GuardResult compatible with GuardModel.check() output.
        """
        from drift.safety.classifier import classify

        self.stats["total"] += 1
        start = time.perf_counter()

        verdict, confidence = classify(content)

        if verdict != "escalate":
            latency_ms = (time.perf_counter() - start) * 1000
            tier = "tier1" if confidence >= 0.95 else "tier2"
            self.stats[tier] += 1
            return GuardResult(
                safe=(verdict == "safe"),
                raw_response=f"tiered:{verdict} (conf={confidence:.3f}, tier={tier})",
                category=None if verdict == "safe" else "classifier",
                latency_ms=round(latency_ms, 1),
            )

        # Escalate to full LLM guard
        self.stats["escalated"] += 1
        return self.guard_model.check(content, prompt)

    def summary(self) -> dict:
        """Return tier distribution stats."""
        total = self.stats["total"]
        if total == 0:
            return self.stats
        return {
            **self.stats,
            "classifier_handled_pct": round(
                (self.stats["tier1"] + self.stats["tier2"]) / total * 100, 1
            ),
            "escalation_pct": round(self.stats["escalated"] / total * 100, 1),
        }


# Llama Guard 3 safety categories for reference
GUARD_CATEGORIES = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}
