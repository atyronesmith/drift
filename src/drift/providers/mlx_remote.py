"""Provider for remote drift MLX server.

Connects to a drift MLX server running on a remote Apple Silicon machine.
Uses curl subprocess (like OllamaProvider) to avoid endpoint security software blocking.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone

from drift.providers.base import LLMResponse


class MLXRemoteProvider:
    """Provider for remote drift MLX server."""

    def __init__(self, model: str, base_url: str | None = None):
        self.model = model
        self.base_url = (
            base_url
            or os.environ.get("MLX_REMOTE_URL", "http://localhost:8080")
        ).rstrip("/")

    def _curl_post(self, path: str, payload: dict, timeout: int = 600) -> dict:
        """POST JSON to the server via curl subprocess."""
        url = f"{self.base_url}{path}"
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", url,
             "-H", "Content-Type: application/json",
             "-d", json.dumps(payload)],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"MLX remote request failed (curl exit {result.returncode}): {result.stderr}"
            )
        data = json.loads(result.stdout)
        if "error" in data:
            raise RuntimeError(f"MLX remote API error: {data['error']}")
        return data

    async def complete(self, system: str, prompt: str, **kwargs) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        start = time.perf_counter()
        data = self._curl_post("/v1/chat/completions", payload)
        elapsed_ms = (time.perf_counter() - start) * 1000

        usage = data.get("usage", {})
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", self.model),
            provider="mlx-remote",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            ttft_ms=None,
            total_ms=round(elapsed_ms, 1),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def probe_logprobs(
        self, system: str, prompt: str, **kwargs
    ):
        """Call /v1/logprobs for full-vocab entropy probing."""
        from drift.probing.logprobs import ProbeResult
        from drift.probing.mlx_logprobs import MLXTokenProbe

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "top_k_report": kwargs.get("top_k_report", 10),
        }

        data = self._curl_post("/v1/logprobs", payload)

        tokens = [
            MLXTokenProbe(
                token=t["token"],
                logprob=t["logprob"],
                prob=t["prob"],
                top_alternatives=t["top_alternatives"],
                exact_entropy=t["exact_entropy"],
            )
            for t in data["tokens"]
        ]

        return ProbeResult(
            tokens=tokens,
            content=data["content"],
            model=data["model"],
        )

    def probe_attention(
        self, system: str, prompt: str, **kwargs
    ):
        """Call /v1/attention for attention weight probing."""
        from drift.probing.mlx_attention import AttentionProbe, AttentionResult

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "top_k_attend": kwargs.get("top_k_attend", 20),
            "last_n_layers": kwargs.get("last_n_layers", 1),
            "store_per_head": kwargs.get("store_per_head", False),
        }

        data = self._curl_post("/v1/attention", payload)

        output_tokens = [
            AttentionProbe(
                token=t["token"],
                token_id=t["token_id"],
                position=t["position"],
                top_attended=[tuple(a) for a in t["top_attended"]],
                per_head=t.get("per_head"),
            )
            for t in data["output_tokens"]
        ]

        return AttentionResult(
            input_tokens=data["input_tokens"],
            output_tokens=output_tokens,
            content=data["content"],
            model=data["model"],
            n_input=data["n_input"],
        )
