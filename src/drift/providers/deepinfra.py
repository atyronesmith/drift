from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import httpx

from drift.providers.base import LLMResponse


class DeepInfraProvider:
    """Thin httpx wrapper for DeepInfra's OpenAI-compatible API."""

    BASE_URL = "https://api.deepinfra.com"

    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPINFRA_API_KEY", "")

    async def complete(self, system: str, prompt: str, **kwargs) -> LLMResponse:
        url = f"{self.BASE_URL}/v1/openai/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": kwargs.get("temperature", 0),
            "stream": False,
        }

        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
        elapsed_ms = (time.perf_counter() - start) * 1000

        data = resp.json()
        usage = data.get("usage", {})

        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", self.model),
            provider="deepinfra",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            ttft_ms=None,
            total_ms=round(elapsed_ms, 1),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
