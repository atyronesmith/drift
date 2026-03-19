from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone

from drift.providers.base import LLMResponse


class DeepInfraProvider:
    """Provider for DeepInfra's OpenAI-compatible API.

    Uses curl subprocess to avoid endpoint security software blocking
    Python's direct TCP connections.
    """

    BASE_URL = "https://api.deepinfra.com"

    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPINFRA_API_KEY", "")

    async def complete(self, system: str, prompt: str, **kwargs) -> LLMResponse:
        url = f"{self.BASE_URL}/v1/openai/chat/completions"
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
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", url,
             "-H", "Content-Type: application/json",
             "-H", f"Authorization: Bearer {self.api_key}",
             "-d", json.dumps(payload)],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"DeepInfra request failed (curl exit {result.returncode}): {result.stderr}")
        elapsed_ms = (time.perf_counter() - start) * 1000

        data = json.loads(result.stdout)
        if "error" in data or "detail" in data:
            raise RuntimeError(f"DeepInfra API error: {result.stdout}")
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
