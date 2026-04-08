from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone

from drift.providers.base import LLMResponse


class OllamaProvider:
    """Provider for Ollama's OpenAI-compatible API.

    Uses curl subprocess for HTTP requests to avoid endpoint security software
    blocking Python's direct TCP connections to LAN addresses.
    """

    def __init__(self, model: str, base_url: str | None = None):
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    async def complete(self, system: str, prompt: str, **kwargs) -> LLMResponse:
        url = f"{self.base_url}/v1/chat/completions"
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
             "-d", json.dumps(payload)],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Ollama request failed (curl exit {result.returncode}): {result.stderr}")
        elapsed_ms = (time.perf_counter() - start) * 1000

        data = json.loads(result.stdout)
        if "error" in data:
            raise RuntimeError(f"Ollama API error: {result.stdout}")
        usage = data.get("usage", {})

        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", self.model),
            provider="ollama",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            ttft_ms=None,
            total_ms=round(elapsed_ms, 1),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
