from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from anthropic import AnthropicVertex

from drift.providers.base import LLMResponse


class VertexProvider:
    """Claude via Google Cloud Vertex AI."""

    def __init__(self, model: str, project_id: str | None = None, region: str | None = None):
        self.model = model
        self.project_id = project_id or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
        self.region = region or os.environ.get("CLOUD_ML_REGION")
        self.client = AnthropicVertex(project_id=self.project_id, region=self.region)

    async def complete(self, system: str, prompt: str, **kwargs) -> LLMResponse:
        temperature = kwargs.get("temperature", 0)
        start = time.perf_counter()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            provider="vertex",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            ttft_ms=None,
            total_ms=round(elapsed_ms, 1),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
