"""Provider for MLX on Apple Silicon.

Runs models in-process via mlx-lm — no server required.
Exposes full vocabulary distributions for exact Shannon entropy
calculation (vs Ollama's top-k approximation).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from drift.providers.base import LLMResponse


class MLXProvider:
    """Provider using mlx-lm for native Apple Silicon inference."""

    def __init__(self, model: str, **kwargs):
        try:
            from mlx_lm import load
        except ImportError:
            raise ImportError(
                "mlx-lm is required for the MLX provider. "
                "Install with: pip install mlx-lm"
            )
        self.model_name = model
        self._tokenizer: object | None = None
        self._model: object | None = None

    def _ensure_loaded(self):
        if self._model is None:
            from mlx_lm import load
            self._model, self._tokenizer = load(self.model_name)

    async def complete(self, system: str, prompt: str, **kwargs) -> LLMResponse:
        from mlx_lm import generate

        self._ensure_loaded()

        temperature = kwargs.get("temperature", 0)
        max_tokens = kwargs.get("max_tokens", 4096)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        formatted = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self._tokenizer.encode(formatted)
        input_token_count = len(input_ids)

        temp = temperature if temperature > 0 else 0

        start = time.perf_counter()
        output = generate(
            self._model,
            self._tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            temp=temp,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        output_ids = self._tokenizer.encode(output)
        output_token_count = len(output_ids)

        return LLMResponse(
            content=output,
            model=self.model_name,
            provider="mlx",
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            ttft_ms=None,
            total_ms=round(elapsed_ms, 1),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
