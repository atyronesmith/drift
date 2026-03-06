from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Protocol, runtime_checkable


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str  # "anthropic" | "ollama"
    input_tokens: int
    output_tokens: int
    ttft_ms: float | None  # time to first token if available
    total_ms: float  # total request duration
    timestamp: str  # ISO 8601

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> LLMResponse:
        return cls(**d)


@runtime_checkable
class Provider(Protocol):
    async def complete(self, system: str, prompt: str, **kwargs) -> LLMResponse: ...
