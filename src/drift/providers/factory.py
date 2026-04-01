from __future__ import annotations

from drift.providers.ollama import OllamaProvider
from drift.providers.vertex import VertexProvider
from drift.providers.deepinfra import DeepInfraProvider


def make_provider(config: dict):
    """Build a provider from a config dict (experiment-level or step-level)."""
    provider_name = config.get("provider", "ollama")
    model = config["model"]

    if provider_name == "vertex":
        return VertexProvider(
            model=model,
            project_id=config.get("project_id"),
            region=config.get("region"),
        )
    elif provider_name == "deepinfra":
        return DeepInfraProvider(
            model=model,
            api_key=config.get("api_key"),
        )
    elif provider_name == "mlx":
        from drift.providers.mlx import MLXProvider
        return MLXProvider(model=model)
    else:
        return OllamaProvider(model=model, base_url=config.get("base_url"))
