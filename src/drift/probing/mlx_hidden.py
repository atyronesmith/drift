"""Hidden-state (residual stream) probing via MLX on Apple Silicon.

Captures transformer block outputs at chosen layers and sequence positions.
The headline mode is prefill-only: a single parallel forward pass over the
input (no decode loop) yields hidden states at every input position, which
downstream linear probes consume to fill decision fields without generation.

Capture works by patching the TransformerBlock class __call__ (same pattern
as mlx_attention.py) — architecture-agnostic, no per-model mask handling.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass


@dataclass
class HiddenStateResult:
    """Hidden states captured from a forward pass.

    states[layer][position] -> hidden vector (list of float, length = d_model).
    Positions are absolute indices into the full token sequence; in
    prefill-only mode the sequence is just the input.
    """

    model: str
    n_input: int
    layers: list[int]
    positions: list[int]
    states: dict[int, dict[int, list[float]]]
    input_tokens: list[str]
    content: str | None  # None in prefill-only mode

    def vector(self, layer: int, position: int) -> list[float]:
        """Return the hidden vector at (layer, position). Raises KeyError if absent."""
        return self.states[layer][position]

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "n_input": self.n_input,
            "layers": self.layers,
            "positions": self.positions,
            "states": {
                str(layer): {str(pos): vec for pos, vec in by_pos.items()}
                for layer, by_pos in self.states.items()
            },
            "input_tokens": self.input_tokens,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HiddenStateResult":
        return cls(
            model=data["model"],
            n_input=data["n_input"],
            layers=list(data["layers"]),
            positions=list(data["positions"]),
            states={
                int(layer): {int(pos): vec for pos, vec in by_pos.items()}
                for layer, by_pos in data["states"].items()
            },
            input_tokens=list(data["input_tokens"]),
            content=data.get("content"),
        )


def _resolve_positions(spec: str | list[int], n_input: int) -> list[int]:
    """Resolve a position spec against the input length.

    "last_input" -> [n_input - 1]
    "all_input"  -> [0 .. n_input - 1]
    explicit list -> validated pass-through
    """
    if isinstance(spec, str):
        if spec == "last_input":
            return [n_input - 1]
        if spec == "all_input":
            return list(range(n_input))
        raise ValueError(f"Unknown position spec: {spec!r}")

    positions = list(spec)
    for p in positions:
        if p < 0 or p >= n_input:
            raise ValueError(f"Position {p} out of range for n_input={n_input}")
    return positions


def _resolve_layers(spec: str | list[int], n_layers: int) -> list[int]:
    """Resolve a layer spec: "all" or an explicit, validated list."""
    if isinstance(spec, str):
        if spec == "all":
            return list(range(n_layers))
        raise ValueError(f"Unknown layer spec: {spec!r}")

    layers = list(spec)
    for l in layers:
        if l < 0 or l >= n_layers:
            raise ValueError(f"Layer {l} out of range for n_layers={n_layers}")
    return layers


def _capture_block_outputs(mlx_model, full_ids: list[int],
                            target_layers: list[int], positions: list[int]) -> dict:
    """Run one forward pass, capturing block outputs at (layer, position) pairs.

    Patches the TransformerBlock class __call__ to record outputs. Only the
    requested positions are materialized — the full [layers x seq x d_model]
    tensor never exists on the Python side.

    Returns {layer: {position: list[float]}} in float32.
    """
    import mlx.core as mx

    layers = mlx_model.model.layers
    n_layers = len(layers)
    target_set = set(target_layers)

    block_cls = type(layers[0])
    orig_call = block_cls.__call__
    captured: dict[int, dict[int, list[float]]] = {}
    layer_counter = [0]

    def _capturing_call(self, *args, **kwargs):
        layer_idx = layer_counter[0] % n_layers
        layer_counter[0] += 1
        out = orig_call(self, *args, **kwargs)
        if layer_idx in target_set:
            h = out[0] if isinstance(out, tuple) else out  # [B, L, D]
            by_pos = {}
            for pos in positions:
                vec = h[0, pos, :].astype(mx.float32)
                mx.eval(vec)
                by_pos[pos] = vec.tolist()
            captured[layer_idx] = by_pos
        return out

    block_cls.__call__ = _capturing_call
    try:
        mlx_model(mx.array([full_ids]))
    finally:
        block_cls.__call__ = orig_call

    return captured


def _probe_hidden_with_model(
    mlx_model,
    tokenizer,
    system: str,
    prompt: str,
    model_name: str = "",
    layers: str | list[int] = "all",
    positions: str | list[int] = "last_input",
    prefill_only: bool = True,
    max_tokens: int = 0,
    temperature: float = 0,
) -> HiddenStateResult:
    """Core hidden-state probing using a pre-loaded model.

    Used by both local probe_hidden_states() and the MLX server endpoint.

    Prefill-only mode (default): a single forward pass over the input.
    Generation mode (prefill_only=False, max_tokens > 0): generate normally
    first, then replay the full sequence with capture hooks — same two-phase
    pattern as attention probing.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(formatted)
    n_input = len(input_ids)
    input_tokens_decoded = [tokenizer.decode([t]) for t in input_ids]

    n_layers = len(mlx_model.model.layers)
    resolved_layers = _resolve_layers(layers, n_layers)
    resolved_positions = _resolve_positions(positions, n_input)

    content = None
    full_ids = input_ids

    if not prefill_only and max_tokens > 0:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature if temperature > 0 else 0)
        content = generate(
            mlx_model, tokenizer, prompt=formatted,
            max_tokens=max_tokens, sampler=sampler,
        )
        full_ids = tokenizer.encode(formatted + content)

    states = _capture_block_outputs(
        mlx_model, full_ids, resolved_layers, resolved_positions
    )

    return HiddenStateResult(
        model=model_name,
        n_input=n_input,
        layers=resolved_layers,
        positions=resolved_positions,
        states=states,
        input_tokens=input_tokens_decoded,
        content=content,
    )


def probe_hidden_states(
    system: str,
    prompt: str,
    model: str = "mlx-community/Qwen2.5-32B-Instruct-4bit",
    layers: str | list[int] = "all",
    positions: str | list[int] = "last_input",
    prefill_only: bool = True,
    max_tokens: int = 0,
    temperature: float = 0,
) -> HiddenStateResult:
    """Probe hidden states with a locally loaded MLX model.

    Args:
        system: System prompt.
        prompt: User prompt.
        model: HuggingFace model ID (mlx-community format).
        layers: "all" or explicit layer indices to capture.
        positions: "last_input", "all_input", or explicit positions.
        prefill_only: If True (default), single forward pass, no generation.
        max_tokens: Generation budget when prefill_only=False.
        temperature: Sampling temperature for generation mode.
    """
    from mlx_lm import load

    mlx_model, tokenizer = load(model)
    return _probe_hidden_with_model(
        mlx_model, tokenizer, system, prompt,
        model_name=model, layers=layers, positions=positions,
        prefill_only=prefill_only, max_tokens=max_tokens,
        temperature=temperature,
    )


def probe_hidden_remote(
    system: str,
    prompt: str,
    base_url: str | None = None,
    layers: str | list[int] = "all",
    positions: str | list[int] = "last_input",
    prefill_only: bool = True,
    max_tokens: int = 0,
    timeout: int = 600,
) -> HiddenStateResult:
    """Probe hidden states via a running drift MLX server (/v1/hidden).

    Uses curl subprocess — endpoint security blocks Python's direct TCP
    sockets (same constraint as the Ollama/DeepInfra providers).
    """
    base = (base_url or os.environ.get("DRIFT_MLX_URL", "http://localhost:8080")).rstrip("/")
    url = f"{base}/v1/hidden"

    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "layers": layers,
        "positions": positions,
        "prefill_only": prefill_only,
        "max_tokens": max_tokens,
    }

    result = subprocess.run(
        ["curl", "-s", "-X", "POST", url,
         "-H", "Content-Type: application/json",
         "-d", json.dumps(payload)],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Request failed (curl exit {result.returncode}): {result.stderr}")

    data = json.loads(result.stdout)
    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")

    return HiddenStateResult.from_dict(data)
