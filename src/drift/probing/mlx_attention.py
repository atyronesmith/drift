"""Attention-weight probing via MLX on Apple Silicon.

Uses a two-phase approach:
  1. Generate output normally (fused SDPA, correct results)
  2. Replay the full sequence (input + output) with patched attention
     to capture weights without numerical drift

This reveals which input tokens the model attends to when producing
each output token, enabling attention-based rule discovery.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler


@dataclass
class AttentionProbe:
    """Attention data for a single generated token."""
    token: str
    token_id: int
    position: int  # position in generated sequence (0-indexed)
    # Top attended input positions: [(input_pos, weight), ...]
    top_attended: list[tuple[int, float]]
    # Per-head attention over input (only stored if requested)
    per_head: list[list[float]] | None = None


@dataclass
class AttentionResult:
    """Full attention probe result for a generation."""
    input_tokens: list[str]  # decoded input tokens
    output_tokens: list[AttentionProbe]
    content: str
    model: str
    n_input: int  # number of input tokens (for offset calculations)

    def attribution_map(self, top_k: int = 5) -> list[dict]:
        """For each output token, show top-k attended input tokens."""
        result = []
        for probe in self.output_tokens:
            attended = []
            for pos, weight in probe.top_attended[:top_k]:
                if pos < len(self.input_tokens):
                    attended.append({
                        "input_token": self.input_tokens[pos],
                        "input_pos": pos,
                        "weight": weight,
                    })
            result.append({
                "output_token": probe.token,
                "output_pos": probe.position,
                "attends_to": attended,
            })
        return result

    def field_attribution(
        self,
        input_text: str,
        output_field_tokens: list[int],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Aggregate attention from output field tokens back to input spans.

        Args:
            input_text: The raw input text (for context).
            output_field_tokens: Indices into self.output_tokens for the field.
            top_k: Number of top input positions to return.

        Returns:
            [(input_token, total_weight), ...] sorted by weight descending.
        """
        weight_by_pos: dict[int, float] = {}
        for idx in output_field_tokens:
            if idx >= len(self.output_tokens):
                continue
            for pos, weight in self.output_tokens[idx].top_attended:
                weight_by_pos[pos] = weight_by_pos.get(pos, 0) + weight

        ranked = sorted(weight_by_pos.items(), key=lambda x: x[1], reverse=True)
        result = []
        for pos, weight in ranked[:top_k]:
            tok = self.input_tokens[pos] if pos < len(self.input_tokens) else "<gen>"
            result.append((tok, weight))
        return result


def _get_model_attention_class(model):
    """Find the Attention class used by this model architecture."""
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name == "Attention":
            return type(module)
    raise ValueError("Could not find Attention class in model")


def _probe_attention_with_model(
    mlx_model,
    tokenizer,
    system: str,
    prompt: str,
    model_name: str = "",
    max_tokens: int = 4096,
    temperature: float = 0,
    top_k_attend: int = 20,
    last_n_layers: int = 1,
    store_per_head: bool = False,
) -> AttentionResult:
    """Core attention probing using a pre-loaded model.

    This function is used by both local probe_attention() and the remote
    MLX server endpoint.

    Phase 1: Generate output normally using fused SDPA (correct results).
    Phase 2: Replay full sequence (input + output) with patched attention
             in a single forward pass to capture weights.
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

    # Phase 1: Generate normally
    sampler = make_sampler(temp=temperature if temperature > 0 else 0)
    content = generate(
        mlx_model, tokenizer, prompt=formatted,
        max_tokens=max_tokens, sampler=sampler,
    )

    # Encode the generated output
    full_text = formatted + content
    full_ids = tokenizer.encode(full_text)
    output_ids = full_ids[n_input:]

    # Phase 2: Replay full sequence to capture attention weights
    n_layers = len(mlx_model.model.layers)
    target_layers = set(range(n_layers - last_n_layers, n_layers))

    captured: list[tuple[int, mx.array]] = []
    layer_counter = [0]

    attn_cls = _get_model_attention_class(mlx_model)
    orig_call = attn_cls.__call__

    def _capturing_call(self, x, mask=None, cache=None):
        layer_idx = layer_counter[0] % n_layers
        layer_counter[0] += 1

        if layer_idx not in target_layers:
            return orig_call(self, x, mask, cache)

        B, L, D = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            keys_exp = mx.repeat(keys, n_rep, axis=1)
            values_exp = mx.repeat(values, n_rep, axis=1)
        else:
            keys_exp, values_exp = keys, values

        # Compute in float32 to match fused SDPA behavior and avoid NaN
        scores = (queries.astype(mx.float32) @ keys_exp.astype(mx.float32).transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            if isinstance(mask, str) and mask == "causal":
                T_q, T_k = scores.shape[2], scores.shape[3]
                causal = mx.triu(mx.full((T_q, T_k), float('-inf')), k=1)
                scores = scores + causal
            else:
                scores = scores + mask.astype(mx.float32)

        weights = mx.softmax(scores, axis=-1)
        captured.append((layer_idx, weights))

        output = (weights @ values_exp).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    attn_cls.__call__ = _capturing_call
    try:
        # Single forward pass with full sequence (no KV cache)
        full_tokens = mx.array([full_ids])
        mlx_model(full_tokens)
        mx.eval(*(w for _, w in captured))
    finally:
        attn_cls.__call__ = orig_call

    # Extract per-output-token attention from the captured weights
    output_probes: list[AttentionProbe] = []

    for out_idx, token_id in enumerate(output_ids):
        seq_pos = n_input + out_idx

        avg_weights = None
        for _, w in captured:
            w_at_pos = w[0, :, seq_pos, :]  # [n_heads, full_seq]
            head_avg = w_at_pos.mean(axis=0)  # [full_seq]
            if avg_weights is None:
                avg_weights = head_avg
            else:
                avg_weights = avg_weights + head_avg
        avg_weights = avg_weights / len(captured)
        avg_list = avg_weights.tolist()

        # Top-k attended positions (only consider input positions)
        input_weights = [(pos, avg_list[pos]) for pos in range(n_input)]
        input_weights.sort(key=lambda x: x[1], reverse=True)
        top_attended = input_weights[:top_k_attend]

        per_head = None
        if store_per_head:
            _, last_w = captured[-1]
            per_head = last_w[0, :, seq_pos, :n_input].tolist()

        output_probes.append(AttentionProbe(
            token=tokenizer.decode([token_id]),
            token_id=token_id,
            position=out_idx,
            top_attended=top_attended,
            per_head=per_head,
        ))

    return AttentionResult(
        input_tokens=input_tokens_decoded,
        output_tokens=output_probes,
        content=content,
        model=model_name,
        n_input=n_input,
    )


def probe_attention(
    system: str,
    prompt: str,
    model: str = "mlx-community/Qwen2.5-32B-Instruct-4bit",
    max_tokens: int = 4096,
    temperature: float = 0,
    top_k_attend: int = 20,
    last_n_layers: int = 1,
    store_per_head: bool = False,
) -> AttentionResult:
    """Probe with attention weight capture (two-phase).

    Args:
        system: System prompt.
        prompt: User prompt.
        model: HuggingFace model ID (mlx-community format).
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        top_k_attend: Number of top attended positions to store per token.
        last_n_layers: How many final layers to average attention from.
        store_per_head: Whether to store per-head attention weights.

    Returns:
        AttentionResult with per-token attention attribution.
    """
    from mlx_lm import load
    mlx_model, tokenizer = load(model)
    return _probe_attention_with_model(
        mlx_model, tokenizer, system, prompt,
        model_name=model, max_tokens=max_tokens,
        temperature=temperature, top_k_attend=top_k_attend,
        last_n_layers=last_n_layers, store_per_head=store_per_head,
    )
