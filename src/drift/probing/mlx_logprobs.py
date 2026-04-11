"""Logprob probing via MLX on Apple Silicon.

Unlike Ollama's top-k logprobs, MLX gives access to the full vocabulary
distribution, enabling exact Shannon entropy calculation.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from drift.probing.logprobs import ProbeResult, TokenProbe


class MLXTokenProbe(TokenProbe):
    """TokenProbe with exact full-vocabulary entropy."""

    def __init__(self, token, logprob, prob, top_alternatives, exact_entropy):
        super().__init__(
            token=token, logprob=logprob, prob=prob,
            top_alternatives=top_alternatives,
        )
        self._exact_entropy = exact_entropy

    @property
    def entropy(self) -> float:
        """Exact Shannon entropy from full vocabulary distribution."""
        return self._exact_entropy


def _probe_logprobs_with_model(
    mlx_model,
    tokenizer,
    system: str,
    prompt: str,
    model_name: str = "",
    max_tokens: int = 4096,
    temperature: float = 0,
    top_k_report: int = 10,
) -> ProbeResult:
    """Core logprob probing using a pre-loaded model.

    This function is used by both local probe_with_mlx() and the remote
    MLX server endpoint.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = mx.array(tokenizer.encode(formatted))

    tokens_out: list[TokenProbe] = []
    generated_tokens: list[int] = []

    sampler = make_sampler(temp=temperature if temperature > 0 else 0)
    prompt_tokens = input_ids

    for token, logits in generate_step(prompt_tokens, mlx_model, sampler=sampler):
        token_id = token.item() if hasattr(token, 'item') else int(token)

        if token_id in getattr(tokenizer, 'eos_token_ids', {tokenizer.eos_token_id}):
            break
        if len(generated_tokens) >= max_tokens:
            break

        generated_tokens.append(token_id)

        # Full-vocab log probabilities
        logits_flat = logits.reshape(-1)
        log_probs = nn.log_softmax(logits_flat, axis=-1)
        mx.eval(log_probs)  # Force evaluation to free computation graph
        log_probs_list = log_probs.tolist()
        del logits_flat, log_probs, logits

        token_logprob = log_probs_list[token_id]
        token_prob = math.exp(token_logprob)

        # Exact Shannon entropy from full distribution
        probs_full = [math.exp(lp) for lp in log_probs_list]
        total = sum(probs_full)
        if total > 0:
            probs_full = [p / total for p in probs_full]
        exact_entropy = -sum(p * math.log2(p) for p in probs_full if p > 0)

        # Top-k alternatives for reporting (excluding selected token)
        indexed = sorted(enumerate(log_probs_list), key=lambda x: x[1], reverse=True)
        top_alts = []
        for alt_id, alt_lp in indexed[:top_k_report + 1]:
            if alt_id == token_id:
                continue
            top_alts.append({
                "token": tokenizer.decode([alt_id]),
                "logprob": alt_lp,
            })
            if len(top_alts) >= top_k_report:
                break

        token_str = tokenizer.decode([token_id])
        tokens_out.append(MLXTokenProbe(
            token=token_str,
            logprob=token_logprob,
            prob=token_prob,
            top_alternatives=top_alts,
            exact_entropy=exact_entropy,
        ))

        prompt_tokens = mx.array([token_id])

    content = tokenizer.decode(generated_tokens)
    return ProbeResult(tokens=tokens_out, content=content, model=model_name)


def probe_with_mlx(
    system: str,
    prompt: str,
    model: str = "mlx-community/Qwen2.5-32B-Instruct-4bit",
    max_tokens: int = 4096,
    temperature: float = 0,
    top_k_report: int = 10,
) -> ProbeResult:
    """Probe using MLX with full vocabulary distributions.

    Args:
        system: System prompt.
        prompt: User prompt.
        model: HuggingFace model ID (mlx-community format).
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        top_k_report: Number of top alternatives to include per token.

    Returns:
        ProbeResult with exact entropy from full vocab distribution.
    """
    from mlx_lm import load
    mlx_model, tokenizer = load(model)
    return _probe_logprobs_with_model(
        mlx_model, tokenizer, system, prompt,
        model_name=model, max_tokens=max_tokens,
        temperature=temperature, top_k_report=top_k_report,
    )
