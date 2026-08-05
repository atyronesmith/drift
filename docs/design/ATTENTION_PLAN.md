# Attention-Based Rule Discovery

## Status: Implemented

Attention probing is implemented in `src/drift/probing/mlx_attention.py` and available remotely via the drift MLX server's `/v1/attention` endpoint. See [MLX.md](MLX.md) for server setup.

## Why Attention Matters

The logprob entropy approach tells us **how certain** the model is, but not **why**. Attention weights tell us which input tokens the model looked at when producing each output token:

```
  Logprobs only:
    Output "P3" → entropy=0.86 → "the model is uncertain" → stays with LLM

  With attention:
    Output "P3" → attends to "Priority: Medium" (weight 0.4)
                              "Category: Network" (weight 0.3)
                              "keeps disconnecting" (weight 0.2)
                → we now know WHAT drives the decision
                → could build: if priority=Medium AND category=Network → P3
```

## Implementation

### Two-phase approach

Naive monkey-patching during autoregressive generation causes numerical drift — manual attention computation diverges from fused SDPA over many tokens, producing garbage output. The solution is two phases:

1. **Generate normally** — fused SDPA, correct output
2. **Replay full sequence** — single forward pass with input+output tokens, patched attention captures weights without affecting generation quality

### Key technical details

- **Float32 softmax** — manual attention must cast Q/K to float32 before computing scores to avoid NaN (matching fused SDPA kernel behavior)
- **GQA handling** — Qwen2.5 uses grouped query attention (fewer KV heads than query heads), requiring `mx.repeat` to expand KV heads
- **Causal mask** — during replay, the mask can be the string `"causal"` rather than a tensor; the patched code generates an explicit upper-triangular -inf mask

### Files

| File | Purpose |
|------|---------|
| `src/drift/probing/mlx_attention.py` | Core attention probing (local + server shared code) |
| `src/drift/server/mlx_server.py` | `/v1/attention` endpoint |
| `src/drift/providers/mlx_remote.py` | `probe_attention()` client method |

### Data structures

```python
@dataclass
class AttentionProbe:
    token: str          # decoded output token
    token_id: int
    position: int       # position in generated sequence
    top_attended: list[tuple[int, float]]  # [(input_pos, weight), ...]
    per_head: list[list[float]] | None     # optional per-head weights

@dataclass
class AttentionResult:
    input_tokens: list[str]
    output_tokens: list[AttentionProbe]
    content: str
    model: str
    n_input: int
```

`AttentionResult` provides helper methods:
- `attribution_map(top_k)` — per output token, top-k attended input tokens
- `field_attribution(input_text, output_field_tokens, top_k)` — aggregate attention from a set of output tokens back to input spans

## Next steps

1. **Attention vs logprobs experiment** — compare attention-based field classification (fields attending to input data = extraction, attending to system prompt = reasoning) against entropy-based classification on helpdesk tickets
2. **Conditional rule extraction** — for fields where attention points to structured input fields (e.g., severity → priority), build lookup table rules from multi-ticket probing
3. **Integration with autogen** — feed attention data into `autogen.py` to improve Tier 2 rule discovery (mappings, conditionals)

## Caveats

- **Attention != causation.** Attention weights show correlation, not causation. A token may attend to "Medium" without it being the deciding factor.
- **Last-layer bias.** By default only the final layer's attention is captured (`last_n_layers=1`). Earlier layers may contain different information.
- **Memory.** Full per-head attention for long sequences is large. Use `store_per_head=False` (default) unless needed.

## Empirical alternative

For structured fields, the empirical approach (`scripts/conditional_severity.py`) can discover the same mappings by running N tickets and grouping results — no attention weights needed. Attention is most valuable for free-text-dependent fields and explainability.
