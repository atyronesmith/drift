# Plan: Attention-Based Rule Discovery

## Goal

Use transformer attention weights to trace which input tokens influence each output token. This would reveal the LLM's internal decision logic for fields like severity, allowing us to build deterministic rules that replace the LLM even for "reasoning" fields.

## Why Attention Matters

The current approach (logprob entropy) tells us **how certain** the model is, but not **why**. Attention weights tell us which input tokens the model looked at when producing each output token. For severity classification:

```
  Current (logprobs only):
    Output "P3" → entropy=0.86 → "the model is uncertain" → stays with LLM

  With attention:
    Output "P3" → attends to "Priority: Medium" (weight 0.4)
                              "Category: Network" (weight 0.3)
                              "keeps disconnecting" (weight 0.2)
                → we now know WHAT drives the decision
                → could build: if priority=Medium AND category=Network → P3
```

## Architecture

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  Step 1: Run model via HuggingFace Transformers                  │
  │                                                                  │
  │  Load quantized model (GPTQ/AWQ) locally or on GPU server.      │
  │  Run inference with output_attentions=True.                      │
  │  Capture per-layer, per-head attention matrices.                 │
  │                                                                  │
  │  Output: attention[layer][head][output_pos][input_pos]           │
  └──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  Step 2: Attribution mapping                                     │
  │                                                                  │
  │  For each high-entropy output token (e.g., "3" in "P3"):        │
  │  - Average attention across heads/layers (or use last layer)    │
  │  - Rank input tokens by attention weight                         │
  │  - Map input tokens back to input fields                         │
  │    (e.g., token "Medium" → Priority field)                       │
  │                                                                  │
  │  Output: {output_field: [(input_field, weight), ...]}            │
  └──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  Step 3: Conditional rule extraction                             │
  │                                                                  │
  │  If severity attends primarily to (priority, category):          │
  │  - Run N tickets through step 1+2                                │
  │  - Build decision tree: input features → output value            │
  │  - Validate tree against held-out tickets                        │
  │                                                                  │
  │  Output: conditional rules (lookup tables or decision trees)     │
  └──────────────────────────────────────────────────────────────────┘
```

## Requirements

- **HuggingFace Transformers** with `output_attentions=True`
- **Quantized model** — Qwen2.5-32B requires ~20GB VRAM with 4-bit quantization (GPTQ or AWQ). The Ollama server's GPU should work.
- **Alternative**: use a smaller model (Qwen2.5-7B) for attention analysis, validate findings on the full 32B via Ollama

## Challenges

1. **Attention ≠ causation.** Attention weights show correlation, not causation. A token may attend to "Medium" without it being the deciding factor. Techniques like attention rollout or gradient-based attribution are more reliable but harder to implement.

2. **Multi-head complexity.** Qwen2.5-32B has 64 attention heads across 64 layers. Which heads matter? Typically the last few layers are most relevant for output decisions.

3. **Memory.** Full attention matrices for a 2048-token sequence across all layers/heads are large. May need to only capture the last layer or specific heads.

4. **Ollama doesn't expose attention.** We'd need a separate inference path through HuggingFace Transformers, adding a dependency and complexity. This would be a research tool, not part of the production pipeline.

## Simpler Alternative: Empirical Conditional Rules

Before building attention extraction, test whether severity is conditionally deterministic by brute force:

1. Run 100+ tickets through the LLM via standard Ollama inference
2. Group results by (priority, category) or other input features
3. Check if severity is consistent within each group
4. If yes, build a lookup table rule

This approach (implemented in `scripts/conditional_severity.py`) achieves the same outcome without attention weights — it discovers the mapping empirically rather than mechanistically. If the mapping is clean, attention analysis becomes unnecessary for this specific field.

Attention analysis remains valuable for:
- Understanding WHY mappings work (explainability)
- Discovering non-obvious input features that drive decisions
- Fields where the mapping depends on free-text content, not structured fields
