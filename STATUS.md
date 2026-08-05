# Drift — Project Status

**Last updated:** 2026-06-12 20:48:03 EDT

## Current state

- Hybrid synthesis ladder operational: Tier 1 (regex/literal from logprob entropy),
  Tier 2 (mapping/conditional from cross-probe consistency + attention hints),
  Tier 3 (bag-of-words logistic regression, confidence-gated, JSD drift monitoring).
- MLX server (`drift serve`) exposes `/v1/chat/completions`, `/v1/logprobs` (full-vocab
  entropy), `/v1/attention` (input attribution).
- Tier 3 classifiers: coverage scales with labeled data (80.6% on `severity` at n=500,
  τ=0.95); the earlier ~25% figure at n≈400 was sample starvation, not a hard limit
  (see docs/THEORY.md "Beyond the Entropy Floor" and Finding 1 below). Vocabulary-poor
  fields (`root_cause`) still need hidden-state probes or LLM fallback.
- Safety track: tiered guard (keyword → logistic regression → LLM guard), Llama Guard
  3-1B FPR finding documented in docs/LLAMA_GUARD_SE.md.

## Active research thread: reasoning extraction

Goal: replace more of the LLM by extracting decision logic from hidden states and full
output distributions into (possibly probabilistic) generated code.

- Analysis: `docs/design/REASONING_EXTRACTION_ANALYSIS.md`
- Implementation plan: `docs/design/HIDDEN_STATE_PROBES_PLAN.md`
- Core infrastructure implemented and verified: an MLX server exposes full-vocabulary
  logprobs and hidden-state capture; a `hidden_probe` rule type with confidence-gated
  escalation and layer selection compiles probes into the same hybrid pipeline as the
  regex/classifier tiers.
- **Finding 1 — the Tier 3 "25% ceiling" was sample starvation, not feature poverty.**
  With 500 labels, plain bag-of-words logistic regression reaches 80% coverage on
  `severity` (see docs/THEORY.md "Beyond the Entropy Floor"). The hidden-probe advantage
  is real only where vocabulary doesn't carry the signal (`root_cause`: 24x coverage,
  1.4% BoW vs 34.1% hidden).
- **Finding 2 — cross-model transfer works.** Probes trained on a small model's
  (Qwen2.5-0.5B) hidden states predict a much larger model's (32B) decisions at parity
  with the large model's own self-probes, at roughly 270x lower latency (~80ms vs
  ~21.5s per ticket).
- Open question: whether a bag-of-words + hidden-state ensemble strictly dominates
  either approach alone, and how probe confidence calibration holds up on multi-class
  fields.
