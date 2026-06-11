# Drift — Project Status

**Last updated:** 2026-06-11 12:11:08 EDT

## Current state

- Hybrid synthesis ladder operational: Tier 1 (regex/literal from logprob entropy),
  Tier 2 (mapping/conditional from cross-probe consistency + attention hints),
  Tier 3 (bag-of-words logistic regression, confidence-gated, JSD drift monitoring).
- MLX server (`drift serve`) exposes `/v1/chat/completions`, `/v1/logprobs` (full-vocab
  entropy), `/v1/attention` (input attribution).
- Known ceiling: Tier 3 classifiers cover ~25% of cases at 0.95 confidence; remaining
  fields fall through to full LLM decode (see docs/THEORY.md "Beyond the Entropy Floor").
- Safety track: tiered guard (keyword → logistic regression → LLM guard), Llama Guard
  3-1B FPR finding documented in docs/LLAMA_GUARD_SE.md.

## Active research thread: reasoning extraction (Tier 4+)

Goal: replace more of the LLM by extracting decision logic from hidden states and full
output distributions into (possibly probabilistic) generated code.

- Analysis: `docs/design/REASONING_EXTRACTION_ANALYSIS.md`
- Implementation plan (7 phases, decision-gated): `docs/design/HIDDEN_STATE_PROBES_PLAN.md`
- Status: **plan drafted, implementation not started** (stopped at plan per direction,
  2026-06-11).
- Next action when resumed: Phase 0 — `src/drift/probing/mlx_hidden.py` + `/v1/hidden`
  endpoint (prefill-only hidden-state capture), then Phase 1 Tier 4 `hidden_probe` rules
  benchmarked against the Tier 3 coverage ceiling. Go/no-go answer expected in ~1 week of
  work (Decision Gate 1).

## Housekeeping

- 2026-06-11: design docs consolidated into `docs/design/` (ATTENTION_PLAN.md,
  ANSIBLE_TRIAGE_PLAN.md, CONDITIONAL_CLASSIFIERS.md moved; references updated in
  CLAUDE.md and docs/HYBRID.md).
