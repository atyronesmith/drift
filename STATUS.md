# Drift — Project Status

**Last updated:** 2026-06-12 17:30:58 EDT

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
- Status as of 2026-06-12: **Phases 0–3 implemented and verified; Gate 1 in progress.**
  - Phase 0 done: `mlx_hidden.py`, `/v1/hidden`, `probe-hidden` CLI. 32B all-layer
    last-token capture works (64x5120, ~5s prefill server-resident).
  - Phase 1 done (infra): `hidden_probe` rule type with class distributions and
    confidence-gated escalation, shared single prefill in executor, compiler codegen,
    probe trainer with in-fold standardization folded into exported weights.
  - Phase 2/3 done (infra): layer-sweep script + L* selection; `max_layer` truncated
    prefill verified equal to full-pass prefix.
  - Phase 5/6 cores done offline: concept-bottleneck programs (`concepts.py`,
    `concept_program` rule type), soft-label distillation (`distill.py`, ECE).
  - **Gate 1 interim (n=100, helpdesk triage)**: at layer 48/64, `severity` probe
    reaches **75.8% coverage @ 0.95 confidence, 93.3% acc-on-covered** vs BoW 7.1%
    coverage — clean pass. 7–10-class fields (`root_cause_category`,
    `assignment_group`) are data-starved at n=100 (~10/class vs 5120 dims); corpus
    extension to 500 tickets capturing now (~3h). Final Gate 1 readout after that.
  - Corpus: `results/hidden_probe_corpus/` (labels + all-layer hidden states,
    Console-AI helpdesk dataset re-fetched to `datasets/helpdesk_tickets.jsonl`).
  - Known issue: standardized probes are overconfident on multi-class fields
    (acc-on-covered < threshold); temperature calibration lands in Phase 6.

## Housekeeping

- 2026-06-11: design docs consolidated into `docs/design/` (ATTENTION_PLAN.md,
  ANSIBLE_TRIAGE_PLAN.md, CONDITIONAL_CLASSIFIERS.md moved; references updated in
  CLAUDE.md and docs/HYBRID.md).
