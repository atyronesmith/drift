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
  (see docs/THEORY.md "Beyond the Entropy Floor" and Gate 1 results below). Vocabulary-poor
  fields (`root_cause`) still need hidden-state probes or LLM fallback.
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
  - **Gate 1 final (n=500, helpdesk triage, layer 48, threshold 0.95)**:
    formally NOT passed, but the stop condition is not triggered either. Coverage
    (hidden vs BoW): severity 89.8% vs 80.6%, root_cause 34.1% vs 1.4% (24x),
    assignment_group 49.3% vs 44.1%. BoW CV accuracy beats hidden on all fields
    at n=500 (0.97/0.74/0.82 vs 0.93/0.65/0.70).
  - **Headline finding 1 — the Tier 3 "25% ceiling" was sample starvation, not
    feature poverty.** With 500 labels, plain BoW logistic regression reaches
    80% coverage on severity. The hidden-probe advantage is real only where
    vocabulary doesn't carry the signal (root_cause: 24x coverage).
  - **Headline finding 2 — cross-model transfer works.** Probes on Qwen2.5-0.5B
    hidden states (layers 4–12 of 24) predict the 32B's decisions at parity with
    the 32B's own n=100 self-probes (severity 0.846, root_cause 0.561,
    assignment 0.591). Runtime: ~80ms 0.5B prefill vs 21.5s 32B full decode
    (~270x). Multi-class fields appear ceiling-limited by 32B label
    inconsistency (drift), not representation quality.
  - **Latency (32B server, 390-token ticket)**: full decode 21.5s; full prefill
    5.3s; truncated prefill max_layer=24 2.2s; 0.5B prefill 0.08s.
  - Corpus: `results/hidden_probe_corpus/` — 500 labels (0 parse failures) +
    all-layer hidden states for 32B and 0.5B. Gate reports: gate1_layer48.json,
    transfer_hidden_Qwen2.5-0.5B-Instruct-4bit.json; sweep: results/layer_sweep/.
  - Next decisions: (a) BoW+hidden ensemble features (likely strictly dominates
    both); (b) prompted-small-model arm of Gate 4 (needs qwen3:4b via Ollama);
    (c) Phase 6 temperature calibration (probes overconfident on multi-class);
    (d) Phase 7 writeup reframing: probes win on latency and vocabulary-free
    signal, BoW wins on sample efficiency at fixed n.

## Housekeeping

- 2026-06-11: design docs consolidated into `docs/design/` (ATTENTION_PLAN.md,
  ANSIBLE_TRIAGE_PLAN.md, CONDITIONAL_CLASSIFIERS.md moved; references updated in
  CLAUDE.md and docs/HYBRID.md).
