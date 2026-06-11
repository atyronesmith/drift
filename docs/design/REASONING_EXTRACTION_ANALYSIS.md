# Reasoning Extraction: Analysis & Research Direction

**Date:** 2026-06-11 12:11:08 EDT
**Status:** Analysis complete; implementation plan in [HIDDEN_STATE_PROBES_PLAN.md](HIDDEN_STATE_PROBES_PLAN.md)

## Goal

Extend drift beyond extracting *deterministic* structure (regex, literals, mappings) to
extracting *reasoning logic* — using hidden states and full probability distributions from
the model to construct code that replaces more of the LLM. The generated code may operate
probabilistically; that is acceptable and in fact desirable (calibrated outputs, principled
escalation).

## Where the project is today

The synthesis ladder (`src/drift/hybrid/autogen.py`):

| Tier | Mechanism | Signal used | Runtime cost |
|------|-----------|-------------|--------------|
| 1 | regex / literal | token-level entropy from logprobs, traced to input | ~0 |
| 2 | mapping / conditional | cross-probe output consistency + attention-density hints | ~0 |
| 3 | logistic regression classifier | bag-of-words + structured one-hot features | ~0 |
| — | LLM residual | everything above entropy threshold | full decode |

Key existing infrastructure (all verified in source):

- **MLX server** (`src/drift/server/mlx_server.py`) — owns the model in-process; endpoints
  `/v1/chat/completions`, `/v1/logprobs` (full-vocab exact entropy), `/v1/attention`.
- **Attention probing** (`src/drift/probing/mlx_attention.py`) — two-phase: generate with
  fused SDPA, then replay the full sequence with a patched `Attention.__call__` to capture
  weights. Proves we can intercept the forward pass on Apple Silicon.
- **Classifier rule runtime** (`src/drift/hybrid/rules.py::_classifier_rule`) — pure-Python
  logistic regression with softmax, confidence threshold, and escalation hints
  (`escalation_hint`, `escalation_confidence`) when below threshold.
- **Drift monitoring** (`src/drift/safety/monitor.py`, `hybrid/calibration.py`) — JSD-based
  runtime monitoring of classifier confidence distributions against a training-time baseline.
- **Residual model test** (`scripts/residual_model_test.py`) — measures whether a small model
  (qwen3:4b) can handle the remaining LLM fields as well as qwen2.5:32b.

## The gap (named in THEORY.md)

The current system is a *degenerate two-level hierarchy*: a trivial model (bag-of-words
logistic regression) and a full-scale model, with nothing between. The Tier 3 classifier hits
a ~25% coverage ceiling at high confidence because keyword features run out of explanatory
power. Everything contextual — disambiguation, pragmatic inference, domain reasoning — lives
in the layers of the full model and is currently unreachable.

## Finding 1: The prefill/decode asymmetry is the architectural unlock

Hidden states are computed *by* the model, so code consuming them at runtime has not
eliminated the LLM. But it transforms the cost model:

- **Today's cost**: full autoregressive decode — hundreds of *sequential* forward passes to
  emit the output JSON.
- **Probe cost**: **one prefill pass** over the input. Fully parallel, no decode loop, and it
  yields hidden states at every position simultaneously.

The target artifact is therefore: *one prefill pass + k linear probes (matmuls) fills k
decision fields at once*. Even with the full 32B model that is roughly a 10–50x latency
reduction for tasks like helpdesk triage. Three escalating ways to shrink the prefill itself:

1. **Layer truncation** — if a field is linearly decodable at layer 20 of 64 with target
   accuracy, only layers 0–20 ever need to run at inference.
2. **Cross-model probe transfer** — train a probe on a small model's hidden states against
   the large model's output labels. If it matches, 32B reasoning has been distilled into
   4B-prefill + matmul.
3. **Full distillation** — use the probe as a teacher for a model that runs on raw text.

## Finding 2: Hidden-state probes should break the Tier 3 ceiling

A linear probe on the residual stream at the last input token sees the model's *own computed
representation* — after contextual disambiguation, coreference, and domain inference have
already happened. The interpretability literature is consistent: decisions are linearly
decodable from mid-layer residual streams well before the output token, often near-perfectly
on narrow tasks. The Tier 3 ceiling exists because keyword features saturate; hidden-state
features should not, for classification-shaped fields. This is the cheapest high-confidence
experiment available: same training harness as `_discover_classifier_rules`, different
feature matrix.

## Finding 3: Reasoning structure via concept-bottleneck programs

A direct probe is a black-box linear readout. To get code that *reads* as reasoning logic,
decompose: define the latent variables a human would use ("is the user locked out", "is this
hardware", "is the requester a VIP"), train one small probe per concept, then fit an
interpretable program (shallow decision tree, weighted rules) over the probed concept values.
Generated code then literally reads:

```python
p_lockout  = sigmoid(w_lockout @ h)    # probed at layer 18
p_hardware = sigmoid(w_hw @ h)
if p_lockout > 0.9 and p_hardware <= 0.5:
    assignment_group = "IdentityOps"   # learned branch, confidence 0.97
```

This is auditable in a way neither the LLM nor a monolithic probe is — directly relevant to
the safety/guard work.

## Finding 4: Full-distribution distillation

Today only top-5 logprobs (Ollama path) and hard labels (Tier 3 training) are used. The MLX
server already computes the *entire* softmax (`/v1/logprobs` reports exact entropy). Distill
with soft labels (KL against the teacher distribution at decision tokens — Hinton-style):

- Students trained on soft labels are markedly better calibrated → confidence-gated
  escalation becomes principled instead of heuristic.
- The teacher distribution at decision tokens is a far better drift baseline than the current
  JSD-over-student-confidence in `safety/monitor.py`.

## Finding 5: The horizon — SAEs and circuit tracing

Sparse autoencoder features over the residual stream are near-binary and interpretable; rules
over SAE activations are very close to code. Anthropic's circuit-tracing/attribution-graph
work (2025) is the maximal version of this research goal: trace the circuit for a narrow task
and re-implement it as a small computation graph. Treat as the paper's framing horizon, not
the next step — SAE training is expensive and the per-task payoff over concept probes is
unproven.

## Limitations to design around

1. **Probes do not transfer across model versions.** A weights update invalidates every
   probe. The continuous-calibration/JSD infrastructure already anticipates exactly this
   failure mode — probe retraining slots into the same auto-retrain workflow.
2. **Generation fields won't fall to probes.** `summary` / `next_steps` are irreducibly
   sequential. Probes capture decisions, classifications, extractions. End state:
   *probes for decisions + small residual model for prose + escalation on low confidence*.
3. **Predictive ≠ causal.** A probe can decode a label from dataset correlations the model
   doesn't use. For engineering, held-out accuracy + drift monitoring suffices. For PAPER.md
   claims of "extracted reasoning," activation-patching validation (intervene on the probed
   direction, confirm the output flips) is required — and is philosophically consistent with
   THEORY.md's point that chain-of-thought is narrative, not recording: probes attack the
   actual forward-pass computation.
4. **Memory pressure.** A 32B model's residual stream is ~5K dims × layers × sequence
   length. Capture must be restricted to requested layers/positions (the attention probe
   already learned this lesson — it stores only top-k attended positions).

## Theory connection

The entropy floor (THEORY.md) told us which output tokens need *no* model. The probe-depth
profile (accuracy vs. layer, per field) tells us *how much* model each decision needs. This
extends the compression framing: each field has a "reasoning depth," and the goal is to pay
for exactly that depth and no more.
