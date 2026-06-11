# Implementation Plan: Hidden-State Probes & Reasoning Extraction (Tier 4+)

**Date:** 2026-06-11 12:11:08 EDT
**Status:** DRAFT — plan only, no implementation started
**Background:** [REASONING_EXTRACTION_ANALYSIS.md](REASONING_EXTRACTION_ANALYSIS.md)

## Summary

Seven phases, each independently shippable, each with a decision gate. Phases 0–1 are the
critical path: hidden-state capture and a Tier 4 `hidden_probe` rule type benchmarked against
the Tier 3 ~25% coverage ceiling. Everything after depends on Phase 1's result.

Methodology: TDD where prudent — each phase lists the failing tests to write first.
MLX-dependent tests are marked `@pytest.mark.integration` (require Apple Silicon + a small
model, e.g. `mlx-community/Qwen2.5-0.5B-Instruct-4bit`); pure-Python logic is unit-tested
with synthetic vectors.

---

## Phase 0 — Hidden-state capture infrastructure

**Goal:** Capture residual-stream vectors at chosen layers/positions, locally and over HTTP.
Prefill-only mode (no generation) is the headline feature.

### New: `src/drift/probing/mlx_hidden.py`

Mirror the structure of `mlx_attention.py`, but simpler — no patching of attention math
needed. Wrap each transformer block's `__call__` (or iterate `model.model.layers` manually
after the embedding) and record the block output for requested layers only.

```python
@dataclass
class HiddenStateResult:
    model: str
    n_input: int
    layers: list[int]                  # which layers were captured
    positions: list[int]               # which sequence positions were captured
    # states[layer_idx][position_idx] -> list[float] (hidden_dim)
    states: dict[int, dict[int, list[float]]]
    input_tokens: list[str]
    content: str | None                # None in prefill-only mode

def probe_hidden_states(
    system: str, prompt: str,
    model: str = "mlx-community/Qwen2.5-32B-Instruct-4bit",
    layers: list[int] | str = "all",   # or explicit [12, 20, 31]
    positions: str | list[int] = "last_input",  # "last_input" | "all_input" | explicit
    prefill_only: bool = True,         # no decode loop
    max_tokens: int = 0,               # >0 enables generation mode (for training capture)
) -> HiddenStateResult: ...

def _probe_hidden_with_model(mlx_model, tokenizer, ...) -> HiddenStateResult:
    # shared with server endpoint, same pattern as _probe_attention_with_model
```

Implementation notes:
- Capture the block *output* (post-residual-add) in float32; convert to list only for
  requested (layer, position) pairs — never materialize full [layers × seq × dim].
- `positions="last_input"` is the default and the cheap path: one vector per layer.
- Generation mode (`max_tokens > 0`) reuses the two-phase replay trick from
  `mlx_attention.py`: generate normally, then one full-sequence forward pass with hooks.
  Needed in Phase 6 to capture teacher softmax at decision-token positions.
- `mx.metal.clear_cache()` discipline as in the server.

### Server: `/v1/hidden` endpoint in `src/drift/server/mlx_server.py`

- `POST /v1/hidden` → `_handle_hidden()`, delegating to `_probe_hidden_with_model`.
- Request: `{messages, layers, positions, prefill_only, max_tokens}`.
- Response: `HiddenStateResult` serialized; include `timing.total_ms` (this number IS the
  runtime cost claim for Tier 4 — report it everywhere).
- Register in the startup banner and in `do_POST` routing.

### Client: `probe_hidden_remote()` in `mlx_hidden.py`

curl-subprocess client (same endpoint-security constraint as other providers — see
CLAUDE.md: direct TCP from Python is blocked). Follow `providers/mlx_remote.py` conventions.

### CLI

`drift probe-hidden <task> --layers all --positions last_input -o hidden.json` in `cli.py`,
following the existing `probe` command's shape.

### Tests (write first)

- `tests/test_mlx_hidden.py::test_result_serialization_roundtrip` — unit; build a
  `HiddenStateResult` from synthetic data, serialize/deserialize, verify shapes.
- `tests/test_mlx_hidden.py::test_position_resolution` — unit; `"last_input"` resolves to
  `n_input - 1`, explicit lists pass through, out-of-range raises.
- `tests/test_mlx_hidden.py::test_capture_smoke` — integration; 0.5B model, 2 layers,
  last_input; assert vector dim == model hidden dim, no NaN, prefill-only returns
  `content=None`.
- `tests/test_mlx_server.py::test_hidden_endpoint` — integration; server roundtrip.

### Exit criteria

Prefill-only capture of last-input-token states at all layers on the 32B model in single-digit
seconds; memory stable across repeated requests.

---

## Phase 1 — Tier 4 `hidden_probe` rules (THE decision-gate phase)

**Goal:** Replace Tier 3's bag-of-words features with hidden-state features. Measure coverage
at the same confidence threshold against the ~25% ceiling on helpdesk triage.

### Training capture

- Extend the probe corpus flow: for each training input, call `/v1/hidden`
  (prefill-only, `layers` = sweep list or chosen layer, `positions="last_input"`).
- Storage: `results/<run>/hidden_states.npz` — arrays keyed `"{probe_idx}:{layer}"`,
  float16 on disk. A 60-probe × 64-layer × 5120-dim capture ≈ 40 MB; acceptable. Sidecar
  `hidden_states.meta.json` records model name, layer list, position spec, capture date.

### Synthesis: `_discover_hidden_probe_rules()` in `autogen.py` (or new `src/drift/hybrid/probes.py` if >200 lines)

Clone the `_discover_classifier_rules` harness:
- Same candidate selection (llm_required fields, 2–20 unique values, ≥20 probes).
- Feature matrix = hidden state at layer L (parameter; Phase 2 finds optimal L — start with
  ~60% depth, e.g. layer 38/64 for the 32B model).
- Same `LogisticRegression` + cross-validation; **add L2 strength sweep** (`C` grid) — with
  5120-dim features and ~60 samples, regularization is the difference between working and
  memorizing. Report `n_samples / n_features` in stats and warn below 0.05.
- Same `compute_baseline()` call for drift monitoring.
- Emit `GeneratedRule(rule_type="hidden_probe", ...)` with new fields:
  `layer: int`, `position: str`, `probe_model: str`, `weights_ref: str | None`.

### Weight storage

5120-dim × n_classes float weights are too big for readable YAML. Store as sidecar:
`<task>/probes/<field>.npz` with sha256 recorded in the YAML rule (`weights_ref`,
`weights_sha256`). `render_hybrid_yaml()` writes the ref; small probes (<64 dims, tests)
may inline as today.

### Runtime: `rules.py` + `executor.py`

- New `_hidden_probe_rule(path, rule, hidden_state)` in `rules.py`: dot product + softmax +
  threshold + escalation hints — identical contract to `_classifier_rule`, different feature
  source. **Returns the full class distribution** in a new `RuleResult.distribution` field
  (probabilistic operation is a feature, not a bug).
- `execute_rules()` gains an optional `hidden_states: dict[int, list[float]] | None`
  parameter (layer → vector). `execute_rule()` passes it through.
- `executor.run_hybrid()`: if any rule is `hidden_probe`, make **one** `/v1/hidden` call for
  the union of required layers before rule execution; all probe rules share it. New
  `hidden_provider` config (server URL + model). If the call fails → all probe rules return
  `matched=False` and escalate to LLM (graceful degradation, no new failure mode).
- `compiler.py`: standalone engines embed the npz path + the `/v1/hidden` client call.

### Benchmark: `scripts/hidden_probe_vs_tier3.py`

Head-to-head on helpdesk triage Tier 3 candidate fields
(`root_cause_category`, `assignment_group`, …):

| Metric | Tier 3 (BoW) | Tier 4 (hidden) |
|---|---|---|
| CV accuracy | | |
| **Coverage @ 0.95 confidence** | ~25% (known) | ? |
| ECE (calibration) | | |
| Runtime ms (rule eval + hidden fetch) | | |

### Tests (write first)

- `tests/test_rules_hidden_probe.py::test_probe_rule_softmax` — unit; 4-dim synthetic
  weights + vector, assert class, confidence, distribution sums to 1.
- `::test_probe_rule_below_threshold_escalates` — unit; assert `matched=False`,
  `escalation_hint` set, `escalation_confidence` < threshold.
- `::test_execute_rules_threads_hidden_states` — unit; fake hidden_states dict, mixed
  regex + hidden_probe rule list.
- `::test_missing_hidden_state_escalates` — unit; `hidden_states=None` → escalation, no crash.
- `tests/test_autogen_hidden.py::test_discover_hidden_probe_rules_synthetic` — unit;
  synthetic linearly-separable hidden states → rule emitted; shuffled labels → no rule.
- `tests/test_executor_hidden.py::test_single_hidden_fetch_for_multiple_probes` — unit;
  mock provider, assert exactly one `/v1/hidden` call for 3 probe rules.

### DECISION GATE 1

- Coverage @ 0.95 confidence ≥ 2× Tier 3 (i.e. ≥ 50%) on at least one field → proceed.
- Coverage ≤ Tier 3 + 10pts → stop; write up the negative result (it's still a paper
  section: "the model's representation of its own decision is not linearly accessible at
  this scale/quantization"), skip to Phase 6 (distillation has independent value).

---

## Phase 2 — Layer sweep: per-field reasoning-depth profiles

**Goal:** The accuracy-vs-layer curve per field — the central figure for PAPER.md and the
input to Phase 3.

### Work

- `scripts/probe_layer_sweep.py`: for each candidate field, train probes at every layer
  (capture is already all-layers from Phase 1), plot accuracy + coverage vs layer.
- Output `results/layer_sweep/<task>/<field>.json` + a markdown report table.
- Define **reasoning depth** `L*(field)` = earliest layer reaching ≥95% of the field's best
  cross-layer accuracy. Add `L*` to synthesis stats and choose each rule's `layer` from it
  (replaces Phase 1's fixed-layer guess).
- Extend THEORY.md with the entropy-floor → reasoning-depth connection (one section).

### Tests

- `tests/test_layer_sweep.py::test_lstar_selection` — unit; synthetic accuracy curves
  (monotone, plateau, noisy) → correct L*.

### Exit criteria

Depth profiles for all helpdesk fields; per-field layer choice wired into synthesis.

---

## Phase 3 — Truncated-prefill runtime

**Goal:** Stop the forward pass at `max(L*)` — pay for exactly the reasoning depth needed.

### Work

- `/v1/hidden` gains `max_layer: int` — the server runs embedding + layers `0..max_layer`
  only and skips the rest (loop over `model.model.layers[:max_layer+1]` manually; do not
  call the full model).
- Benchmark in `scripts/hidden_probe_vs_tier3.py`: full decode vs full prefill vs truncated
  prefill, wall-clock on the 32B model.
- Optional (only if memory pressure appears): a `--max-layer` server start flag that loads
  only the needed layer weights.

### Tests

- `tests/test_mlx_hidden.py::test_truncated_equals_full_prefix` — integration; states at
  layer k from a `max_layer=k` run match the full run (within float tolerance).

### Exit criteria

Latency table; expected shape: truncated prefill ≪ full prefill ≪ full decode.

---

## Phase 4 — Cross-model probe transfer

**Goal:** Probe a small model's hidden states to predict the *large* model's decisions.
Composes with `residual_model_test.py`.

### Work

- `scripts/probe_transfer_test.py`:
  1. Labels = qwen2.5:32b hybrid outputs on the training corpus (already exist in results).
  2. Features = hidden states from small candidates (Qwen3-4B, Qwen2.5-0.5B/1.5B MLX builds)
     on the same inputs, layer-swept.
  3. Train/evaluate probes; compare against (a) 32B self-probes from Phase 1, (b) the
     prompted small model from `residual_model_test.py`.
- Rule emission: `probe_model` field already exists from Phase 1 — a transfer rule is just a
  `hidden_probe` rule whose `probe_model` is the small model; executor's hidden_provider
  config points at a small-model server instance. No new rule type.
- Decision matrix in the report: per field, cheapest runtime configuration meeting the
  accuracy bar (32B-truncated vs 4B-full vs 4B-truncated).

### Tests

- Reuses Phase 1 unit tests (model name is just config). Integration smoke with 0.5B.

### DECISION GATE 4

Transfer probe ≥ prompted-small-model agreement AND ≥ Tier 3 → small-model prefill becomes
the default Tier 4 runtime; else stay on truncated large model.

---

## Phase 5 — Concept-bottleneck programs (`concept_program` rule type)

**Goal:** Interpretable reasoning logic: per-concept probes + a shallow decision program.

### Work

- **Concept spec**: `tasks/<name>/concepts.yaml` — concept name + natural-language
  definition + labeling prompt. Concepts are labeled on the training corpus by the large
  model (one cheap batched call per concept; labels cached to
  `results/<run>/concept_labels.json`).
- **Training** (`src/drift/hybrid/concepts.py`):
  1. Per-concept binary probe at that concept's L* (logistic regression on hidden states).
  2. Decision program over concept probabilities: sklearn `DecisionTreeClassifier`
     (max_depth ≤ 4) fit on probed (not gold) concept values, so the program sees
     runtime-realistic inputs.
  3. Export: tree serialized as nested `{if: {concept, op, threshold}, then:, else:}` dicts
     in YAML — human-readable and auditable.
- **Runtime** in `rules.py`: `_concept_program_rule()` — evaluate concept probes (sharing
  the Phase 1 hidden-state fetch), walk the tree, combine leaf purity × concept confidence
  into an escalation-capable confidence score.
- **Evaluation**: accuracy delta vs direct Phase 1 probe on the same fields = the measured
  *price of interpretability*. Report both.

### Tests (write first)

- `tests/test_concepts.py::test_tree_export_roundtrip` — unit; fit on synthetic concepts,
  export, re-execute exported form, identical predictions.
- `::test_concept_program_rule_eval` — unit; hand-built 2-concept tree + fake hidden state.
- `::test_low_concept_confidence_escalates` — unit.

### Exit criteria

One task fully expressed as concept programs; accuracy within 5pts of direct probes OR an
honest writeup of the gap.

---

## Phase 6 — Full-distribution distillation & calibration (independent of Gates 1/4)

**Goal:** Use the teacher's full softmax at decision tokens for soft-label training and
better drift baselines.

### Work

- **Capture**: extend `mlx_logprobs.py` to optionally store the full (or top-200 renormalized)
  distribution at *decision-token positions* (first token of each JSON field value —
  identify via the existing `_map_tokens_to_chars` machinery in `synthesizer.py`).
  Store per-field teacher distributions in probe JSON.
- **Soft-label training**: small numpy softmax-regression trainer
  (`src/drift/hybrid/distill.py`) minimizing cross-entropy against soft targets, with
  temperature parameter. (sklearn can't do soft targets; ~80 lines of numpy, full-batch
  gradient descent, unit-testable.) Applies to both Tier 3 (BoW) and Tier 4 (hidden)
  feature sets.
- **Calibration evaluation**: ECE + reliability data added to the benchmark script for hard
  vs soft-label variants of every classifier/probe.
- **Monitor upgrade**: `safety/monitor.py` baseline becomes the *teacher* distribution
  (per-class JSD between student runtime distribution and teacher training distribution),
  replacing student-self-confidence baselines where teacher data exists.

### Tests (write first)

- `tests/test_distill.py::test_soft_targets_recover_known_distribution` — unit; synthetic
  teacher, assert student KL → ~0 on training data.
- `::test_temperature_scaling_monotone_entropy` — unit.
- `::test_ece_computation` — unit; hand-computed ECE on a toy set.
- `tests/test_monitor_teacher_baseline.py` — unit; JSD against teacher baseline flags a
  shifted distribution, passes an unshifted one.

### Exit criteria

Soft-label students show lower ECE at equal accuracy; monitor consumes teacher baselines.

---

## Phase 7 — Causal validation & writeup

**Goal:** Support the "extracted reasoning" claim in PAPER.md.

### Work

- `scripts/activation_patching_test.py`: for 2–3 high-performing probes, ablate/steer along
  the probe direction at layer L* during generation (reuse the attention-probe patching
  pattern, but on block outputs) and measure output flip rate. Probe direction causally
  drives the output ≥ X% of the time → "extracted reasoning logic" is defensible; otherwise
  claim "decision-predictive representations" only.
- PAPER.md: new section (prefill/decode asymmetry, depth profiles, coverage results,
  causal validation). THEORY.md: reasoning-depth section (from Phase 2). RESULTS.md: new
  experiment entries per phase. Update CLAUDE.md architecture section with new modules.

---

## Risks

| Risk | Mitigation |
|---|---|
| n≪d overfitting (60 samples, 5120 dims) | C-grid + CV (Phase 1); grow probe corpus before declaring Gate 1 failure |
| 4-bit quantization noise in hidden states | If Gate 1 marginal, rerun capture with 8-bit model before concluding |
| Hidden-state capture memory blowup | Only requested (layer, position) pairs ever materialized; float16 on disk |
| Probe invalidation on model update | `probe_model` + sha256 recorded in rules; monitor JSD triggers retrain (existing CONTINUOUS_CALIBRATION workflow) |
| MLX server is single-host / Apple-only | Acceptable for research; production story = Phase 4 small models or distillation (Phase 6) |
| Concept labels from the teacher are noisy | Tree fit on probed values absorbs some noise; report concept-probe CV accuracy separately |

## Effort estimate

| Phase | Estimate | Depends on |
|---|---|---|
| 0 — capture + endpoint | 2–3 days | — |
| 1 — Tier 4 rules + benchmark | 3–5 days | 0 |
| 2 — layer sweep | 1–2 days | 1 |
| 3 — truncated prefill | 1–2 days | 2 |
| 4 — cross-model transfer | 2–3 days | 1 (gate) |
| 5 — concept programs | 4–6 days | 1 (gate), ideally 2 |
| 6 — distillation + calibration | 3–4 days | 0 only (parallelizable) |
| 7 — causal validation + writeup | 3–4 days | any subset |

Critical path to the go/no-go answer (Phases 0–1): **about one week.**
