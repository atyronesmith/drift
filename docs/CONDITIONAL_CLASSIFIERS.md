# Conditional Classifiers and Continuous Calibration

## Background

Reviewed a hierarchical classifier decomposition spec (April 2026). Rejected the tree architecture — the useful kernel is that Tier 3 classifier accuracy improves when conditioned on other tier outputs. This is an extension of the existing tier system, not a new architecture.

## Three Items to Build

### 1. Conditional Classifiers (drift)

Extend Tier 3 classifier training so feature vectors include Tier 1/2 extracted values.

**Current state:** `autogen.py` `_discover_classifier_rules()` builds bag-of-words + structured one-hot features from raw input text. Tier 1/2 outputs are not used as features.

**Change:** After Tier 1/2 rules are discovered, re-run classifier training with an expanded feature matrix that includes extracted field values. If a field like `severity` is already extracted by regex, a classifier for `priority` can use it as a feature — capturing the correlation without building a tree.

**Where:**
- `src/drift/hybrid/autogen.py` — extend `_discover_classifier_rules()` to accept prior tier results as features
- `src/drift/hybrid/compiler.py` — compiled engines must execute Tier 1/2 before Tier 3 (already true) and pass results into classifier feature construction

**Validation:** Compare classifier accuracy with and without conditional features on existing tasks (incident_triage, alert_triage have correlated fields).

### 2. Drift Detection (drift trains, denoise monitors)

Monitor classifier confidence distributions at runtime. Trigger recalibration when distributions shift.

**Metric:** Jensen-Shannon divergence between training-time confidence distribution and a rolling window of runtime predictions.

**Where:**
- **drift** — `src/drift/hybrid/calibration.py` — compute and serialize baseline confidence distributions during training. Provide retraining pipeline entry point.
- **denoise** — `src/denoise/monitor.py` — collect runtime predictions, compute JSD, trigger recalibration. Supports `auto_retrain` flag (retrain automatically vs. flag for review).

**Thresholds (starting points, empirically tuned):**
- JSD > 0.1 — warning, log drift detected
- JSD > 0.2 — trigger recalibration (auto or manual per flag)
- Rolling window: 1000 predictions minimum before computing JSD

### 3. Escalation Enrichment (both repos)

When a classifier escalates to LLM (confidence below threshold), pass the partial extraction as structured context so the LLM doesn't start from scratch.

**Where:**
- **drift** — `src/drift/hybrid/executor.py` — include extracted fields in the LLM prompt for escalated cases
- **drift** — `src/drift/hybrid/compiler.py` — compiled engines pass partial results to LLM call
- **denoise** — `src/denoise/guard.py` — on escalation, return partial classification context alongside the `("escalate", confidence)` tuple

## Implementation Order

1. ~~Conditional classifiers~~ — **Done**. `autogen.py` accepts resolved_paths (Tier 1+2) for classifier features.
2. ~~Escalation enrichment~~ — **Done**. `rules.py` preserves tentative predictions on escalation. `executor.py` and `compiler.py` include hints in LLM prompt.
3. ~~Drift detection~~ — **Done**. `calibration.py` computes baseline distributions during training. `denoise/monitor.py` provides `ClassifierMonitor` with JSD drift detection.
