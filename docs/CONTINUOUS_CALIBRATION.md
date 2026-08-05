# Continuous Calibration

## Workflow

1. **Bootstrap:** Run an initial corpus through drift's training pipeline (`drift distill`) to produce a first set of classifier weights.
2. **Runtime collection:** During inference, `drift.safety.monitor.ClassifierMonitor` logs every classifier prediction (verdict, confidence, feature hash) to a rolling buffer.
3. **Drift detection:** Periodically compute Jensen-Shannon divergence between the training-time confidence distribution and the runtime buffer. If JSD exceeds threshold, the model needs updating.
4. **Recalibration:** Either retrain automatically or flag for review, controlled by a configuration flag.

## Architecture

### `src/drift/safety/monitor.py`

Collects runtime prediction data and detects distribution drift.

```python
monitor = ClassifierMonitor(
    baseline_path="weights/baseline_distribution.json",  # from training
    window_size=1000,         # min predictions before computing JSD
    warn_threshold=0.1,       # log warning
    retrain_threshold=0.2,    # trigger recalibration
    auto_retrain=False,       # True = retrain without asking
)

# Called on every classification
monitor.record(verdict="safe", confidence=0.94, features={...})

# Periodic check (called by application or on schedule)
status = monitor.check()
# status.drifted: bool
# status.jsd: float
# status.recommendation: "ok" | "warn" | "retrain"
```

### Recalibration modes

- `auto_retrain=False` (default) — returns `"retrain"` recommendation, application decides what to do
- `auto_retrain=True` — triggers retraining pipeline automatically via callback

### Integration

Training, monitoring, and drift detection all live in drift now (`src/drift/safety/`). The handoff:

1. `ClassifierMonitor` detects JSD > threshold
2. `ClassifierMonitor.export_samples()` exports collected runtime samples to a directory
3. drift's `distill` command runs on the expanded corpus
4. New weights are written back to the classifier's weights path
5. `TieredGuard`/`ClassifierMonitor` hot-reload weights on next classification call

### Thresholds

Starting points — these are empirically tuned per domain:

| Threshold | Value | Meaning |
|---|---|---|
| `warn_threshold` | 0.1 | Log drift, no action |
| `retrain_threshold` | 0.2 | Trigger recalibration |
| `window_size` | 1000 | Min samples before JSD is meaningful |

## What drift owns

Guard classification and drift monitoring live entirely in drift's `src/drift/safety/` (moved here from logdigest — see `CLAUDE.md`). logdigest is now scoped to pre-digestion only; it doesn't own any part of the classifier/monitoring pipeline.

| Concern | Owner |
|---|---|
| Runtime prediction logging | drift (`src/drift/safety/monitor.py`) |
| JSD computation | drift (`src/drift/safety/monitor.py`) |
| Drift detection | drift (`src/drift/safety/monitor.py`) |
| Sample export for retraining | drift (`src/drift/safety/monitor.py`) |
| Classifier training / retraining | drift (`scripts/train_guard_classifier.py`) |
| Baseline distribution generation | drift (`src/drift/hybrid/calibration.py`) |
| Weight serialization format | drift |

## Implementation Status

- `src/drift/safety/monitor.py` — `ClassifierMonitor` with `record()`, `check()`, `export_samples()`, `reset()`
- `tests/test_monitor.py` — 15 tests covering JSD, thresholds, callbacks, export, reset
- `src/drift/hybrid/calibration.py` computes baselines; `GeneratedRule.baseline_distribution` carries them through training
