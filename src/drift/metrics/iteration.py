from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from drift.metrics.text import compute_all


def compute_iteration_metrics(runs: list[dict], step_idx: int) -> dict:
    """Compute drift metrics for a loop step across runs."""
    loop_data = [r["steps"][step_idx] for r in runs]
    counts = [d["total_iterations"] for d in loop_data]
    reasons = [d["termination_reason"] for d in loop_data]

    # Iteration count distribution
    mean_count = float(np.mean(counts))
    iter_metrics = {
        "iteration_counts": counts,
        "mean_iterations": round(mean_count, 2),
        "min_iterations": int(np.min(counts)),
        "max_iterations": int(np.max(counts)),
        "iteration_count_cv": round(float(np.std(counts) / mean_count), 4) if mean_count > 0 else 0.0,
    }

    # Termination agreement
    mode_count = int(sp_stats.mode(counts, keepdims=False).mode)
    iter_metrics["mode_iterations"] = mode_count
    iter_metrics["termination_agreement_rate"] = round(
        sum(1 for c in counts if c == mode_count) / len(counts), 4
    )
    iter_metrics["termination_reasons"] = {
        reason: reasons.count(reason) for reason in set(reasons)
    }

    # Per-iteration, per-inner-step text metrics (only across runs that reached that iteration)
    max_iter = max(counts)
    inner_step_names = [s["name"] for s in loop_data[0]["iterations"][0]]
    per_iteration = []

    for it in range(max_iter):
        # Filter to runs that reached this iteration
        reaching_runs = [d for d in loop_data if d["total_iterations"] > it]
        n_runs = len(reaching_runs)

        inner_step_metrics = []
        for si, step_name in enumerate(inner_step_names):
            outputs = [d["iterations"][it][si]["response"]["content"] for d in reaching_runs]
            sm = compute_all(outputs)
            sm["name"] = step_name
            sm["model"] = reaching_runs[0]["iterations"][it][si]["response"]["model"]
            sm["provider"] = reaching_runs[0]["iterations"][it][si]["response"]["provider"]
            timing = [d["iterations"][it][si]["response"]["total_ms"] for d in reaching_runs]
            sm["mean_latency_ms"] = round(sum(timing) / len(timing), 1)
            inner_step_metrics.append(sm)

        per_iteration.append({
            "iteration": it + 1,
            "num_runs_reaching": n_runs,
            "steps": inner_step_metrics,
        })

    iter_metrics["per_iteration"] = per_iteration

    # Overall text metrics using the last iteration's last step from each run
    final_outputs = []
    for d in loop_data:
        last_iter = d["iterations"][-1]
        final_outputs.append(last_iter[-1]["response"]["content"])
    iter_metrics.update(compute_all(final_outputs))

    return iter_metrics
