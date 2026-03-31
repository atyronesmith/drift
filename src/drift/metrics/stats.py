"""Bootstrap confidence intervals and significance tests for drift metrics."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: list[float | int | bool],
    stat_fn=np.mean,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute a bootstrap confidence interval for a statistic.

    Returns dict with keys: estimate, ci_lower, ci_upper, ci_level.
    """
    arr = np.array(values, dtype=float)
    if len(arr) < 2:
        val = float(stat_fn(arr)) if len(arr) == 1 else None
        return {"estimate": val, "ci_lower": None, "ci_upper": None, "ci_level": ci}

    rng = np.random.default_rng(seed)
    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_stats.append(float(stat_fn(sample)))

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_stats, alpha * 100))
    upper = float(np.percentile(boot_stats, (1 - alpha) * 100))

    return {
        "estimate": float(stat_fn(arr)),
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_level": ci,
    }


def permutation_test(
    a: list[float | int | bool],
    b: list[float | int | bool],
    stat_fn=np.mean,
    n_perm: int = 10000,
    seed: int = 42,
) -> dict:
    """Two-sample permutation test for difference in means (or other stat).

    Returns dict with keys: observed_diff, p_value, significant (at 0.05).
    """
    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)

    observed = float(stat_fn(arr_b) - stat_fn(arr_a))
    combined = np.concatenate([arr_a, arr_b])
    n_a = len(arr_a)

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_diff = float(stat_fn(combined[n_a:]) - stat_fn(combined[:n_a]))
        if abs(perm_diff) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_perm + 1)  # +1 to include observed

    return {
        "observed_diff": observed,
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
    }
