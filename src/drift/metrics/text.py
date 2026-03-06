from __future__ import annotations

from itertools import combinations

import numpy as np
from Levenshtein import distance as levenshtein_distance


def exact_match_rate(outputs: list[str]) -> float:
    """Fraction of outputs identical to the first output."""
    if not outputs:
        return 0.0
    reference = outputs[0]
    matches = sum(1 for o in outputs if o == reference)
    return matches / len(outputs)


def normalized_edit_distance(a: str, b: str) -> float:
    """Levenshtein distance normalized by max length. Returns 0.0 for identical strings."""
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    return levenshtein_distance(a, b) / max_len


def pairwise_edit_distances(outputs: list[str]) -> list[float]:
    """Normalized edit distances for all pairs."""
    return [normalized_edit_distance(a, b) for a, b in combinations(outputs, 2)]


def output_length_cv(outputs: list[str]) -> float:
    """Coefficient of variation of output lengths."""
    lengths = [len(o) for o in outputs]
    mean = np.mean(lengths)
    if mean == 0:
        return 0.0
    return float(np.std(lengths) / mean)


def lcs_ratio(a: str, b: str) -> float:
    """Length of longest common subsequence / max length."""
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    # Space-optimized LCS
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n] / max(m, n)


def compute_all(outputs: list[str]) -> dict:
    """Compute all text metrics for a list of outputs."""
    edit_dists = pairwise_edit_distances(outputs)
    return {
        "exact_match_rate": exact_match_rate(outputs),
        "mean_edit_distance": float(np.mean(edit_dists)) if edit_dists else 0.0,
        "median_edit_distance": float(np.median(edit_dists)) if edit_dists else 0.0,
        "max_edit_distance": float(np.max(edit_dists)) if edit_dists else 0.0,
        "length_cv": output_length_cv(outputs),
        "num_outputs": len(outputs),
        "num_unique": len(set(outputs)),
    }
