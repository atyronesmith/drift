"""Baseline confidence distribution computation for classifier drift detection.

During training (drift distill), computes per-classifier confidence
distributions and serializes them as baselines. At runtime,
drift.safety.monitor.ClassifierMonitor compares live predictions
against these baselines using Jensen-Shannon divergence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def compute_baseline(
    weights: list[list[float]],
    bias: list[float],
    classes: list[str],
    vocabulary: list[str],
    structured_features: list[dict] | None,
    training_texts: list[str],
    training_extracted: list[dict],
    n_bins: int = 20,
) -> dict:
    """Compute baseline confidence distribution for a trained classifier.

    Runs the classifier over all training inputs and builds a histogram
    of top-class confidence values. This histogram is the baseline that
    drift.safety.monitor monitors for drift via JSD.

    Args:
        weights: Classifier weight matrix [n_classes x n_features]
        bias: Bias vector [n_classes]
        classes: Class labels
        vocabulary: BoW vocabulary
        structured_features: Structured feature definitions
        training_texts: Input texts used during training
        training_extracted: Extracted field dicts for each training input
        n_bins: Number of histogram bins (uniform over [0, 1])

    Returns:
        Dict with histogram, class_distribution, and metadata
    """
    import re

    structured_features = structured_features or []
    n_vocab = len(vocabulary)
    n_sf = sum(len(sf["categories"]) for sf in structured_features)
    n_features = n_vocab + n_sf
    vocab_index = {w: i for i, w in enumerate(vocabulary)}

    confidences = []
    class_counts: dict[str, int] = {c: 0 for c in classes}

    for text, extracted in zip(training_texts, training_extracted):
        # Build feature vector (same logic as rules.py _classifier_rule)
        features = [0.0] * n_features
        for w in re.findall(r"[a-z]+", text.lower()):
            if w in vocab_index:
                features[vocab_index[w]] = 1.0

        offset = n_vocab
        for sf in structured_features:
            source_val = _get_nested(extracted, sf["source"])
            for j, cat in enumerate(sf["categories"]):
                if str(source_val) == str(cat):
                    features[offset + j] = 1.0
            offset += len(sf["categories"])

        # Compute softmax probabilities
        n_classes = len(classes)
        logits = [0.0] * n_classes
        for c in range(n_classes):
            s = bias[c]
            for j, fval in enumerate(features):
                if fval != 0:
                    s += weights[c][j] * fval
            logits[c] = s

        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        total = sum(exp_logits)
        probs = [e / total for e in exp_logits]

        best_idx = max(range(n_classes), key=lambda i: probs[i])
        confidences.append(probs[best_idx])
        class_counts[classes[best_idx]] += 1

    # Build histogram
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    histogram = [0] * n_bins
    for conf in confidences:
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        histogram[bin_idx] += 1

    # Normalize to probabilities
    n_total = len(confidences)
    hist_probs = [h / n_total for h in histogram] if n_total > 0 else histogram

    return {
        "histogram": hist_probs,
        "bin_edges": bin_edges,
        "n_bins": n_bins,
        "n_samples": n_total,
        "class_distribution": class_counts,
        "mean_confidence": sum(confidences) / n_total if n_total else 0.0,
        "classes": classes,
    }


def save_baseline(baseline: dict, path: str | Path) -> None:
    """Write baseline distribution to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2))


def load_baseline(path: str | Path) -> dict:
    """Load baseline distribution from a JSON file."""
    return json.loads(Path(path).read_text())


def compute_jsd(p: list[float], q: list[float]) -> float:
    """Compute Jensen-Shannon divergence between two distributions.

    JSD is symmetric and bounded [0, 1] (using log base 2).
    Returns 0 for identical distributions, 1 for maximally different.
    """
    if len(p) != len(q):
        raise ValueError(f"Distribution lengths differ: {len(p)} vs {len(q)}")

    n = len(p)
    m = [(p[i] + q[i]) / 2 for i in range(n)]

    def _kl(a: list[float], b: list[float]) -> float:
        total = 0.0
        for i in range(n):
            if a[i] > 0 and b[i] > 0:
                total += a[i] * math.log2(a[i] / b[i])
        return total

    return (_kl(p, m) + _kl(q, m)) / 2


def _get_nested(obj: dict, path: str):
    """Get a value from a nested dict using dot notation."""
    for key in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return None
    return obj
