"""Hidden-state probe training (Tier 4).

Trains logistic-regression probes on residual-stream vectors captured by
drift.probing.mlx_hidden. With n_samples << d_model, L2 strength is the
difference between generalizing and memorizing — the C grid is swept and
chosen by cross-validated accuracy.

Requires scikit-learn (optional dependency, same as Tier 3 classifiers).
"""

from __future__ import annotations

import numpy as np


def train_hidden_probe(
    X: np.ndarray,
    y: np.ndarray,
    layer: int,
    probe_model: str,
    position: str = "last_input",
    c_grid: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0),
    cv: int = 5,
    min_accuracy: float = 0.0,
    min_class_count: int = 3,
    seed: int = 42,
) -> dict | None:
    """Train one logistic probe on hidden states for one output field.

    Args:
        X: [n, d_model] hidden-state features (float).
        y: [n] string labels.
        layer: which layer X came from (recorded in the rule).
        probe_model: model ID the hidden states came from.
        c_grid: inverse-regularization values to sweep.
        cv: cross-validation folds (capped by smallest class count).
        min_accuracy: return None if best CV accuracy is below this.
        min_class_count: drop classes with fewer samples than this.

    Returns a dict with rule fields (type/layer/position/classes/weights/
    bias/probe_model), plus training metadata: cv_accuracy, best_C,
    oof_probs (out-of-fold probabilities for honest coverage estimates),
    oof_y (labels aligned to oof_probs), n_samples. None if untrainable.
    """
    from collections import Counter

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

    y = np.asarray(y)
    X = np.asarray(X, dtype=np.float32)

    counts = Counter(y)
    rare = {c for c, n in counts.items() if n < min_class_count}
    if rare:
        keep = np.array([yi not in rare for yi in y])
        X, y = X[keep], y[keep]
        counts = Counter(y)

    classes = sorted(counts)
    if len(classes) < 2 or len(y) < 10:
        return None

    n_splits = min(cv, min(counts.values()))
    if n_splits < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    best_c, best_acc = None, -1.0
    for c in c_grid:
        lr = LogisticRegression(max_iter=2000, C=c, random_state=seed)
        acc = cross_val_score(lr, X, y, cv=skf, scoring="accuracy").mean()
        if acc > best_acc:
            best_acc, best_c = acc, c

    if best_acc < min_accuracy:
        return None

    lr = LogisticRegression(max_iter=2000, C=best_c, random_state=seed)
    oof_probs = cross_val_predict(lr, X, y, cv=skf, method="predict_proba")

    lr.fit(X, y)

    return {
        "type": "hidden_probe",
        "layer": layer,
        "position": position,
        "classes": list(lr.classes_),
        "weights": lr.coef_.tolist() if len(lr.classes_) > 2 else _binary_to_per_class(lr),
        "bias": lr.intercept_.tolist() if len(lr.classes_) > 2 else _binary_bias_to_per_class(lr),
        "probe_model": probe_model,
        "cv_accuracy": round(float(best_acc), 4),
        "best_C": best_c,
        "oof_probs": oof_probs,
        "oof_y": y,
        "n_samples": len(y),
    }


def _binary_to_per_class(lr) -> list[list[float]]:
    """sklearn binary LR stores one weight row (for classes_[1]).

    The hidden_probe runtime computes per-class logits + softmax, so expand
    to two rows: [-w/2, +w/2] reproduces the same decision and probabilities.
    """
    w = lr.coef_[0]
    return [(-w / 2).tolist(), (w / 2).tolist()]


def _binary_bias_to_per_class(lr) -> list[float]:
    b = float(lr.intercept_[0])
    return [-b / 2, b / 2]


def coverage_at_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    classes: list[str],
    threshold: float,
) -> tuple[float, float | None]:
    """Confidence-gated coverage and accuracy-on-covered.

    coverage = fraction of samples where max prob >= threshold (the cases
    the probe would handle without LLM escalation). accuracy = fraction of
    covered samples predicted correctly; None if nothing is covered.
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    top = probs.max(axis=1)
    pred = np.asarray(classes)[probs.argmax(axis=1)]
    covered = top >= threshold
    coverage = float(covered.mean())
    if not covered.any():
        return coverage, None
    accuracy = float((pred[covered] == y_true[covered]).mean())
    return coverage, accuracy


def select_lstar(accuracy_by_layer: dict[int, float], fraction: float = 0.95) -> int:
    """Reasoning depth L*: earliest layer reaching `fraction` of best accuracy.

    The accuracy-vs-layer curve tells you how much model a decision needs;
    L* is where to truncate the forward pass for that field.
    """
    best = max(accuracy_by_layer.values())
    cutoff = best * fraction
    for layer in sorted(accuracy_by_layer):
        if accuracy_by_layer[layer] >= cutoff:
            return layer
    return max(accuracy_by_layer)  # unreachable, best itself qualifies


def export_probe_rule(probe: dict, path: str, weights_path: str | None = None,
                      confidence_threshold: float = 0.95) -> dict:
    """Convert a trained probe into a hybrid rule dict.

    If weights_path is given, weights/bias are saved to that .npz and the
    rule references it (weights_ref) instead of inlining a 5120-dim matrix.
    """
    rule = {
        "path": path,
        "type": "hidden_probe",
        "layer": probe["layer"],
        "position": probe["position"],
        "classes": probe["classes"],
        "confidence_threshold": confidence_threshold,
        "probe_model": probe["probe_model"],
        "cv_accuracy": probe["cv_accuracy"],
    }
    if weights_path:
        import hashlib

        w = np.array(probe["weights"], dtype=np.float32)
        b = np.array(probe["bias"], dtype=np.float32)
        np.savez_compressed(weights_path, weights=w, bias=b)
        digest = hashlib.sha256(open(weights_path, "rb").read()).hexdigest()
        rule["weights_ref"] = str(weights_path)
        rule["weights_sha256"] = digest
    else:
        rule["weights"] = probe["weights"]
        rule["bias"] = probe["bias"]
    return rule
