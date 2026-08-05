"""Soft-label distillation: train students against teacher distributions.

sklearn's LogisticRegression only accepts hard labels. The teacher's full
softmax at decision tokens carries calibration information hard labels
throw away, so this module implements plain numpy softmax regression
trained with cross-entropy against soft targets (Hinton-style distillation,
optionally temperature-scaled).

Students trained on soft labels are markedly better calibrated, which makes
confidence-gated escalation principled instead of heuristic.
"""

from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax with max-subtraction for numerical stability."""
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def train_soft_student(
    X: np.ndarray,
    teacher_probs: np.ndarray,
    temperature: float = 1.0,
    l2: float = 1e-3,
    lr: float = 0.1,
    epochs: int = 500,
    seed: int = 42,
) -> dict:
    """Train a softmax-regression student on soft targets.

    Full-batch gradient descent on cross-entropy H(teacher_T, student) with
    L2 regularization. With temperature T > 1 the teacher distribution is
    re-softened (p^(1/T) renormalized) before training.

    Args:
        X: [n, d] features.
        teacher_probs: [n, k] teacher distribution rows (sum to 1).
            One-hot rows reduce this to ordinary logistic regression.
        temperature: distillation temperature applied to the targets.
        l2: L2 penalty on weights.
        lr: learning rate.
        epochs: full-batch iterations.

    Returns dict: weights [k, d], bias [k], final_loss, temperature.
    """
    X = np.asarray(X, dtype=np.float64)
    targets = np.asarray(teacher_probs, dtype=np.float64)
    n, d = X.shape
    k = targets.shape[1]

    if temperature != 1.0:
        targets = np.power(np.clip(targets, 1e-12, 1.0), 1.0 / temperature)
        targets = targets / targets.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, (k, d))
    b = np.zeros(k)

    loss = float("inf")
    for _ in range(epochs):
        probs = softmax(X @ W.T + b)            # [n, k]
        grad_logits = (probs - targets) / n     # [n, k]
        grad_W = grad_logits.T @ X + l2 * W
        grad_b = grad_logits.sum(axis=0)
        W -= lr * grad_W
        b -= lr * grad_b
        loss = float(
            -(targets * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1).mean()
            + 0.5 * l2 * (W ** 2).sum()
        )

    return {
        "weights": W.tolist(),
        "bias": b.tolist(),
        "final_loss": loss,
        "temperature": temperature,
    }


def expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    classes: list[str],
    n_bins: int = 10,
) -> float:
    """ECE on top-class confidence: sum_b (n_b/n) * |acc_b - conf_b|."""
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    top = probs.max(axis=1)
    pred = np.asarray(classes)[probs.argmax(axis=1)]
    correct = (pred == y_true).astype(float)

    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (top > lo) & (top <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - top[mask].mean())
    return float(total)
