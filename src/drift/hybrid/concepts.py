"""Concept-bottleneck programs (Phase 5 of the reasoning-extraction plan).

Instead of one opaque probe per output field, decompose: train a small
binary probe per human-named concept ("is the user locked out", "is this
hardware"), then fit a shallow decision tree over the probed concept
probabilities. The exported tree is auditable reasoning logic — every
branch reads as an if-statement over named concepts.

The accuracy delta vs a direct hidden_probe on the same field is the
measured price of interpretability; report both.
"""

from __future__ import annotations

import numpy as np


def fit_concept_program(
    concept_probs: np.ndarray,
    concept_names: list[str],
    y: np.ndarray,
    max_depth: int = 4,
    min_samples_leaf: int = 5,
    seed: int = 42,
) -> dict:
    """Fit a shallow decision tree over probed concept values.

    Train on *probed* (not gold) concept probabilities so the program sees
    runtime-realistic inputs and absorbs probe noise.

    Returns dict: tree (nested if/then/else export), accuracy (train),
    train_predictions, classes.
    """
    from sklearn.tree import DecisionTreeClassifier

    X = np.asarray(concept_probs, dtype=np.float32)
    y = np.asarray(y)

    clf = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed
    )
    clf.fit(X, y)
    pred = clf.predict(X)

    tree = _export_tree(clf, concept_names)
    return {
        "tree": tree,
        "accuracy": float((pred == y).mean()),
        "train_predictions": pred.tolist(),
        "classes": [str(c) for c in clf.classes_],
    }


def _export_tree(clf, concept_names: list[str]) -> dict:
    """Serialize an sklearn tree as nested {if/then/else} dicts.

    Leaves carry the majority class, purity (used as confidence), and
    sample count. sklearn splits are `feature <= threshold` with the left
    child taken when true.
    """
    t = clf.tree_

    def node(i: int) -> dict:
        if t.children_left[i] == -1:  # leaf
            counts = t.value[i][0]
            best = int(counts.argmax())
            total = float(counts.sum())
            return {
                "value": str(clf.classes_[best]),
                "confidence": round(float(counts[best] / total), 4),
                "samples": int(total),
            }
        return {
            "if": {
                "concept": concept_names[t.feature[i]],
                "op": "<=",
                "threshold": round(float(t.threshold[i]), 6),
            },
            "then": node(t.children_left[i]),
            "else": node(t.children_right[i]),
        }

    return node(0)


def eval_concept_program(tree: dict, concept_values: dict[str, float]) -> tuple[str, float]:
    """Walk an exported tree with probed concept values.

    Returns (predicted value, leaf confidence).
    """
    node = tree
    while "value" not in node:
        cond = node["if"]
        v = concept_values[cond["concept"]]
        op = cond["op"]
        if op == "<=":
            taken = v <= cond["threshold"]
        elif op == ">":
            taken = v > cond["threshold"]
        else:
            raise ValueError(f"Unknown op: {op!r}")
        node = node["then"] if taken else node["else"]
    return node["value"], node["confidence"]
