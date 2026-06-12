"""Tests for hidden-state probe training (drift.hybrid.probes)."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from drift.hybrid.probes import (
    coverage_at_threshold,
    train_hidden_probe,
)


def _separable_data(n=60, d=16, seed=0):
    """Two linearly separable classes in d dims."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d)).astype(np.float32)
    y = np.array(["a" if x[0] + x[1] > 0 else "b" for x in X])
    # push apart for clean separation
    X[y == "a", 0] += 2.0
    X[y == "b", 0] -= 2.0
    return X, y


class TestTrainHiddenProbe:
    def test_separable_data_high_accuracy(self):
        X, y = _separable_data()
        probe = train_hidden_probe(X, y, layer=12, probe_model="m")
        assert probe is not None
        assert probe["cv_accuracy"] > 0.9
        assert sorted(probe["classes"]) == ["a", "b"]
        assert len(probe["weights"]) == 2
        assert len(probe["weights"][0]) == X.shape[1]
        assert probe["layer"] == 12
        assert probe["type"] == "hidden_probe"

    def test_shuffled_labels_rejected(self):
        X, y = _separable_data()
        rng = np.random.default_rng(1)
        y_shuffled = rng.permutation(y)
        probe = train_hidden_probe(X, y_shuffled, layer=12, probe_model="m",
                                   min_accuracy=0.7)
        assert probe is None

    def test_single_class_rejected(self):
        X, _ = _separable_data()
        y = np.array(["a"] * len(X))
        assert train_hidden_probe(X, y, layer=0, probe_model="m") is None

    def test_oof_probs_shape(self):
        X, y = _separable_data()
        probe = train_hidden_probe(X, y, layer=12, probe_model="m")
        oof = probe["oof_probs"]
        assert oof.shape == (len(y), len(probe["classes"]))
        np.testing.assert_allclose(oof.sum(axis=1), 1.0, rtol=1e-5)


class TestCoverage:
    def test_hand_computed(self):
        classes = ["a", "b"]
        probs = np.array([
            [0.99, 0.01],   # covered, correct (a)
            [0.97, 0.03],   # covered, wrong (true b)
            [0.60, 0.40],   # not covered
            [0.02, 0.98],   # covered, correct (b)
        ])
        y = np.array(["a", "b", "a", "b"])
        cov, acc = coverage_at_threshold(probs, y, classes, threshold=0.95)
        assert cov == pytest.approx(3 / 4)
        assert acc == pytest.approx(2 / 3)

    def test_nothing_covered(self):
        probs = np.array([[0.6, 0.4], [0.5, 0.5]])
        y = np.array(["a", "b"])
        cov, acc = coverage_at_threshold(probs, y, ["a", "b"], threshold=0.95)
        assert cov == 0.0
        assert acc is None


class TestLstarSelection:
    def test_monotone_curve(self):
        from drift.hybrid.probes import select_lstar
        acc = {0: 0.5, 8: 0.7, 16: 0.9, 24: 0.95, 32: 0.96, 40: 0.96}
        # best = 0.96, 95% of best = 0.912 -> earliest layer >= 0.912 is 24
        assert select_lstar(acc) == 24

    def test_plateau_picks_earliest(self):
        from drift.hybrid.probes import select_lstar
        acc = {0: 0.9, 8: 0.9, 16: 0.9}
        assert select_lstar(acc) == 0

    def test_noisy_curve(self):
        from drift.hybrid.probes import select_lstar
        acc = {0: 0.6, 8: 0.93, 16: 0.85, 24: 0.95}
        # 95% of 0.95 = 0.9025 -> layer 8 qualifies despite later dip
        assert select_lstar(acc) == 8

    def test_custom_fraction(self):
        from drift.hybrid.probes import select_lstar
        acc = {0: 0.5, 8: 0.8, 16: 1.0}
        assert select_lstar(acc, fraction=0.79) == 8
