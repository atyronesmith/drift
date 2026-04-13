"""Tests for classifier baseline calibration."""

import json
import math
import pytest

from drift.hybrid.calibration import compute_baseline, compute_jsd, save_baseline, load_baseline


def _make_training_data(n=50):
    """Generate synthetic training data with clear class separation."""
    texts = []
    extracted = []
    for i in range(n):
        if i % 3 == 0:
            texts.append("printer disconnecting network slow")
            extracted.append({"ticket": {"category": "Network"}})
        elif i % 3 == 1:
            texts.append("password reset access denied login")
            extracted.append({"ticket": {"category": "Account"}})
        else:
            texts.append("application crash error update failed")
            extracted.append({"ticket": {"category": "Software"}})
    return texts, extracted


def test_compute_baseline_returns_histogram():
    texts, extracted = _make_training_data()
    # Simple 2-class classifier weights
    baseline = compute_baseline(
        weights=[[2.0, 1.0, -1.0, -1.0], [-1.0, -1.0, 2.0, 1.0]],
        bias=[0.0, 0.0],
        classes=["network", "access_control"],
        vocabulary=["printer", "network", "password", "access"],
        structured_features=None,
        training_texts=texts,
        training_extracted=extracted,
    )
    assert "histogram" in baseline
    assert len(baseline["histogram"]) == 20  # default n_bins
    assert baseline["n_bins"] == 20
    assert baseline["n_samples"] == len(texts)
    assert "class_distribution" in baseline
    assert "mean_confidence" in baseline
    assert 0 < baseline["mean_confidence"] <= 1.0
    # Histogram should sum to ~1.0
    assert abs(sum(baseline["histogram"]) - 1.0) < 1e-6


def test_compute_baseline_with_structured_features():
    texts, extracted = _make_training_data()
    baseline = compute_baseline(
        weights=[
            [1.0, -1.0, 3.0, -1.0],  # network: strong on Network category
            [-1.0, 1.0, -1.0, 3.0],  # access_control: strong on Account
        ],
        bias=[0.0, 0.0],
        classes=["network", "access_control"],
        vocabulary=["printer", "password"],
        structured_features=[
            {"source": "ticket.category", "categories": ["Network", "Account"]},
        ],
        training_texts=texts,
        training_extracted=extracted,
    )
    assert baseline["n_samples"] == len(texts)
    assert sum(baseline["class_distribution"].values()) == len(texts)


def test_compute_baseline_custom_bins():
    texts, extracted = _make_training_data(30)
    baseline = compute_baseline(
        weights=[[1.0], [-1.0]],
        bias=[0.0, 0.0],
        classes=["a", "b"],
        vocabulary=["printer"],
        structured_features=None,
        training_texts=texts,
        training_extracted=extracted,
        n_bins=10,
    )
    assert len(baseline["histogram"]) == 10
    assert baseline["n_bins"] == 10


def test_compute_jsd_identical():
    p = [0.1, 0.2, 0.3, 0.4]
    assert compute_jsd(p, p) == 0.0


def test_compute_jsd_different():
    p = [1.0, 0.0, 0.0, 0.0]
    q = [0.0, 0.0, 0.0, 1.0]
    jsd = compute_jsd(p, q)
    assert jsd == 1.0  # maximally different


def test_compute_jsd_symmetric():
    p = [0.3, 0.7, 0.0]
    q = [0.1, 0.4, 0.5]
    assert compute_jsd(p, q) == compute_jsd(q, p)


def test_compute_jsd_bounded():
    p = [0.2, 0.3, 0.5]
    q = [0.4, 0.1, 0.5]
    jsd = compute_jsd(p, q)
    assert 0 <= jsd <= 1.0


def test_compute_jsd_length_mismatch():
    with pytest.raises(ValueError):
        compute_jsd([0.5, 0.5], [0.3, 0.3, 0.4])


def test_save_and_load_baseline(tmp_path):
    baseline = {
        "histogram": [0.1, 0.2, 0.7],
        "n_bins": 3,
        "n_samples": 100,
        "class_distribution": {"a": 30, "b": 70},
        "mean_confidence": 0.85,
        "classes": ["a", "b"],
    }
    path = tmp_path / "baselines" / "clf.json"
    save_baseline(baseline, path)
    loaded = load_baseline(path)
    assert loaded == baseline
