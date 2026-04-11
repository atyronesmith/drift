#!/usr/bin/env python3
"""Train a guard classifier on Llama Guard labels for tiered guard escalation.

Trains a logistic regression classifier on safe/unsafe labels from the guard
logprobs data. Exports compiled weights for use in denoise's guard.py module.

Input: results/guard_logprobs.json (from probe_guard_logprobs.py)
Output: results/guard_classifier_weights.json — vocabulary, weights, bias
        results/guard_tier1_keywords.json — regex keywords for Tier 1

Usage:
    uv run python scripts/train_guard_classifier.py
    uv run python scripts/train_guard_classifier.py --logprobs results/guard_logprobs.json
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her",
    "my", "our", "your", "his", "its", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "and", "or", "but", "if", "when", "while", "as", "at", "by", "for",
    "in", "of", "on", "to", "with", "from", "not", "no", "so", "than",
    "too", "very", "just", "also", "how", "all", "each", "every", "any",
    "few", "more", "most", "other", "some", "such", "only", "own", "same",
    "then", "there", "here", "now", "about", "up", "out", "into", "over",
    "after", "before", "between", "under", "again", "further",
}


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer matching the compiled engine."""
    return re.findall(r"[a-z]+", text.lower())


def load_full_content(results_dir: Path) -> dict[tuple, str]:
    """Load full LLM output content keyed by (experiment, timestamp, run_index, step)."""
    content_map = {}

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("__"):
            continue

        for ts_dir in sorted(exp_dir.iterdir()):
            if not ts_dir.is_dir():
                continue

            for run_file in sorted(ts_dir.glob("run_*.json")):
                try:
                    run = json.loads(run_file.read_text())
                except (json.JSONDecodeError, OSError):
                    continue

                run_idx = run.get("run_index", 0)
                run_type = run.get("type", "unknown")
                exp_name = exp_dir.name

                if run_type in ("baseline", "hybrid"):
                    content = run.get("response", {}).get("content", "")
                    if content:
                        content_map[(exp_name, ts_dir.name, run_idx, run_type)] = content

                elif run_type == "pipeline":
                    for step in run.get("steps", []):
                        if isinstance(step, dict):
                            name = step.get("name", "unknown")
                            content = step.get("response", {}).get("content", "")
                            if content:
                                content_map[(exp_name, ts_dir.name, run_idx, name)] = content
                        elif isinstance(step, list):
                            for iteration in step:
                                if isinstance(iteration, list):
                                    for s in iteration:
                                        name = s.get("name", "unknown")
                                        content = s.get("response", {}).get("content", "")
                                        if content:
                                            content_map[(exp_name, ts_dir.name, run_idx, name)] = content

    return content_map


def build_features(contents: list[str], vocab_size: int = 200):
    """Build feature matrix: bag-of-words + structural features.

    Returns: X, feature_names, vocab
    """
    # Build vocabulary from top words
    word_counts = Counter()
    for content in contents:
        words = set(tokenize(content))
        word_counts.update(w for w in words if w not in STOPWORDS and len(w) >= 3)

    vocab = [w for w, _ in word_counts.most_common(vocab_size)]
    vocab_index = {w: i for i, w in enumerate(vocab)}

    # 3 structural features
    structural_names = ["starts_with_code_fence", "starts_with_json", "starts_with_think"]
    n_features = len(vocab) + len(structural_names)
    feature_names = list(vocab) + structural_names

    X = np.zeros((len(contents), n_features), dtype=np.float32)

    for i, content in enumerate(contents):
        # Bag-of-words (binary)
        for w in tokenize(content):
            if w in vocab_index:
                X[i, vocab_index[w]] = 1.0

        # Structural features
        stripped = content.strip()
        X[i, len(vocab)] = 1.0 if stripped.startswith("```") else 0.0
        X[i, len(vocab) + 1] = 1.0 if stripped.startswith("{") else 0.0
        X[i, len(vocab) + 2] = 1.0 if stripped.startswith("<think>") else 0.0

    return X, feature_names, vocab


def pure_python_predict(weights, bias, features, classes):
    """Logistic regression in pure Python — must match denoise/guard.py exactly."""
    n_classes = len(classes)
    logits = []
    for c in range(n_classes):
        s = bias[c]
        for j, fval in enumerate(features):
            if fval != 0:
                s += weights[c][j] * fval
        logits.append(s)

    max_logit = max(logits)
    exp_logits = [math.exp(l - max_logit) for l in logits]
    total = sum(exp_logits)
    probs = [e / total for e in exp_logits]

    best_idx = probs.index(max(probs))
    return classes[best_idx], probs[best_idx]


def main():
    parser = argparse.ArgumentParser(description="Train guard classifier")
    parser.add_argument("--logprobs", default="results/guard_logprobs.json",
                        help="Guard logprobs JSON from probe script")
    parser.add_argument("--results-dir", default="results",
                        help="Results directory for full content")
    parser.add_argument("--vocab-size", type=int, default=200,
                        help="Vocabulary size for bag-of-words")
    parser.add_argument("--output-dir", default="results/guard_classifier",
                        help="Output directory")
    args = parser.parse_args()

    # Load logprobs data
    logprobs_path = Path(args.logprobs)
    if not logprobs_path.is_absolute():
        logprobs_path = Path(__file__).parent.parent / logprobs_path
    records = json.loads(logprobs_path.read_text())
    print(f"Loaded {len(records)} logprob records")

    # Filter out errors
    records = [r for r in records if "error" not in r]
    print(f"Valid records: {len(records)}")

    # Load full content
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = Path(__file__).parent.parent / results_dir
    content_map = load_full_content(results_dir)
    print(f"Full content map: {len(content_map)} entries")

    # Match records to full content
    contents = []
    labels = []
    categories = []
    logprob_data = []

    for rec in records:
        key = (rec["experiment"], rec["timestamp"], rec["run_index"], rec["step"])
        full_content = content_map.get(key)
        if not full_content:
            continue
        contents.append(full_content)
        labels.append("safe" if rec["safe"] else "unsafe")
        categories.append(rec.get("category") or rec.get("audit_category"))
        logprob_data.append(rec)

    print(f"Matched with full content: {len(contents)}")
    label_counts = Counter(labels)
    print(f"Labels: {dict(label_counts)}")

    cat_counts = Counter(c for c in categories if c)
    print(f"Unsafe categories: {dict(cat_counts)}")

    # Build features
    print(f"\n{'='*70}")
    print("FEATURE ENGINEERING")
    print(f"{'='*70}\n")

    X, feature_names, vocab = build_features(contents, vocab_size=args.vocab_size)
    y = np.array(labels)

    print(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"Vocabulary: {len(vocab)} words")
    print(f"Top 20 vocab: {vocab[:20]}")

    # Structural feature distribution
    n_code = sum(X[:, len(vocab)] == 1.0)
    n_json = sum(X[:, len(vocab) + 1] == 1.0)
    n_think = sum(X[:, len(vocab) + 2] == 1.0)
    print(f"\nStructural features:")
    print(f"  Code fence: {n_code} ({n_code/len(y)*100:.1f}%)")
    print(f"  JSON: {n_json} ({n_json/len(y)*100:.1f}%)")
    print(f"  Think: {n_think} ({n_think/len(y)*100:.1f}%)")

    # Train/test split
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # Logistic Regression
    print("--- Logistic Regression ---")
    lr = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
    cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring="accuracy")
    print(f"5-fold CV: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

    lr.fit(X_train, y_train)
    train_acc = lr.score(X_train, y_train)
    test_acc = lr.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")

    y_pred = lr.predict(X_test)
    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    # Confidence calibration
    probs = lr.predict_proba(X_test)
    max_probs = np.max(probs, axis=1)
    correct = (y_pred == y_test)

    print("Confidence calibration:")
    thresholds = [0.50, 0.70, 0.80, 0.90, 0.95]
    for thresh in thresholds:
        mask = max_probs >= thresh
        n_above = mask.sum()
        if n_above > 0:
            acc_above = correct[mask].mean()
            coverage = n_above / len(y_test)
            print(f"  conf >= {thresh:.2f}: {n_above:3d}/{len(y_test)} "
                  f"({coverage:.0%} coverage), accuracy = {acc_above:.3f}")
    print()

    # Decision Tree (ceiling check)
    print("--- Decision Tree (max_depth=5) ---")
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    cv_dt = cross_val_score(dt, X_train, y_train, cv=5, scoring="accuracy")
    dt.fit(X_train, y_train)
    print(f"5-fold CV: {cv_dt.mean():.3f}, Test: {dt.score(X_test, y_test):.3f}\n")

    # Gradient Boosted (ceiling check)
    print("--- Gradient Boosted Trees (ceiling check) ---")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    cv_gb = cross_val_score(gb, X_train, y_train, cv=5, scoring="accuracy")
    gb.fit(X_train, y_train)
    print(f"5-fold CV: {cv_gb.mean():.3f}, Test: {gb.score(X_test, y_test):.3f}\n")

    # Safety check: S1/S3 recall
    print("--- Safety Category Recall ---")
    # Find S1/S3 samples in test set
    test_indices = []
    train_size = len(y_train)
    # Reconstruct test indices
    from sklearn.model_selection import train_test_split as tts
    _, test_idx = tts(range(len(y)), test_size=0.2, random_state=42, stratify=y)

    for idx in test_idx:
        cat = categories[idx]
        if cat in ("S1", "S3"):
            feat_idx = list(test_idx).index(idx)
            pred = y_pred[feat_idx]
            actual = y_test[feat_idx]
            print(f"  {cat} sample: actual={actual}, predicted={pred}")

    # Pure-Python inference parity
    print("\n--- Pure-Python Inference Parity ---")
    lr_classes = list(lr.classes_)
    # For binary classification, sklearn stores single weight vector.
    # Expand to 2-class format: class_0 weights = -w, class_1 weights = +w
    if lr.coef_.shape[0] == 1:
        lr_weights = [(-lr.coef_[0]).tolist(), lr.coef_[0].tolist()]
        lr_bias = [-lr.intercept_[0], lr.intercept_[0]]
    else:
        lr_weights = lr.coef_.tolist()
        lr_bias = lr.intercept_.tolist()

    mismatches = 0
    for i in range(len(X_test)):
        features = X_test[i].tolist()
        py_pred, py_conf = pure_python_predict(lr_weights, lr_bias, features, lr_classes)
        sk_pred = y_pred[i]
        if py_pred != sk_pred:
            mismatches += 1

    if mismatches == 0:
        print(f"PASS: All {len(X_test)} predictions match sklearn")
    else:
        print(f"FAIL: {mismatches}/{len(X_test)} mismatches")
    print()

    # Top features per class
    print("--- Top Features per Class ---")
    for idx, cls in enumerate(lr_classes):
        coefs_row = lr_weights[idx]
        top_indices = sorted(range(len(coefs_row)), key=lambda j: coefs_row[j], reverse=True)[:10]
        top_feats = [(feature_names[j], round(coefs_row[j], 3)) for j in top_indices]
        print(f"  {cls}: {', '.join(f'{f}={w}' for f, w in top_feats)}")
    print()

    # Export weights
    print(f"{'='*70}")
    print("EXPORT")
    print(f"{'='*70}\n")

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).parent.parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Classifier weights
    export = {
        "vocabulary": vocab,
        "structural_features": [
            {"name": "starts_with_code_fence", "index": len(vocab)},
            {"name": "starts_with_json", "index": len(vocab) + 1},
            {"name": "starts_with_think", "index": len(vocab) + 2},
        ],
        "classes": lr_classes,
        "weights": lr_weights,
        "bias": lr_bias,
        "confidence_threshold": 0.90,
        "training_stats": {
            "n_samples": len(y),
            "n_features": X.shape[1],
            "cv_accuracy": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
            "test_accuracy": round(float(test_acc), 4),
            "parity_mismatches": mismatches,
        },
    }

    weights_path = out_dir / "guard_classifier_weights.json"
    weights_path.write_text(json.dumps(export, indent=2))
    print(f"Classifier weights: {weights_path}")

    # Tier 1 keywords: top features that strongly predict unsafe
    unsafe_idx = lr_classes.index("unsafe")
    coefs = lr_weights[unsafe_idx]

    # Top unsafe-predicting words (from vocabulary only, not structural)
    unsafe_keywords = []
    safe_keywords = []
    for j in range(len(vocab)):
        w = coefs[j]
        if w > 0.5:
            unsafe_keywords.append({"word": vocab[j], "weight": round(float(w), 3)})
        elif w < -0.5:
            safe_keywords.append({"word": vocab[j], "weight": round(float(w), 3)})

    unsafe_keywords.sort(key=lambda x: x["weight"], reverse=True)
    safe_keywords.sort(key=lambda x: x["weight"])

    tier1 = {
        "unsafe_keywords": [k["word"] for k in unsafe_keywords[:30]],
        "safe_keywords": [k["word"] for k in safe_keywords[:30]],
        "unsafe_details": unsafe_keywords,
        "safe_details": safe_keywords,
    }

    keywords_path = out_dir / "guard_tier1_keywords.json"
    keywords_path.write_text(json.dumps(tier1, indent=2))
    print(f"Tier 1 keywords: {keywords_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  LR CV accuracy:    {cv_scores.mean():.1%}")
    print(f"  LR test accuracy:  {test_acc:.1%}")
    print(f"  DT test accuracy:  {dt.score(X_test, y_test):.1%}")
    print(f"  GB test accuracy:  {gb.score(X_test, y_test):.1%}")
    print(f"  Parity check:      {'PASS' if mismatches == 0 else 'FAIL'}")
    print(f"  Unsafe keywords:   {len(unsafe_keywords)}")
    print(f"  Safe keywords:     {len(safe_keywords)}")
    print(f"  Threshold:         0.90")
    print()


if __name__ == "__main__":
    main()
