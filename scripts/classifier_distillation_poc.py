#!/usr/bin/env python3
"""Classifier distillation POC for classification fields.

Trains small classifiers on LLM-labeled helpdesk tickets to replace LLM
inference for root_cause_category and assignment_group fields.

Phase 1: Collect LLM labels (cached to disk)
Phase 2: Feature engineering (bag-of-words + structured one-hot)
Phase 3: Train logistic regression, decision tree, gradient-boosted trees
Phase 4: Validate with 5-fold CV + hold-out test
Phase 5: Pure-Python inference parity check
Phase 6: Export weights as JSON

Usage:
    uv run python scripts/classifier_distillation_poc.py --tickets 400
    uv run python scripts/classifier_distillation_poc.py --labels results/classifier_training/llm_labels.jsonl
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = "qwen2.5:32b"

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
    "hello", "please", "thanks", "thank", "help", "issue", "issues",
    "hey", "dear", "regards", "best", "team", "hi", "support",
}


def format_ticket(t: dict) -> str:
    return (
        f"Ticket ID:    {t['id']}\n"
        f"Subject:      {t['subject']}\n"
        f"From:         {t['requesterEmail']}\n"
        f"Created:      {t['createdAt']}\n"
        f"Priority:     {t['priority']}\n"
        f"Category:     {t['category']}\n"
        f"\nDescription:\n{t['description']}"
    )


def load_tickets(path="datasets/helpdesk_tickets.jsonl", n=400):
    tickets = []
    with open(path) as f:
        for line in f:
            tickets.append(json.loads(line))
            if len(tickets) >= n:
                break
    return tickets


def call_llm(system_prompt: str, user_prompt: str) -> dict | None:
    """Call Ollama via subprocess+curl, return parsed JSON or None."""
    url = f"{OLLAMA_URL}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "stream": False,
    }
    try:
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", url,
             "-H", "Content-Type: application/json",
             "-d", json.dumps(payload)],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        if "error" in data:
            return None
        content = data["choices"][0]["message"]["content"].strip()
        # Strip code fences
        if content.startswith("```"):
            lines = content.split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
        return json.loads(content)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
        return None


def collect_labels(tickets: list[dict], system_prompt: str, user_prompt_tpl: str,
                   cache_path: str) -> list[dict]:
    """Run tickets through LLM, cache labels to disk."""
    # Load existing cache
    cached = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                rec = json.loads(line)
                cached[rec["ticket_id"]] = rec
        print(f"  Loaded {len(cached)} cached labels from {cache_path}")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    labels = []
    new_count = 0

    for i, ticket in enumerate(tickets):
        tid = ticket["id"]

        if tid in cached:
            labels.append(cached[tid])
            continue

        input_text = format_ticket(ticket)
        prompt = user_prompt_tpl.replace("{input}", input_text)

        print(f"  [{i+1:3d}/{len(tickets)}] {tid[:8]}... ", end="", flush=True)
        t0 = time.time()
        parsed = call_llm(system_prompt, prompt)
        elapsed = time.time() - t0

        if not parsed:
            print(f"{elapsed:.1f}s  SKIP (parse error)")
            continue

        assessment = parsed.get("assessment", {})
        rc = assessment.get("root_cause_category")
        ag = assessment.get("assignment_group")

        if not rc or not ag:
            print(f"{elapsed:.1f}s  SKIP (missing fields)")
            continue

        rec = {
            "ticket_id": tid,
            "root_cause_category": rc,
            "assignment_group": ag,
            "severity": assessment.get("severity"),
        }
        labels.append(rec)

        # Append to cache file incrementally
        with open(cache_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        new_count += 1

        print(f"{elapsed:.1f}s  rc={rc:<20s} ag={ag}")

    print(f"  Collected {len(labels)} labels ({new_count} new, {len(labels)-new_count} cached)")
    return labels


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer matching what the generated engine will use."""
    return re.findall(r"[a-z]+", text.lower())


def build_features(tickets: list[dict], labels: list[dict], vocab_size: int = 200):
    """Build feature matrix: bag-of-words on description + structured one-hot.

    Returns: X (n_samples x n_features), feature_names, vocab list,
             priority_values, category_values
    """
    # Index labels by ticket ID
    label_by_id = {rec["ticket_id"]: rec for rec in labels}

    # Collect descriptions and structured fields for labeled tickets
    descriptions = []
    priorities = []
    categories = []
    ticket_ids = []

    for ticket in tickets:
        if ticket["id"] not in label_by_id:
            continue
        descriptions.append(ticket["description"])
        priorities.append(ticket["priority"])
        categories.append(ticket["category"])
        ticket_ids.append(ticket["id"])

    # Build vocabulary from top words (excluding stopwords)
    word_counts = Counter()
    for desc in descriptions:
        words = set(tokenize(desc))  # binary per document
        word_counts.update(w for w in words if w not in STOPWORDS and len(w) >= 3)

    vocab = [w for w, _ in word_counts.most_common(vocab_size)]
    vocab_index = {w: i for i, w in enumerate(vocab)}

    # One-hot categories for structured fields
    priority_values = sorted(set(priorities))
    category_values = sorted(set(categories))

    n_features = len(vocab) + len(priority_values) + len(category_values)
    X = np.zeros((len(descriptions), n_features), dtype=np.float32)
    feature_names = list(vocab) + [f"priority_{p}" for p in priority_values] + \
                    [f"category_{c}" for c in category_values]

    for i, (desc, pri, cat) in enumerate(zip(descriptions, priorities, categories)):
        # Bag-of-words features (binary)
        for w in tokenize(desc):
            if w in vocab_index:
                X[i, vocab_index[w]] = 1.0

        # Priority one-hot
        offset = len(vocab)
        if pri in priority_values:
            X[i, offset + priority_values.index(pri)] = 1.0

        # Category one-hot
        offset += len(priority_values)
        if cat in category_values:
            X[i, offset + category_values.index(cat)] = 1.0

    return X, feature_names, vocab, priority_values, category_values, ticket_ids


def pure_python_predict(weights: list[list[float]], bias: list[float],
                        features: list[float], classes: list[str]) -> tuple[str, float]:
    """Logistic regression inference in pure Python (no numpy/sklearn).

    This is exactly what the generated engine will run.
    """
    n_classes = len(classes)
    logits = []
    for c in range(n_classes):
        s = bias[c]
        for j, fval in enumerate(features):
            if fval != 0:
                s += weights[c][j] * fval
        logits.append(s)

    # Softmax with numerical stability
    max_logit = max(logits)
    exp_logits = [math.exp(l - max_logit) for l in logits]
    total = sum(exp_logits)
    probs = [e / total for e in exp_logits]

    best_idx = probs.index(max(probs))
    return classes[best_idx], probs[best_idx]


def train_and_evaluate(X, y, field_name, feature_names):
    """Train classifiers, cross-validate, report results."""
    n_samples = len(y)
    class_counts = Counter(y)

    # Filter out classes with < 3 samples (can't stratify-split)
    rare_classes = {cls for cls, cnt in class_counts.items() if cnt < 3}
    if rare_classes:
        print(f"  Dropping rare classes (< 3 samples): {sorted(rare_classes)}")
        mask = np.array([yi not in rare_classes for yi in y])
        X = X[mask]
        y = y[mask]
        n_samples = len(y)
        class_counts = Counter(y)

    classes = sorted(set(y))
    n_classes = len(classes)

    print(f"  Samples: {n_samples}, Features: {X.shape[1]}, Classes: {n_classes}")
    print(f"  Classes: {classes}")
    for cls in classes:
        print(f"    {cls}: {class_counts[cls]} ({class_counts[cls]/n_samples:.0%})")
    print()

    # Train/test split (80/20, stratified)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    results = {}

    # 1. Logistic Regression
    print("  --- Logistic Regression ---")
    lr = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
    cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring="accuracy")
    print(f"  5-fold CV: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

    lr.fit(X_train, y_train)
    train_acc = lr.score(X_train, y_train)
    test_acc = lr.score(X_test, y_test)
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy:  {test_acc:.3f}")

    y_pred = lr.predict(X_test)
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")

    results["logistic_regression"] = {
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "train_accuracy": round(float(train_acc), 4),
        "test_accuracy": round(float(test_acc), 4),
        "model": lr,
    }

    # Confidence calibration
    probs = lr.predict_proba(X_test)
    max_probs = np.max(probs, axis=1)
    correct = (y_pred == y_test)

    print("  Confidence calibration:")
    thresholds = [0.50, 0.70, 0.80, 0.90, 0.95]
    for thresh in thresholds:
        mask = max_probs >= thresh
        n_above = mask.sum()
        if n_above > 0:
            acc_above = correct[mask].mean()
            coverage = n_above / len(y_test)
            print(f"    conf >= {thresh:.2f}: {n_above:3d}/{len(y_test)} "
                  f"({coverage:.0%} coverage), accuracy = {acc_above:.3f}")
        else:
            print(f"    conf >= {thresh:.2f}: 0 samples")
    print()

    # 2. Decision Tree
    print("  --- Decision Tree (max_depth=5) ---")
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    cv_scores_dt = cross_val_score(dt, X_train, y_train, cv=5, scoring="accuracy")
    print(f"  5-fold CV: {cv_scores_dt.mean():.3f} (+/- {cv_scores_dt.std()*2:.3f})")

    dt.fit(X_train, y_train)
    print(f"  Train accuracy: {dt.score(X_train, y_train):.3f}")
    print(f"  Test accuracy:  {dt.score(X_test, y_test):.3f}")

    results["decision_tree"] = {
        "cv_mean": round(float(cv_scores_dt.mean()), 4),
        "cv_std": round(float(cv_scores_dt.std()), 4),
        "train_accuracy": round(float(dt.score(X_train, y_train)), 4),
        "test_accuracy": round(float(dt.score(X_test, y_test)), 4),
    }
    print()

    # 3. Gradient Boosted Trees (ceiling check)
    print("  --- Gradient Boosted Trees (ceiling check) ---")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    cv_scores_gb = cross_val_score(gb, X_train, y_train, cv=5, scoring="accuracy")
    print(f"  5-fold CV: {cv_scores_gb.mean():.3f} (+/- {cv_scores_gb.std()*2:.3f})")

    gb.fit(X_train, y_train)
    print(f"  Train accuracy: {gb.score(X_train, y_train):.3f}")
    print(f"  Test accuracy:  {gb.score(X_test, y_test):.3f}")

    results["gradient_boosted"] = {
        "cv_mean": round(float(cv_scores_gb.mean()), 4),
        "cv_std": round(float(cv_scores_gb.std()), 4),
        "train_accuracy": round(float(gb.score(X_train, y_train)), 4),
        "test_accuracy": round(float(gb.score(X_test, y_test)), 4),
    }
    print()

    # 4. Pure-Python inference parity check
    print("  --- Pure-Python Inference Parity ---")
    lr_classes = list(lr.classes_)
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
        print(f"  PASS: All {len(X_test)} predictions match sklearn")
    else:
        print(f"  FAIL: {mismatches}/{len(X_test)} mismatches")
    results["parity_mismatches"] = mismatches
    print()

    # Top features per class (logistic regression)
    print("  --- Top Features per Class (Logistic Regression) ---")
    for idx, cls in enumerate(lr.classes_):
        coefs = lr.coef_[idx]
        top_indices = np.argsort(coefs)[-5:][::-1]
        top_feats = [(feature_names[j], round(float(coefs[j]), 3)) for j in top_indices]
        print(f"  {cls}: {', '.join(f'{f}={w}' for f, w in top_feats)}")
    print()

    return results, lr, X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser(description="Classifier distillation POC")
    parser.add_argument("--tickets", type=int, default=400, help="Number of tickets to label")
    parser.add_argument("--labels", type=str, help="Pre-computed labels file (skip LLM calls)")
    parser.add_argument("--vocab-size", type=int, default=200, help="Vocabulary size for BoW")
    parser.add_argument("--save", default="results/classifier_training", help="Output directory")
    args = parser.parse_args()

    print("=" * 70)
    print("CLASSIFIER DISTILLATION POC")
    print("=" * 70)
    print(f"Model:      {MODEL}")
    print(f"Ollama:     {OLLAMA_URL}")
    print(f"Vocab size: {args.vocab_size}")
    print()

    # ── Phase 1: Collect LLM labels ──
    tickets = load_tickets(n=args.tickets)
    print(f"Loaded {len(tickets)} tickets")

    cache_path = os.path.join(args.save, "llm_labels.jsonl")

    if args.labels:
        print(f"\nLoading pre-computed labels from {args.labels}")
        labels = []
        with open(args.labels) as f:
            for line in f:
                labels.append(json.loads(line))
        print(f"  Loaded {len(labels)} labels")
    else:
        system_prompt = open("tasks/helpdesk_triage/triage_system.md").read()
        user_prompt_tpl = open("tasks/helpdesk_triage/triage_user.md").read()

        print(f"\nPhase 1: Collecting LLM labels ({len(tickets)} tickets)")
        labels = collect_labels(tickets, system_prompt, user_prompt_tpl, cache_path)

    print()

    # ── Phase 2: Feature engineering ──
    print("=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)
    print()

    X, feature_names, vocab, priority_values, category_values, ticket_ids = \
        build_features(tickets, labels, vocab_size=args.vocab_size)

    print(f"  Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"  Vocabulary: {len(vocab)} words")
    print(f"  Priority values: {priority_values}")
    print(f"  Category values: {category_values}")
    print(f"  Top 20 vocab: {vocab[:20]}")
    print()

    # Build label arrays
    label_by_id = {rec["ticket_id"]: rec for rec in labels}
    y_rc = []  # root_cause_category
    y_ag = []  # assignment_group
    for tid in ticket_ids:
        y_rc.append(label_by_id[tid]["root_cause_category"])
        y_ag.append(label_by_id[tid]["assignment_group"])

    y_rc = np.array(y_rc)
    y_ag = np.array(y_ag)

    # ── Phase 3 & 4: Train and evaluate ──
    all_results = {}
    all_models = {}

    for field_name, y in [("root_cause_category", y_rc), ("assignment_group", y_ag)]:
        print("=" * 70)
        print(f"FIELD: {field_name}")
        print("=" * 70)
        print()

        results, lr_model, X_train, X_test, y_train, y_test = \
            train_and_evaluate(X, y, field_name, feature_names)

        all_results[field_name] = results
        all_models[field_name] = lr_model

    # ── Phase 6: Export weights ──
    os.makedirs(args.save, exist_ok=True)

    export = {
        "experiment": "classifier_distillation_poc",
        "model": MODEL,
        "n_samples": len(ticket_ids),
        "vocab_size": len(vocab),
        "vocabulary": vocab,
        "priority_values": priority_values,
        "category_values": category_values,
        "feature_names": feature_names,
        "fields": {},
    }

    for field_name, lr_model in all_models.items():
        field_results = all_results[field_name]
        export["fields"][field_name] = {
            "classes": list(lr_model.classes_),
            "weights": lr_model.coef_.tolist(),
            "bias": lr_model.intercept_.tolist(),
            "logistic_regression": {
                k: v for k, v in field_results["logistic_regression"].items()
                if k != "model"
            },
            "decision_tree": field_results["decision_tree"],
            "gradient_boosted": field_results["gradient_boosted"],
            "parity_mismatches": field_results["parity_mismatches"],
        }

    out_path = os.path.join(args.save, "classifier_weights.json")
    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Weights exported to {out_path}")

    # ── Summary ──
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Field':<25s} {'LR CV':>8s} {'LR Test':>8s} {'DT Test':>8s} {'GB Test':>8s} {'Parity':>8s}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for field_name in all_results:
        r = all_results[field_name]
        lr = r["logistic_regression"]
        dt = r["decision_tree"]
        gb = r["gradient_boosted"]
        parity = "PASS" if r["parity_mismatches"] == 0 else "FAIL"
        print(f"  {field_name:<25s} {lr['cv_mean']:>7.1%} {lr['test_accuracy']:>7.1%} "
              f"{dt['test_accuracy']:>7.1%} {gb['test_accuracy']:>7.1%} {parity:>8s}")
    print()

    # Print proposed YAML rule
    print("=" * 70)
    print("PROPOSED CLASSIFIER RULES (YAML)")
    print("=" * 70)
    print()
    for field_name, lr_model in all_models.items():
        r = all_results[field_name]["logistic_regression"]
        print(f"- path: assessment.{field_name}")
        print(f"  type: classifier")
        print(f"  classifier_type: logistic_regression")
        print(f"  classes: {json.dumps(list(lr_model.classes_))}")
        print(f"  vocabulary: <{len(vocab)} words>")
        print(f"  weights: <{lr_model.coef_.shape[0]}x{lr_model.coef_.shape[1]} matrix>")
        print(f"  bias: {json.dumps([round(b, 4) for b in lr_model.intercept_.tolist()])}")
        print(f"  confidence_threshold: 0.95")
        print(f"  # CV accuracy: {r['cv_mean']:.1%}, Test accuracy: {r['test_accuracy']:.1%}")
        print()


if __name__ == "__main__":
    main()
