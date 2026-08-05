#!/usr/bin/env python3
"""Decision Gate 1 benchmark: Tier 4 hidden-state probes vs Tier 3 bag-of-words.

Same labels, same cross-validation protocol, same confidence threshold —
the only difference is the feature matrix:
  Tier 3: binary bag-of-words (150-word vocab) + one-hot stated_priority/category
  Tier 4: residual-stream vector at one layer, last input token

Reports per field: CV accuracy, coverage@threshold (fraction the rule
handles without LLM escalation), accuracy on covered, ECE.

Usage:
    python scripts/hidden_probe_vs_tier3.py [--layer 38] [--threshold 0.95]
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from drift.hybrid.probes import coverage_at_threshold, train_hidden_probe

PROJECT_ROOT = Path(__file__).parent.parent
CORPUS_DIR = PROJECT_ROOT / "results" / "hidden_probe_corpus"
DATASET_PATH = PROJECT_ROOT / "datasets" / "helpdesk_tickets.jsonl"

FIELDS = [
    "assessment.severity",
    "assessment.root_cause_category",
    "assessment.assignment_group",
]

# Same stopword list as autogen Tier 3
from drift.hybrid.autogen import STOPWORDS  # noqa: E402


def format_ticket(ticket: dict) -> str:
    return "\n".join([
        f"Ticket ID:    {ticket['id']}",
        f"Subject:      {ticket['subject']}",
        f"From:         {ticket['requesterEmail']}",
        f"Created:      {ticket['createdAt']}",
        f"Priority:     {ticket['priority']}",
        f"Category:     {ticket['category']}",
        "",
        "Description:",
        ticket["description"],
    ])


def get_nested(obj, path):
    for key in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return None
    return obj


def load_corpus(layer: int):
    tickets = {json.loads(l)["id"]: json.loads(l)
               for l in DATASET_PATH.read_text().splitlines()}
    rows = []
    for line in (CORPUS_DIR / "labels.jsonl").read_text().splitlines():
        rec = json.loads(line)
        npz = CORPUS_DIR / "hidden" / f"{rec['ticket_id']}.npz"
        if rec.get("output") is None or not npz.exists():
            continue
        states = np.load(npz)["states"]  # [n_layers, d_model] float16
        rows.append({
            "ticket": tickets[rec["ticket_id"]],
            "output": rec["output"],
            "hidden": states[layer].astype(np.float32),
        })
    return rows


def bow_features(rows):
    """Tier 3 feature matrix: binary BoW + one-hot stated priority/category."""
    texts = [format_ticket(r["ticket"]) for r in rows]
    word_counts = Counter()
    for text in texts:
        words = set(re.findall(r"[a-z]+", text.lower()))
        word_counts.update(w for w in words if w not in STOPWORDS and len(w) >= 3)
    vocab = [w for w, _ in word_counts.most_common(150)]
    vocab_index = {w: i for i, w in enumerate(vocab)}

    priorities = sorted({r["ticket"]["priority"] for r in rows})
    categories = sorted({r["ticket"]["category"] for r in rows})

    n_features = len(vocab) + len(priorities) + len(categories)
    X = np.zeros((len(rows), n_features), dtype=np.float32)
    for i, (r, text) in enumerate(zip(rows, texts)):
        for w in re.findall(r"[a-z]+", text.lower()):
            if w in vocab_index:
                X[i, vocab_index[w]] = 1.0
        X[i, len(vocab) + priorities.index(r["ticket"]["priority"])] = 1.0
        X[i, len(vocab) + len(priorities) + categories.index(r["ticket"]["category"])] = 1.0
    return X


def ece(probs, y_true, classes, n_bins=10):
    """Expected calibration error on top-class confidence."""
    probs = np.asarray(probs)
    top = probs.max(axis=1)
    pred = np.asarray(classes)[probs.argmax(axis=1)]
    correct = (pred == np.asarray(y_true)).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (top > lo) & (top <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - top[mask].mean())
    return total


def evaluate(name, X, y, layer, threshold):
    probe = train_hidden_probe(
        X, y, layer=layer, probe_model="benchmark", min_class_count=3,
    )
    if probe is None:
        return {"name": name, "trainable": False}
    cov, acc_cov = coverage_at_threshold(
        probe["oof_probs"], probe["oof_y"], probe["classes"], threshold)
    return {
        "name": name,
        "trainable": True,
        "n": probe["n_samples"],
        "n_classes": len(probe["classes"]),
        "cv_accuracy": probe["cv_accuracy"],
        "best_C": probe["best_C"],
        "coverage": round(cov, 4),
        "accuracy_on_covered": round(acc_cov, 4) if acc_cov is not None else None,
        "ece": round(ece(probe["oof_probs"], probe["oof_y"], probe["classes"]), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=38)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = load_corpus(args.layer)
    print(f"Corpus: {len(rows)} tickets with labels + hidden states")
    print(f"Layer: {args.layer}   Confidence threshold: {args.threshold}")

    X_bow = bow_features(rows)
    X_hidden = np.stack([r["hidden"] for r in rows])
    print(f"Tier 3 features: {X_bow.shape}   Tier 4 features: {X_hidden.shape}\n")

    report = {"n_tickets": len(rows), "layer": args.layer,
              "threshold": args.threshold, "fields": {}}

    header = (f"{'field':<34} {'feat':<7} {'n':>4} {'cls':>4} {'CV acc':>7} "
              f"{'cover':>7} {'acc@cov':>8} {'ECE':>6}")
    print(header)
    print("-" * len(header))

    for field in FIELDS:
        y_all = np.array([str(get_nested(r["output"], field)) for r in rows])
        valid = y_all != "None"
        y = y_all[valid]
        if len(set(y)) < 2:
            print(f"{field:<34} (single class: {set(y)}) — skipped")
            continue

        results = {}
        for feat_name, X in (("bow", X_bow[valid]), ("hidden", X_hidden[valid])):
            r = evaluate(feat_name, X, y, args.layer, args.threshold)
            results[feat_name] = r
            if r["trainable"]:
                acc_cov = r["accuracy_on_covered"]
                print(f"{field:<34} {feat_name:<7} {r['n']:>4} {r['n_classes']:>4} "
                      f"{r['cv_accuracy']:>7.3f} {r['coverage']:>7.1%} "
                      f"{(f'{acc_cov:.3f}' if acc_cov is not None else '—'):>8} "
                      f"{r['ece']:>6.3f}")
            else:
                print(f"{field:<34} {feat_name:<7} untrainable")
        report["fields"][field] = results
        print()

    # Gate 1 summary
    print("=" * len(header))
    print("DECISION GATE 1 (target: hidden coverage >= 2x bow coverage, >= 50%)")
    for field, res in report["fields"].items():
        b, h = res.get("bow", {}), res.get("hidden", {})
        if b.get("trainable") and h.get("trainable"):
            ratio = (h["coverage"] / b["coverage"]) if b["coverage"] > 0 else float("inf")
            verdict = "PASS" if (h["coverage"] >= 0.5 and ratio >= 2.0) else \
                      ("partial" if h["coverage"] > b["coverage"] else "FAIL")
            print(f"  {field:<34} bow={b['coverage']:.1%}  hidden={h['coverage']:.1%}  "
                  f"({ratio:.1f}x)  [{verdict}]")

    out = args.output or str(CORPUS_DIR / f"gate1_layer{args.layer}.json")
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
