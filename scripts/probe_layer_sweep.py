#!/usr/bin/env python3
"""Layer sweep: per-field reasoning-depth profiles (Phase 2).

Trains a probe at every layer for each classification field and reports
the accuracy/coverage-vs-depth curve plus L* (earliest layer reaching 95%
of the field's best accuracy). L* is the truncation point for Phase 3.

Usage:
    python scripts/probe_layer_sweep.py [--step 4] [--threshold 0.95]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from drift.hybrid.probes import (
    coverage_at_threshold,
    select_lstar,
    train_hidden_probe,
)

PROJECT_ROOT = Path(__file__).parent.parent
CORPUS_DIR = PROJECT_ROOT / "results" / "hidden_probe_corpus"
OUT_DIR = PROJECT_ROOT / "results" / "layer_sweep"

FIELDS = [
    "assessment.severity",
    "assessment.root_cause_category",
    "assessment.assignment_group",
]


def get_nested(obj, path):
    for key in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return None
    return obj


def load_corpus():
    """Return (labels list of output dicts, hidden array [n, n_layers, d])."""
    outputs, states = [], []
    for line in (CORPUS_DIR / "labels.jsonl").read_text().splitlines():
        rec = json.loads(line)
        npz = CORPUS_DIR / "hidden" / f"{rec['ticket_id']}.npz"
        if rec.get("output") is None or not npz.exists():
            continue
        outputs.append(rec["output"])
        states.append(np.load(npz)["states"])  # [n_layers, d] float16
    return outputs, np.stack(states)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=4, help="Layer stride for the sweep.")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--fields", nargs="+", default=FIELDS)
    args = parser.parse_args()

    outputs, hidden = load_corpus()
    n, n_layers, d = hidden.shape
    print(f"Corpus: {n} tickets, {n_layers} layers, d_model={d}")
    layers = sorted(set(list(range(0, n_layers, args.step)) + [n_layers - 1]))
    print(f"Sweeping {len(layers)} layers (stride {args.step})\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for field in args.fields:
        y_all = np.array([str(get_nested(o, field)) for o in outputs])
        valid = y_all != "None"
        y = y_all[valid]
        if len(set(y)) < 2:
            print(f"{field}: single class — skipped")
            continue

        print(f"{field}  (n={valid.sum()}, classes={len(set(y))})")
        print(f"  {'layer':>5} {'CV acc':>7} {'cover':>7} {'acc@cov':>8}")

        curve = {}
        for layer in layers:
            X = hidden[valid, layer, :].astype(np.float32)
            probe = train_hidden_probe(X, y, layer=layer, probe_model="sweep")
            if probe is None:
                continue
            cov, acc_cov = coverage_at_threshold(
                probe["oof_probs"], probe["oof_y"], probe["classes"], args.threshold)
            curve[layer] = {
                "cv_accuracy": probe["cv_accuracy"],
                "coverage": round(cov, 4),
                "accuracy_on_covered": round(acc_cov, 4) if acc_cov is not None else None,
                "best_C": probe["best_C"],
            }
            print(f"  {layer:>5} {probe['cv_accuracy']:>7.3f} {cov:>7.1%} "
                  f"{(f'{acc_cov:.3f}' if acc_cov is not None else '—'):>8}")

        acc_by_layer = {l: c["cv_accuracy"] for l, c in curve.items()}
        lstar = select_lstar(acc_by_layer)
        best_layer = max(acc_by_layer, key=acc_by_layer.get)
        print(f"  L* = {lstar}  (best acc {acc_by_layer[best_layer]:.3f} at layer "
              f"{best_layer}; L* reaches {acc_by_layer[lstar]:.3f})\n")

        summary[field] = {
            "curve": curve, "lstar": lstar, "best_layer": best_layer,
            "best_accuracy": acc_by_layer[best_layer],
            "n": int(valid.sum()),
        }
        safe = field.replace(".", "_")
        (OUT_DIR / f"{safe}.json").write_text(json.dumps(summary[field], indent=2))

    (OUT_DIR / "summary.json").write_text(json.dumps(
        {f: {k: v for k, v in s.items() if k != "curve"} for f, s in summary.items()},
        indent=2))
    print(f"Curves saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
