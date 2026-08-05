#!/usr/bin/env python3
"""Cross-model probe transfer (Phase 4, Decision Gate 4).

Trains probes on a SMALL model's hidden states against the 32B model's
output labels. If small-model representations linearly encode the large
model's decisions, the Tier 4 runtime becomes small-model prefill + matmul.

Compares, per field and layer:
  - transfer probe (small hidden -> 32B labels): CV accuracy, coverage@0.95
  - the 32B self-probe numbers from results/layer_sweep (if present)

Usage:
    python scripts/probe_transfer_test.py \
        [--model-dir hidden_Qwen2.5-0.5B-Instruct-4bit] [--step 4]
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
SWEEP_DIR = PROJECT_ROOT / "results" / "layer_sweep"

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


def load_corpus(hidden_dir: Path):
    outputs, states = [], []
    for line in (CORPUS_DIR / "labels.jsonl").read_text().splitlines():
        rec = json.loads(line)
        npz = hidden_dir / f"{rec['ticket_id']}.npz"
        if rec.get("output") is None or not npz.exists():
            continue
        outputs.append(rec["output"])
        states.append(np.load(npz)["states"])
    return outputs, np.stack(states)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="hidden_Qwen2.5-0.5B-Instruct-4bit",
                        help="Subdirectory of the corpus with small-model hidden states.")
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()

    hidden_dir = CORPUS_DIR / args.model_dir
    meta = json.loads((hidden_dir / "meta.json").read_text())
    outputs, hidden = load_corpus(hidden_dir)
    n, n_layers, d = hidden.shape
    print(f"Transfer corpus: {n} tickets, {n_layers} layers, d_model={d}")
    print(f"Small model: {meta['model']}")
    print(f"Labels: 32B outputs (teacher)\n")

    layers = sorted(set(list(range(0, n_layers, args.step)) + [n_layers - 1]))
    report = {"small_model": meta["model"], "n": n, "threshold": args.threshold,
              "fields": {}}

    for field in FIELDS:
        y_all = np.array([str(get_nested(o, field)) for o in outputs])
        valid = y_all != "None"
        y = y_all[valid]
        if len(set(y)) < 2:
            continue

        print(f"{field}  (n={valid.sum()}, classes={len(set(y))})")
        print(f"  {'layer':>5} {'CV acc':>7} {'cover':>7} {'acc@cov':>8}")
        curve = {}
        for layer in layers:
            X = hidden[valid, layer, :].astype(np.float32)
            probe = train_hidden_probe(X, y, layer=layer, probe_model=meta["model"])
            if probe is None:
                continue
            cov, acc_cov = coverage_at_threshold(
                probe["oof_probs"], probe["oof_y"], probe["classes"], args.threshold)
            curve[layer] = {"cv_accuracy": probe["cv_accuracy"],
                            "coverage": round(cov, 4),
                            "accuracy_on_covered":
                                round(acc_cov, 4) if acc_cov is not None else None}
            print(f"  {layer:>5} {probe['cv_accuracy']:>7.3f} {cov:>7.1%} "
                  f"{(f'{acc_cov:.3f}' if acc_cov is not None else '—'):>8}")

        acc_by_layer = {l: c["cv_accuracy"] for l, c in curve.items()}
        lstar = select_lstar(acc_by_layer)
        best_layer = max(acc_by_layer, key=acc_by_layer.get)

        # Pull the 32B self-probe reference if the sweep has run
        ref = None
        ref_path = SWEEP_DIR / f"{field.replace('.', '_')}.json"
        if ref_path.exists():
            ref_data = json.loads(ref_path.read_text())
            ref = {"best_accuracy": ref_data["best_accuracy"],
                   "best_layer": ref_data["best_layer"], "lstar": ref_data["lstar"]}

        line = (f"  transfer best: {acc_by_layer[best_layer]:.3f} @ layer {best_layer}"
                f"  (L*={lstar})")
        if ref:
            line += f"   | 32B self-probe best: {ref['best_accuracy']:.3f} @ layer {ref['best_layer']}"
        print(line + "\n")

        report["fields"][field] = {
            "curve": curve, "lstar": lstar, "best_layer": best_layer,
            "best_accuracy": acc_by_layer[best_layer], "reference_32b": ref,
        }

    out = CORPUS_DIR / f"transfer_{args.model_dir}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Report saved to {out}")


if __name__ == "__main__":
    main()
