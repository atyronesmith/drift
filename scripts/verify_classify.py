#!/usr/bin/env python3
"""Verify classifier engine predictions against LLM labels.

Runs the classify prompt on all summaries via Ollama, then compares
to the engine's predictions.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

OLLAMA_MODEL = "qwen2.5:32b"
OLLAMA_HOST = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_SCRIPT = REPO_ROOT / "engines" / "ansible_classify.py"

SYSTEM_PROMPT = Path("tasks/ansible_classify/system.md").read_text()
USER_TEMPLATE = Path("tasks/ansible_classify/user.md").read_text()


def call_ollama(system: str, user: str) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 50},
    })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(payload)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["curl", "-s", "-X", "POST",
             f"{OLLAMA_HOST}/api/chat",
             "-H", "Content-Type: application/json",
             "-d", f"@{tmp_path}"],
            capture_output=True, text=True, timeout=60,
        )
    finally:
        os.unlink(tmp_path)

    resp = json.loads(result.stdout)
    text = resp.get("message", {}).get("content", "").strip()
    try:
        return json.loads(text).get("category", text)
    except json.JSONDecodeError:
        return text


def run_engine(filepath: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(ENGINE_SCRIPT), filepath],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return {}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def main():
    summary_dir = Path("datasets/ansible_summaries")
    files = sorted(summary_dir.glob("*.txt"))

    print(f"Verifying {len(files)} summaries: engine vs LLM")
    print(f"{'='*80}\n")

    match = 0
    engine_classified = 0
    total = 0
    results = []

    for f in files:
        total += 1
        summary = f.read_text().strip()

        # Engine prediction
        engine_result = run_engine(str(f))
        engine_cat = engine_result.get("category")

        # LLM prediction
        user = USER_TEMPLATE.replace("{input}", summary)
        llm_cat = call_ollama(SYSTEM_PROMPT, user)

        if engine_cat:
            engine_classified += 1
            correct = engine_cat == llm_cat
            if correct:
                match += 1
            status = "OK" if correct else "MISMATCH"
        else:
            status = "SKIP (low confidence)"
            correct = None

        print(f"[{total:2d}] {f.stem}")
        print(f"     LLM:    {llm_cat}")
        if engine_cat:
            print(f"     Engine: {engine_cat}  [{status}]")
        else:
            print(f"     Engine: — [{status}]")
        print()

        results.append({
            "file": f.name,
            "summary": summary[:100],
            "llm_category": llm_cat,
            "engine_category": engine_cat,
            "correct": correct,
        })

    print(f"{'='*80}")
    print(f"RESULTS")
    print(f"  Engine classified: {engine_classified}/{total} ({engine_classified/total*100:.0f}%)")
    if engine_classified:
        print(f"  Accuracy (where classified): {match}/{engine_classified} ({match/engine_classified*100:.0f}%)")
    print(f"  Escalated to LLM: {total - engine_classified}/{total}")

    # Distribution of LLM categories
    from collections import Counter
    cats = Counter(r["llm_category"] for r in results)
    print(f"\n  LLM category distribution:")
    for cat, count in cats.most_common():
        print(f"    {cat}: {count}")

    Path("results/classify_verification.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved to results/classify_verification.json")


if __name__ == "__main__":
    main()
