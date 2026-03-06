from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

from drift.providers.base import LLMResponse


def make_run_dir(output_dir: str, experiment_name: str) -> Path:
    """Create and return a timestamped directory for an experiment session."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / experiment_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run(run_dir: Path, run_index: int, data: dict) -> Path:
    """Save a single run's results to JSON."""
    path = run_dir / f"run_{run_index:03d}.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def load_runs(run_dir: str | Path) -> list[dict]:
    """Load all run JSON files from a directory, sorted by filename."""
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("run_*.json"))
    return [json.loads(f.read_text()) for f in files]


def save_metrics(output_path: str | Path, metrics: dict) -> Path:
    """Save computed metrics to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    return path


def load_metrics(path: str | Path) -> dict:
    """Load metrics from JSON."""
    return json.loads(Path(path).read_text())
