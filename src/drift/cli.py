from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()
import yaml

from drift.providers.factory import make_provider
from drift.pipeline.pipeline import run_pipeline
from drift.storage import make_run_dir, save_run, load_runs, save_metrics
from drift.metrics.text import compute_all
from drift.tasks import resolve_task_prompts


async def _run_baseline(config: dict, num_runs: int, output_dir: str) -> Path:
    """Run a single-call experiment num_runs times."""
    exp = config["experiment"]
    provider = make_provider(exp)
    system = exp["system_prompt"]
    prompt = exp["user_prompt"]
    if "input" in exp:
        prompt = prompt.replace("{input}", exp["input"])
    run_dir = make_run_dir(output_dir, exp["name"])

    click.echo(f"Experiment: {exp['name']}")
    click.echo(f"Type: baseline")
    click.echo(f"Model: {exp['model']}")
    click.echo(f"Runs: {num_runs}")
    click.echo(f"Output: {run_dir}")
    click.echo()

    for i in range(num_runs):
        click.echo(f"  Run {i + 1}/{num_runs}...", nl=False)
        response = await provider.complete(system, prompt)
        result = {
            "run_index": i,
            "type": "baseline",
            "response": response.to_dict(),
            "config": exp,
        }
        save_run(run_dir, i, result)
        click.echo(f" {response.total_ms:.0f}ms")

    click.echo(f"\nDone. Results saved to {run_dir}")
    return run_dir


async def _run_pipeline(config: dict, num_runs: int, output_dir: str) -> Path:
    """Run a multi-step pipeline experiment num_runs times."""
    exp = config["experiment"]
    steps = exp["steps"]
    initial_input = exp["input"]
    run_dir = make_run_dir(output_dir, exp["name"])

    step_models = [f"{s.get('provider','ollama')}:{s['model']}" for s in steps]

    click.echo(f"Experiment: {exp['name']}")
    click.echo(f"Type: pipeline ({len(steps)} steps)")
    click.echo(f"Models: {' → '.join(step_models)}")
    click.echo(f"Runs: {num_runs}")
    click.echo(f"Output: {run_dir}")
    click.echo()

    for i in range(num_runs):
        click.echo(f"  Run {i + 1}/{num_runs}:")
        task_dir = (Path.cwd() / "tasks" / exp["task"]) if "task" in exp else None
        step_results = await run_pipeline(steps, initial_input, task_dir=task_dir)
        result = {
            "run_index": i,
            "type": "pipeline",
            "steps": [{"name": sr.name, "response": sr.response} for sr in step_results],
            "config": exp,
        }
        save_run(run_dir, i, result)

    click.echo(f"\nDone. Results saved to {run_dir}")
    return run_dir


@click.group()
def cli():
    """Drift — measure LLM output variability."""
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--runs", default=30, help="Number of identical runs.")
@click.option("--output-dir", default="results", help="Output directory.")
def run(config_path: str, runs: int, output_dir: str):
    """Run an experiment from a YAML config."""
    config = yaml.safe_load(Path(config_path).read_text())
    config = resolve_task_prompts(config, Path.cwd() / "tasks")
    exp_type = config["experiment"].get("type", "baseline")
    if exp_type == "pipeline":
        asyncio.run(_run_pipeline(config, runs, output_dir))
    else:
        asyncio.run(_run_baseline(config, runs, output_dir))


@cli.command()
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output path for metrics JSON.")
def measure(results_dir: str, output: str | None):
    """Compute text metrics on collected results."""
    runs = load_runs(results_dir)
    if not runs:
        click.echo("No run files found.", err=True)
        sys.exit(1)

    run_type = runs[0].get("type", "baseline")

    if run_type == "pipeline":
        metrics = _measure_pipeline(runs)
    else:
        metrics = _measure_baseline(runs)

    if output:
        save_metrics(output, metrics)
        click.echo(f"Metrics saved to {output}")
    else:
        click.echo(json.dumps(metrics, indent=2))


def _measure_baseline(runs: list[dict]) -> dict:
    outputs = [r["response"]["content"] for r in runs]
    metrics = compute_all(outputs)
    metrics["type"] = "baseline"
    metrics["model"] = runs[0]["response"]["model"]
    metrics["provider"] = runs[0]["response"]["provider"]
    timing = [r["response"]["total_ms"] for r in runs]
    metrics["mean_latency_ms"] = round(sum(timing) / len(timing), 1)
    return metrics


def _measure_pipeline(runs: list[dict]) -> dict:
    num_steps = len(runs[0]["steps"])
    step_names = [s["name"] for s in runs[0]["steps"]]

    per_step = []
    for step_idx in range(num_steps):
        outputs = [r["steps"][step_idx]["response"]["content"] for r in runs]
        step_metrics = compute_all(outputs)
        step_metrics["name"] = step_names[step_idx]
        step_metrics["model"] = runs[0]["steps"][step_idx]["response"]["model"]
        step_metrics["provider"] = runs[0]["steps"][step_idx]["response"]["provider"]
        timing = [r["steps"][step_idx]["response"]["total_ms"] for r in runs]
        step_metrics["mean_latency_ms"] = round(sum(timing) / len(timing), 1)
        per_step.append(step_metrics)

    # Final step = overall pipeline output
    final_outputs = [r["steps"][-1]["response"]["content"] for r in runs]
    final_metrics = compute_all(final_outputs)

    return {
        "type": "pipeline",
        "num_steps": num_steps,
        "num_runs": len(runs),
        "per_step": per_step,
        "final": final_metrics,
        "drift_curve": {
            "exact_match_rate": [s["exact_match_rate"] for s in per_step],
            "mean_edit_distance": [s["mean_edit_distance"] for s in per_step],
            "num_unique": [s["num_unique"] for s in per_step],
        },
    }


if __name__ == "__main__":
    cli()
