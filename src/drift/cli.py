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
from drift.pipeline.pipeline import run_pipeline, StepResult, LoopStepResult
from drift.storage import make_run_dir, save_run, load_runs, save_metrics
from drift.metrics.text import compute_all
from drift.metrics.iteration import compute_iteration_metrics
from drift.metrics.functional import compute_functional_metrics
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

    step_labels = []
    for s in steps:
        if s.get("type") == "loop":
            step_labels.append(f"loop:{s.get('name', 'loop')}")
        else:
            step_labels.append(f"{s.get('provider','ollama')}:{s['model']}")

    click.echo(f"Experiment: {exp['name']}")
    click.echo(f"Type: pipeline ({len(steps)} steps)")
    click.echo(f"Models: {' → '.join(step_labels)}")
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
            "steps": _serialize_results(step_results),
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
@click.option(
    "--functional-step", default=None,
    help="Step name to run functional equivalence tests on (e.g., 'implementer').",
)
@click.option(
    "--functional-tests", "functional_tests_path", default=None,
    type=click.Path(exists=True),
    help="Path to functional test module (auto-discovered from task if omitted).",
)
@click.option(
    "--tasks-dir", default="tasks",
    help="Tasks directory (default: tasks/).",
)
def measure(results_dir: str, output: str | None, functional_step: str | None,
            functional_tests_path: str | None, tasks_dir: str):
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

    if functional_step and run_type == "pipeline":
        ft_path = Path(functional_tests_path) if functional_tests_path else None
        func_metrics = _measure_functional(runs, functional_step, Path(tasks_dir), ft_path)
        if func_metrics:
            metrics["functional"] = func_metrics

    if output:
        save_metrics(output, metrics)
        click.echo(f"Metrics saved to {output}")
    else:
        click.echo(json.dumps(metrics, indent=2))


def _serialize_results(step_results: list) -> list[dict]:
    """Serialize a mix of StepResult and LoopStepResult to dicts."""
    out = []
    for sr in step_results:
        if isinstance(sr, LoopStepResult):
            out.append({
                "name": sr.name,
                "type": "loop",
                "total_iterations": sr.total_iterations,
                "termination_reason": sr.termination_reason,
                "iterations": [
                    [{"name": s.name, "response": s.response} for s in iteration]
                    for iteration in sr.iterations
                ],
            })
        else:
            out.append({"name": sr.name, "response": sr.response})
    return out


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
    has_loops = any(s.get("type") == "loop" for s in runs[0]["steps"])

    per_step = []
    for step_idx in range(num_steps):
        step_data = runs[0]["steps"][step_idx]

        if step_data.get("type") == "loop":
            loop_metrics = compute_iteration_metrics(runs, step_idx)
            loop_metrics["name"] = step_data["name"]
            loop_metrics["type"] = "loop"
            per_step.append(loop_metrics)
        else:
            outputs = [r["steps"][step_idx]["response"]["content"] for r in runs]
            step_metrics = compute_all(outputs)
            step_metrics["name"] = step_names[step_idx]
            step_metrics["model"] = runs[0]["steps"][step_idx]["response"]["model"]
            step_metrics["provider"] = runs[0]["steps"][step_idx]["response"]["provider"]
            timing = [r["steps"][step_idx]["response"]["total_ms"] for r in runs]
            step_metrics["mean_latency_ms"] = round(sum(timing) / len(timing), 1)
            per_step.append(step_metrics)

    # Final pipeline output — last step or last iteration of last loop
    final_outputs = _get_final_outputs(runs)
    final_metrics = compute_all(final_outputs)

    result = {
        "type": "pipeline",
        "num_steps": num_steps,
        "num_runs": len(runs),
        "per_step": per_step,
        "final": final_metrics,
        "drift_curve": {
            "exact_match_rate": [s["exact_match_rate"] for s in per_step if "exact_match_rate" in s],
            "mean_edit_distance": [s["mean_edit_distance"] for s in per_step if "mean_edit_distance" in s],
            "num_unique": [s["num_unique"] for s in per_step if "num_unique" in s],
        },
    }
    return result


def _measure_functional(
    runs: list[dict], step_name: str, tasks_dir: Path,
    explicit_path: Path | None = None,
) -> dict | None:
    """Run functional equivalence tests on a named pipeline step."""
    if explicit_path:
        test_module_path = explicit_path
    else:
        task = runs[0].get("config", {}).get("task")
        if not task:
            click.echo("Warning: no task in config, cannot locate functional tests. "
                        "Use --functional-tests to specify the path.", err=True)
            return None
        test_module_path = tasks_dir / task / "functional_tests.py"

    if not test_module_path.exists():
        click.echo(f"Warning: {test_module_path} not found, skipping functional tests.", err=True)
        return None

    # Find the step index by name
    step_names = [s["name"] for s in runs[0]["steps"]]
    if step_name not in step_names:
        click.echo(f"Warning: step '{step_name}' not found in [{', '.join(step_names)}].", err=True)
        return None

    step_idx = step_names.index(step_name)
    outputs = [r["steps"][step_idx]["response"]["content"] for r in runs]

    click.echo(f"Running functional tests on step '{step_name}' ({len(outputs)} variants)...")
    metrics = compute_functional_metrics(outputs, test_module_path)
    click.echo(
        f"  {metrics['variants_all_pass']}/{metrics['num_variants']} variants pass all tests "
        f"({metrics['functional_equivalence_rate']:.1%} equivalence rate)"
    )
    if metrics["failures"]:
        for f in metrics["failures"]:
            click.echo(f"  variant {f['variant']}: {', '.join(f['failed_tests'].keys())}")

    metrics["step"] = step_name
    return metrics


def _get_final_outputs(runs: list[dict]) -> list[str]:
    """Extract the final output from each run, handling loop steps."""
    outputs = []
    for r in runs:
        last_step = r["steps"][-1]
        if last_step.get("type") == "loop":
            last_iter = last_step["iterations"][-1]
            outputs.append(last_iter[-1]["response"]["content"])
        else:
            outputs.append(last_step["response"]["content"])
    return outputs


if __name__ == "__main__":
    cli()
