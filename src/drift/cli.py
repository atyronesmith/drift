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
from drift.metrics.text import compute_all, pairwise_edit_distances
from drift.metrics.iteration import compute_iteration_metrics
from drift.metrics.functional import compute_functional_metrics
from drift.metrics.stats import bootstrap_ci, permutation_test
from drift.tasks import resolve_task_prompts
from drift.analysis import analyze_metrics, format_report


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
        response = await provider.complete(system, prompt, temperature=exp.get("temperature", 0))
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
    ft_path = Path(functional_tests_path) if functional_tests_path else None
    metrics = _compute_metrics(results_dir, functional_step, ft_path, Path(tasks_dir))

    if output:
        save_metrics(output, metrics)
        click.echo(f"Metrics saved to {output}")
    else:
        click.echo(json.dumps(metrics, indent=2))


def _compute_metrics(
    results_dir: str,
    functional_step: str | None = None,
    functional_tests_path: Path | None = None,
    tasks_dir: Path = Path("tasks"),
) -> dict:
    """Core metrics computation. Used by both measure and analyze commands."""
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
        func_metrics = _measure_functional(runs, functional_step, tasks_dir, functional_tests_path)
        if func_metrics:
            metrics["functional"] = func_metrics
    elif run_type == "baseline":
        func_metrics = _measure_functional_baseline(runs, tasks_dir, functional_tests_path)
        if func_metrics:
            metrics["functional"] = func_metrics

    return metrics


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON instead of text report.")
@click.option(
    "--functional-step", default=None,
    help="Step name for functional tests (pipeline results dirs only).",
)
@click.option(
    "--functional-tests", "functional_tests_path", default=None,
    type=click.Path(exists=True),
    help="Path to functional test module (auto-discovered from task if omitted).",
)
@click.option("--tasks-dir", default="tasks", help="Tasks directory (default: tasks/).")
def analyze(path: str, as_json: bool, functional_step: str | None,
            functional_tests_path: str | None, tasks_dir: str):
    """Analyze metrics and recommend: distill, keep, or fix.

    PATH can be a metrics JSON file or a results directory.
    """
    p = Path(path)
    if p.is_dir():
        ft_path = Path(functional_tests_path) if functional_tests_path else None
        metrics = _compute_metrics(path, functional_step, ft_path, Path(tasks_dir))
    else:
        metrics = json.loads(p.read_text())

    report = analyze_metrics(metrics)
    if as_json:
        click.echo(json.dumps(report, indent=2))
    else:
        click.echo(format_report(report))


@cli.command()
@click.argument("path_a", type=click.Path(exists=True))
@click.argument("path_b", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@click.option("--functional-step", default=None, help="Step name for functional tests.")
@click.option("--functional-tests", "functional_tests_path", default=None,
              type=click.Path(exists=True), help="Path to functional test module.")
@click.option("--tasks-dir", default="tasks", help="Tasks directory.")
def compare(path_a: str, path_b: str, as_json: bool, functional_step: str | None,
            functional_tests_path: str | None, tasks_dir: str):
    """Compare two result sets side by side.

    Each PATH can be a metrics JSON file or a results directory.
    """
    ft_path = Path(functional_tests_path) if functional_tests_path else None
    metrics_a = _load_or_compute(path_a, functional_step, ft_path, Path(tasks_dir))
    metrics_b = _load_or_compute(path_b, functional_step, ft_path, Path(tasks_dir))
    result = _build_comparison(metrics_a, metrics_b, path_a, path_b)
    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(_format_comparison(result))


def _load_or_compute(path: str, functional_step: str | None = None,
                     functional_tests_path: Path | None = None,
                     tasks_dir: Path = Path("tasks")) -> dict:
    """Load metrics from JSON file or compute from results directory."""
    p = Path(path)
    if p.is_dir():
        return _compute_metrics(path, functional_step, functional_tests_path, tasks_dir)
    return json.loads(p.read_text())


def _build_comparison(a: dict, b: dict, label_a: str, label_b: str) -> dict:
    """Build a structured comparison between two metric sets."""
    def _get(m, key, default=None):
        return m.get(key, m.get("final", {}).get(key, default))

    comparison = {
        "a": {"label": label_a, "type": a.get("type")},
        "b": {"label": label_b, "type": b.get("type")},
        "metrics": {},
    }

    keys = [
        ("exact_match_rate", "Exact match", ".0%"),
        ("num_unique", "Unique outputs", "d"),
        ("mean_edit_distance", "Mean edit distance", ".4f"),
        ("length_cv", "Length CV", ".3f"),
    ]
    for key, label, _ in keys:
        val_a = _get(a, key)
        val_b = _get(b, key)
        comparison["metrics"][key] = {"label": label, "a": val_a, "b": val_b}

    # Latency
    lat_a = a.get("mean_latency_ms")
    lat_b = b.get("mean_latency_ms")
    comparison["metrics"]["mean_latency_ms"] = {"label": "Mean latency", "a": lat_a, "b": lat_b}

    # Model info
    comparison["a"]["model"] = a.get("model") or _pipeline_models(a)
    comparison["b"]["model"] = b.get("model") or _pipeline_models(b)

    # Functional equivalence
    func_a = a.get("functional", {})
    func_b = b.get("functional", {})
    if func_a or func_b:
        comparison["functional"] = {
            "a": func_a.get("functional_equivalence_rate"),
            "b": func_b.get("functional_equivalence_rate"),
            "a_bugs": func_a.get("variants_with_bugs"),
            "b_bugs": func_b.get("variants_with_bugs"),
            "a_ci": func_a.get("ci_95"),
            "b_ci": func_b.get("ci_95"),
        }

    # Confidence intervals for edit distance
    ci_a = a.get("edit_distance_ci_95") or _get(a, "edit_distance_ci_95")
    ci_b = b.get("edit_distance_ci_95") or _get(b, "edit_distance_ci_95")
    if ci_a or ci_b:
        comparison["edit_distance_ci"] = {"a": ci_a, "b": ci_b}

    return comparison


def _pipeline_models(m: dict) -> str | None:
    steps = m.get("per_step", [])
    models = [s.get("model", "") for s in steps if s.get("model")]
    return " → ".join(models) if models else None


def _format_comparison(comp: dict) -> str:
    lines = []
    lines.append("DRIFT COMPARISON")
    lines.append("=" * 60)

    a_label = Path(comp["a"]["label"]).name
    b_label = Path(comp["b"]["label"]).name

    lines.append(f"  A: {a_label}")
    if comp["a"].get("model"):
        lines.append(f"     Model: {comp['a']['model']}")
    lines.append(f"  B: {b_label}")
    if comp["b"].get("model"):
        lines.append(f"     Model: {comp['b']['model']}")
    lines.append("")

    lines.append(f"  {'Metric':<24} {'A':>12} {'B':>12} {'Delta':>12}")
    lines.append("  " + "-" * 56)

    for key, info in comp["metrics"].items():
        val_a = info["a"]
        val_b = info["b"]
        label = info["label"]

        if val_a is None and val_b is None:
            continue

        # Format values
        if key == "exact_match_rate":
            sa = f"{val_a:.0%}" if val_a is not None else "—"
            sb = f"{val_b:.0%}" if val_b is not None else "—"
        elif key == "mean_latency_ms":
            sa = f"{val_a/1000:.1f}s" if val_a is not None else "—"
            sb = f"{val_b/1000:.1f}s" if val_b is not None else "—"
        elif key == "num_unique":
            sa = str(int(val_a)) if val_a is not None else "—"
            sb = str(int(val_b)) if val_b is not None else "—"
        else:
            sa = f"{val_a:.4f}" if val_a is not None else "—"
            sb = f"{val_b:.4f}" if val_b is not None else "—"

        # Delta
        if val_a is not None and val_b is not None:
            delta = val_b - val_a
            if key == "exact_match_rate":
                sd = f"{delta:+.0%}"
            elif key == "mean_latency_ms":
                sd = f"{delta/1000:+.1f}s"
            elif key == "num_unique":
                sd = f"{delta:+.0f}"
            else:
                sd = f"{delta:+.4f}"
        else:
            sd = "—"

        lines.append(f"  {label:<24} {sa:>12} {sb:>12} {sd:>12}")

    # Confidence intervals for edit distance
    ed_ci = comp.get("edit_distance_ci")
    if ed_ci:
        lines.append("")
        lines.append("  CONFIDENCE INTERVALS (95%)")
        lines.append("  " + "-" * 56)
        for label, ci in [("A edit distance", ed_ci.get("a")), ("B edit distance", ed_ci.get("b"))]:
            if ci:
                lines.append(f"  {label:<24} [{ci['lower']:.4f}, {ci['upper']:.4f}]")

    func = comp.get("functional")
    if func:
        lines.append("")
        lines.append("  FUNCTIONAL EQUIVALENCE")
        lines.append("  " + "-" * 56)
        fa = f"{func['a']:.0%}" if func["a"] is not None else "—"
        fb = f"{func['b']:.0%}" if func["b"] is not None else "—"
        ba = str(func.get("a_bugs", "—"))
        bb = str(func.get("b_bugs", "—"))
        lines.append(f"  {'Equivalence rate':<24} {fa:>12} {fb:>12}")
        lines.append(f"  {'Variants with bugs':<24} {ba:>12} {bb:>12}")
        # CIs for functional equivalence
        for side, ci_key in [("A", "a_ci"), ("B", "b_ci")]:
            ci = func.get(ci_key)
            if ci:
                lines.append(f"  {side + ' 95% CI':<24} [{ci['lower']:.0%}, {ci['upper']:.0%}]")

    lines.append("")
    return "\n".join(lines)


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
    config = runs[0].get("config", {})
    if "temperature" in config:
        metrics["temperature"] = config["temperature"]
    # Bootstrap CI for edit distance
    edit_dists = pairwise_edit_distances(outputs)
    if len(edit_dists) >= 2:
        ci = bootstrap_ci(edit_dists)
        metrics["edit_distance_ci_95"] = {
            "lower": round(ci["ci_lower"], 6),
            "upper": round(ci["ci_upper"], 6),
        }
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

    # Find the step — check top-level steps first, then inside loops
    outputs = _extract_step_outputs(runs, step_name)
    if outputs is None:
        all_names = _collect_step_names(runs[0]["steps"])
        click.echo(f"Warning: step '{step_name}' not found in [{', '.join(all_names)}].", err=True)
        return None

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


def _measure_functional_baseline(
    runs: list[dict], tasks_dir: Path,
    explicit_path: Path | None = None,
) -> dict | None:
    """Run functional equivalence tests on baseline results."""
    if explicit_path:
        test_module_path = explicit_path
    else:
        task = runs[0].get("config", {}).get("task")
        if not task:
            return None
        test_module_path = tasks_dir / task / "functional_tests.py"

    if not test_module_path.exists():
        return None

    outputs = [r["response"]["content"] for r in runs]
    click.echo(f"Running functional tests ({len(outputs)} variants)...")
    metrics = compute_functional_metrics(outputs, test_module_path)
    click.echo(
        f"  {metrics['variants_all_pass']}/{metrics['num_variants']} variants pass all tests "
        f"({metrics['functional_equivalence_rate']:.1%} equivalence rate)"
    )
    if metrics["failures"]:
        for f in metrics["failures"]:
            click.echo(f"  variant {f['variant']}: {', '.join(f['failed_tests'].keys())}")
    return metrics


def _collect_step_names(steps: list[dict]) -> list[str]:
    """Collect all step names including those inside loops."""
    names = []
    for s in steps:
        names.append(s["name"])
        if s.get("type") == "loop":
            for iteration in s.get("iterations", []):
                for inner in iteration:
                    if inner["name"] not in names:
                        names.append(inner["name"])
    return names


def _extract_step_outputs(runs: list[dict], step_name: str) -> list[str] | None:
    """Extract outputs for a named step, searching inside loops if needed."""
    # Check top-level steps
    step_names = [s["name"] for s in runs[0]["steps"]]
    if step_name in step_names:
        step_idx = step_names.index(step_name)
        return [r["steps"][step_idx]["response"]["content"] for r in runs]

    # Search inside loop steps
    for step_idx, step in enumerate(runs[0]["steps"]):
        if step.get("type") != "loop":
            continue
        # Check if any iteration contains this step name
        for iteration in step.get("iterations", []):
            inner_names = [s["name"] for s in iteration]
            if step_name in inner_names:
                inner_idx = inner_names.index(step_name)
                # Extract from last iteration of each run
                outputs = []
                for r in runs:
                    loop = r["steps"][step_idx]
                    last_iter = loop["iterations"][-1]
                    outputs.append(last_iter[inner_idx]["response"]["content"])
                return outputs

    return None


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


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Save probe results to JSON.")
@click.option("--top-logprobs", default=5, help="Number of top alternatives per token.")
def probe(config_path: str, output: str | None, top_logprobs: int):
    """Probe token-level confidence for a baseline experiment.

    Runs the experiment once with logprobs enabled and analyzes
    which tokens are deterministic vs. stochastic.
    """
    from drift.probing.logprobs import probe_with_logprobs

    config = yaml.safe_load(Path(config_path).read_text())
    config = resolve_task_prompts(config, Path.cwd() / "tasks")
    exp = config["experiment"]

    if exp.get("type") == "pipeline":
        click.echo("Error: probe only supports baseline experiments.", err=True)
        sys.exit(1)

    system = exp["system_prompt"]
    prompt = exp["user_prompt"]
    if "input" in exp:
        prompt = prompt.replace("{input}", exp["input"])

    click.echo(f"Probing: {exp['name']}")
    click.echo(f"Model: {exp['model']}")
    click.echo(f"Top logprobs: {top_logprobs}")
    click.echo()

    result = probe_with_logprobs(
        system=system,
        prompt=prompt,
        model=exp["model"],
        base_url=exp.get("base_url"),
        top_logprobs=top_logprobs,
        temperature=exp.get("temperature", 0),
    )

    stats = result.entropy_stats()
    click.echo(f"Tokens: {stats['num_tokens']}")
    click.echo(f"Mean entropy: {stats['mean_entropy']:.3f} bits")
    click.echo(f"Deterministic (<0.1): {stats['pct_deterministic']:.0%}")
    click.echo(f"High entropy (>0.5): {stats['pct_high_entropy']:.0%}")
    click.echo()

    # Show segments
    segments = result.segment_by_entropy()
    click.echo("SEGMENTS (deterministic vs stochastic):")
    click.echo("-" * 60)
    for seg in segments:
        marker = "RULE" if seg["type"] == "deterministic" else "LLM "
        text = seg["text"][:80].replace("\n", "\\n")
        if len(seg["text"]) > 80:
            text += "..."
        click.echo(f"  [{marker}] {text}")

    click.echo()

    # Show top high-entropy tokens
    high = result.high_entropy_tokens(threshold=0.5)
    if high:
        click.echo(f"TOP HIGH-ENTROPY TOKENS ({len(high)} total):")
        click.echo("-" * 60)
        for idx, token, entropy in high[:20]:
            tok_display = repr(token)
            alts = result.tokens[idx].top_alternatives[:3]
            alt_str = ", ".join(f"{a['token']!r}" for a in alts)
            click.echo(f"  [{idx:4d}] {tok_display:<20} entropy={entropy:.2f}  alts: {alt_str}")

    if output:
        probe_data = {
            "model": result.model,
            "stats": stats,
            "segments": [
                {"type": s["type"], "text": s["text"], "start": s["start"], "end": s["end"]}
                for s in segments
            ],
            "high_entropy_tokens": [
                {"index": idx, "token": tok, "entropy": ent,
                 "alternatives": [a["token"] for a in result.tokens[idx].top_alternatives[:5]]}
                for idx, tok, ent in high
            ],
            "tokens": [
                {"token": t.token, "logprob": t.logprob, "entropy": t.entropy}
                for t in result.tokens
            ],
        }
        Path(output).write_text(json.dumps(probe_data, indent=2))
        click.echo(f"\nProbe data saved to {output}")


if __name__ == "__main__":
    cli()
