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


async def _run_hybrid(config: dict, num_runs: int, output_dir: str) -> Path:
    """Run a hybrid (rules + LLM) experiment num_runs times."""
    from drift.hybrid.executor import run_hybrid, build_hybrid_prompt

    exp = config["experiment"]
    spec = exp["hybrid_spec"]
    provider = make_provider(exp)

    rules = spec["rules"]
    llm_fields = spec["llm_fields"]
    system = spec.get("hybrid_system_prompt", exp.get("system_prompt", ""))
    input_text = exp.get("input", "")
    hybrid_prompt_tpl = spec.get("hybrid_prompt_template")

    run_dir = make_run_dir(output_dir, exp["name"])

    click.echo(f"Experiment: {exp['name']}")
    click.echo(f"Type: hybrid ({len(rules)} rules + LLM for {len(llm_fields)} fields)")
    click.echo(f"Model: {exp['model']}")
    click.echo(f"Runs: {num_runs}")
    click.echo(f"Output: {run_dir}")
    click.echo()

    for i in range(num_runs):
        click.echo(f"  Run {i + 1}/{num_runs}...", nl=False)
        output_mode = spec.get("output_mode", "merge")
        hybrid_result = await run_hybrid(
            provider=provider,
            system_prompt=system,
            input_text=input_text,
            rules=rules,
            llm_fields=llm_fields,
            hybrid_prompt_template=hybrid_prompt_tpl,
            temperature=exp.get("temperature", 0),
            output_mode=output_mode,
        )
        resp = hybrid_result.response
        if output_mode == "llm_passthrough":
            content = hybrid_result.raw_llm_response
        else:
            content = json.dumps(hybrid_result.merged, indent=2)
        result = {
            "run_index": i,
            "type": "hybrid",
            "response": {
                "content": content,
                "model": resp.model if resp else exp["model"],
                "provider": resp.provider if resp else exp.get("provider", "ollama"),
                "total_ms": resp.total_ms if resp else 0,
            },
            "hybrid": {
                "rules_applied": hybrid_result.rules_applied,
                "rules_matched": hybrid_result.rules_matched,
                "extracted": hybrid_result.extracted,
                "llm_output": hybrid_result.llm_output,
                "raw_llm_response": hybrid_result.raw_llm_response,
            },
            "config": exp,
        }
        save_run(run_dir, i, result)
        ms = f" {resp.total_ms:.0f}ms" if resp else ""
        click.echo(f" rules={hybrid_result.rules_matched}/{hybrid_result.rules_applied}{ms}")

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
    elif exp_type == "hybrid":
        asyncio.run(_run_hybrid(config, runs, output_dir))
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
        if run_type == "hybrid":
            metrics["type"] = "hybrid"
            metrics["hybrid"] = _measure_hybrid(runs)

    if functional_step and run_type == "pipeline":
        func_metrics = _measure_functional(runs, functional_step, tasks_dir, functional_tests_path)
        if func_metrics:
            metrics["functional"] = func_metrics
    elif run_type in ("baseline", "hybrid"):
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
@click.option("--probe-data", default=None, type=click.Path(exists=True),
              help="Probe results JSON (from drift probe -o) for entropy analysis.")
def analyze(path: str, as_json: bool, functional_step: str | None,
            functional_tests_path: str | None, tasks_dir: str, probe_data: str | None):
    """Analyze metrics and recommend: distill, keep, or fix.

    PATH can be a metrics JSON file or a results directory.
    """
    p = Path(path)
    if p.is_dir():
        ft_path = Path(functional_tests_path) if functional_tests_path else None
        metrics = _compute_metrics(path, functional_step, ft_path, Path(tasks_dir))
    else:
        metrics = json.loads(p.read_text())

    probe = json.loads(Path(probe_data).read_text()) if probe_data else None
    report = analyze_metrics(metrics, probe_data=probe)
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


def _measure_hybrid(runs: list[dict]) -> dict:
    """Compute hybrid-specific metrics."""
    hybrid_data = [r.get("hybrid", {}) for r in runs]
    rules_applied = hybrid_data[0].get("rules_applied", 0) if hybrid_data else 0
    rules_matched = [h.get("rules_matched", 0) for h in hybrid_data]
    return {
        "rules_applied": rules_applied,
        "mean_rules_matched": round(sum(rules_matched) / len(rules_matched), 1),
        "rule_match_rate": round(sum(rules_matched) / (len(rules_matched) * rules_applied), 3) if rules_applied else 0,
    }


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
@click.option("--rules", is_flag=True, help="Run rule extraction analysis on JSON output.")
@click.option("--input", "input_path", default=None, type=click.Path(exists=True),
              help="Override input text with this file (for batch experiments).")
def probe(config_path: str, output: str | None, top_logprobs: int, rules: bool, input_path: str | None):
    """Probe token-level confidence for a baseline experiment.

    Runs the experiment once with logprobs enabled and analyzes
    which tokens are deterministic vs. stochastic.
    """
    from drift.probing.logprobs import probe_with_logprobs

    config = yaml.safe_load(Path(config_path).read_text())
    config = resolve_task_prompts(config, Path.cwd() / "tasks")
    exp = config["experiment"]

    # Override input if provided
    if input_path:
        exp["input"] = Path(input_path).read_text()

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

    if rules:
        from drift.probing.synthesizer import analyze_fields, format_rules
        input_text = exp.get("input", "")
        field_rules = analyze_fields(probe_data, input_text)
        if field_rules:
            click.echo()
            click.echo(format_rules(field_rules))

    if output:
        Path(output).write_text(json.dumps(probe_data, indent=2))
        click.echo(f"\nProbe data saved to {output}")


@cli.command()
@click.argument("probe_path", type=click.Path(exists=True))
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="Original input text file (e.g., tasks/incident_triage/input.md).")
@click.option("--probes-dir", default=None, type=click.Path(exists=True),
              help="Directory of probe JSON files for multi-probe synthesis (Tier 2 discovery).")
@click.option("--inputs-dir", default=None, type=click.Path(exists=True),
              help="Directory of input text files matching probes (required with --probes-dir).")
@click.option("--output", "-o", default=None,
              help="Write generated hybrid.yaml to this path.")
@click.option("--threshold", default=0.3, type=float,
              help="Entropy threshold: fields below this become rules (default 0.3).")
@click.option("--attention", "attention_path", default=None, type=click.Path(exists=True),
              help="Attention probe JSON (from /v1/attention) for mapping discovery hints.")
def synthesize(probe_path: str, input_path: str, probes_dir: str | None,
               inputs_dir: str | None, output: str | None, threshold: float,
               attention_path: str | None):
    """Auto-generate hybrid rules from probe data.

    Single-probe mode (Tier 1 only):
        drift synthesize probe.json --input input.md -o hybrid.yaml

    Multi-probe mode (Tier 1 + Tier 2 discovery):
        drift synthesize probe_0.json --input input_0.txt \\
            --probes-dir probes/ --inputs-dir inputs/ -o hybrid.yaml

    With attention hints (guides mapping rule discovery):
        drift synthesize probe.json --input input.md \\
            --attention attention.json --probes-dir probes/ --inputs-dir inputs/

    The first probe_path/input pair is the primary probe. When --probes-dir
    and --inputs-dir are provided, additional probes are used to discover
    Tier 2 (mapping/conditional) rules from cross-ticket patterns.

    When --attention is provided, per-token attention density analysis
    identifies which input field drives each output field, guiding Tier 2
    mapping discovery and lowering accuracy thresholds.
    """
    from drift.hybrid.autogen import (
        synthesize_rules, synthesize_rules_multi,
        render_hybrid_yaml, format_synthesis_report,
    )

    probe_data = json.loads(Path(probe_path).read_text())
    input_text = Path(input_path).read_text()

    attention_data = None
    if attention_path:
        attention_data = json.loads(Path(attention_path).read_text())
        click.echo(f"Attention hints loaded from {attention_path}")

    if probes_dir and inputs_dir:
        # Multi-probe mode: collect all (probe, input) pairs
        probes_path = Path(probes_dir)
        inputs_path = Path(inputs_dir)

        probe_files = sorted(probes_path.glob("*.json"))
        input_files = sorted(inputs_path.glob("*.txt")) or sorted(inputs_path.glob("*.md"))

        # Build mapping by stem name
        input_map = {f.stem: f for f in input_files}

        pairs = [(probe_data, input_text)]  # primary pair first
        for pf in probe_files:
            if pf.resolve() == Path(probe_path).resolve():
                continue  # skip primary, already added
            stem = pf.stem
            if stem in input_map:
                pairs.append((
                    json.loads(pf.read_text()),
                    input_map[stem].read_text(),
                ))

        click.echo(f"Multi-probe synthesis: {len(pairs)} probes")
        result = synthesize_rules_multi(pairs, entropy_threshold=threshold,
                                        attention_data=attention_data)
    else:
        result = synthesize_rules(probe_data, input_text, entropy_threshold=threshold)

    click.echo(format_synthesis_report(result))

    if output:
        yaml_content = render_hybrid_yaml(result)
        Path(output).write_text(yaml_content)
        click.echo(f"\nHybrid spec written to {output}")


@cli.command()
@click.argument("spec_path", type=click.Path(exists=True))
@click.option("--prompt", "prompt_path", required=True, type=click.Path(exists=True),
              help="System prompt file for LLM calls.")
@click.option("--output", "-o", required=True,
              help="Output path for generated Python engine.")
@click.option("--name", "engine_name", default=None,
              help="Engine name (default: derived from output filename).")
def compile(spec_path: str, prompt_path: str, output: str, engine_name: str | None):
    """Compile a hybrid spec into a standalone Python engine.

    Takes a hybrid.yaml (from drift synthesize) and a system prompt,
    and generates a self-contained Python module.

    Example:
        drift compile tasks/helpdesk_triage/hybrid_auto.yaml \\
            --prompt tasks/helpdesk_triage/triage_system.md \\
            -o engine.py
    """
    from drift.hybrid.compiler import compile_engine

    spec = yaml.safe_load(Path(spec_path).read_text())
    system_prompt = Path(prompt_path).read_text()

    rules = spec.get("rules", [])
    llm_fields = spec.get("llm_fields", [])

    if engine_name is None:
        engine_name = Path(output).stem

    code = compile_engine(
        rules=rules,
        llm_fields=llm_fields,
        system_prompt=system_prompt,
        engine_name=engine_name,
    )

    Path(output).write_text(code)

    # Count tiers
    tier1 = sum(1 for r in rules if r["type"] in ("regex", "regex_all", "literal"))
    tier2 = sum(1 for r in rules if r["type"] in ("mapping", "conditional"))

    click.echo(f"Compiled engine: {output}")
    click.echo(f"  Tier 1 (regex/literal):      {tier1} fields")
    click.echo(f"  Tier 2 (mapping/conditional): {tier2} fields")
    click.echo(f"  Tier 3 (LLM):                {len(llm_fields)} field groups")
    click.echo(f"\nUsage:")
    click.echo(f"  python {output} <input_file>              # Tier 1+2 only (no LLM)")
    click.echo(f"  from {engine_name} import process          # Import in your code")


@cli.command()
@click.argument("engine_path", type=click.Path(exists=True))
@click.option("--input-dir", required=True, type=click.Path(exists=True),
              help="Directory of input text files to validate against.")
@click.option("--expected-dir", default=None, type=click.Path(exists=True),
              help="Directory of expected JSON outputs (optional, for accuracy check).")
@click.option("--fields", default=None,
              help="Comma-separated list of dot-path fields to check (default: all).")
def validate(engine_path: str, input_dir: str, expected_dir: str | None, fields: str | None):
    """Validate a compiled engine against input files.

    Loads the engine, runs process() on each input file, and reports
    field extraction rates. Optionally compares against expected outputs.

    Example:
        drift validate engine.py --input-dir tickets/
        drift validate engine.py --input-dir tickets/ --expected-dir expected/
    """
    import importlib.util

    # Load the engine module
    spec = importlib.util.spec_from_file_location("engine", engine_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "process"):
        click.echo(f"Error: {engine_path} has no process() function", err=True)
        raise SystemExit(1)

    # Collect input files
    input_path = Path(input_dir)
    input_files = sorted(
        input_path.glob("*.txt")
    ) or sorted(
        input_path.glob("*.md")
    )

    if not input_files:
        click.echo(f"Error: no .txt or .md files in {input_dir}", err=True)
        raise SystemExit(1)

    # Load expected outputs if provided
    expected_map = {}
    if expected_dir:
        for ef in Path(expected_dir).glob("*.json"):
            expected_map[ef.stem] = json.loads(ef.read_text())

    # Parse field filter
    check_fields = [f.strip() for f in fields.split(",")] if fields else None

    # Run validation
    click.echo(f"Engine:  {engine_path}")
    click.echo(f"Inputs:  {len(input_files)} files")
    if expected_map:
        click.echo(f"Expected: {len(expected_map)} files")
    click.echo()

    field_hits = {}  # path -> count of non-None extractions
    field_total = {}  # path -> total attempts
    field_correct = {}  # path -> count matching expected
    errors = []
    total = len(input_files)

    for input_file in input_files:
        input_text = input_file.read_text()
        try:
            result = module.process(input_text)
        except Exception as e:
            errors.append((input_file.name, str(e)))
            continue

        # Flatten result to check fields
        flat = _flatten_result(result)

        for path, value in flat:
            if check_fields and path not in check_fields:
                continue
            field_total[path] = field_total.get(path, 0) + 1
            if value is not None:
                field_hits[path] = field_hits.get(path, 0) + 1

        # Compare against expected if available
        stem = input_file.stem
        if stem in expected_map:
            expected_flat = dict(_flatten_result(expected_map[stem]))
            for path, value in flat:
                if check_fields and path not in check_fields:
                    continue
                if path in expected_flat and value == expected_flat[path]:
                    field_correct[path] = field_correct.get(path, 0) + 1

    # Report
    click.echo(f"{'Field':<40} {'Extracted':>10} {'Rate':>8}", nl=True)
    click.echo(f"{'-'*40} {'-'*10} {'-'*8}")

    all_paths = sorted(field_total.keys())
    for path in all_paths:
        hits = field_hits.get(path, 0)
        tot = field_total[path]
        rate = hits / tot * 100 if tot > 0 else 0
        click.echo(f"{path:<40} {hits:>5}/{tot:<4} {rate:>6.0f}%")

    extracted_total = sum(field_hits.values())
    attempted_total = sum(field_total.values())
    overall_rate = extracted_total / attempted_total * 100 if attempted_total > 0 else 0
    click.echo(f"\nOverall extraction rate: {extracted_total}/{attempted_total} ({overall_rate:.0f}%)")

    if expected_map:
        click.echo(f"\n{'Field':<40} {'Correct':>10} {'Accuracy':>10}")
        click.echo(f"{'-'*40} {'-'*10} {'-'*10}")
        for path in all_paths:
            if path in field_correct:
                correct = field_correct[path]
                tot = min(field_total[path], len(expected_map))
                acc = correct / tot * 100 if tot > 0 else 0
                click.echo(f"{path:<40} {correct:>5}/{tot:<4} {acc:>8.0f}%")

        correct_total = sum(field_correct.values())
        expected_total = sum(
            min(field_total.get(p, 0), len(expected_map))
            for p in field_correct
        )
        overall_acc = correct_total / expected_total * 100 if expected_total > 0 else 0
        click.echo(f"\nOverall accuracy: {correct_total}/{expected_total} ({overall_acc:.0f}%)")

    if errors:
        click.echo(f"\nErrors ({len(errors)}/{total}):")
        for name, err in errors[:10]:
            click.echo(f"  {name}: {err}")

    click.echo(f"\nProcessed {total - len(errors)}/{total} inputs successfully")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--output", "-o", required=True,
              help="Output path for generated Python engine.")
@click.option("--name", "engine_name", default=None,
              help="Engine name (default: derived from output filename).")
@click.option("--threshold", default=0.3, type=float,
              help="Entropy threshold for rule generation (default 0.3).")
@click.option("--input", "input_path", default=None, type=click.Path(exists=True),
              help="Override input text with this file.")
def build(config_path: str, output: str, engine_name: str | None,
          threshold: float, input_path: str | None):
    """Probe, synthesize, and compile in one step.

    Takes a baseline experiment config and produces a standalone
    Python engine. Runs the LLM once to probe, auto-generates rules,
    and compiles everything into a single .py file.

    Example:
        drift build experiments/baseline-helpdesk.yaml -o engine.py
    """
    from drift.probing.logprobs import probe_with_logprobs
    from drift.hybrid.autogen import synthesize_rules, render_hybrid_yaml, format_synthesis_report
    from drift.hybrid.compiler import compile_engine

    # Load config
    config = yaml.safe_load(Path(config_path).read_text())
    config = resolve_task_prompts(config, Path.cwd() / "tasks")
    exp = config["experiment"]

    if exp.get("type") == "pipeline":
        click.echo("Error: build only supports baseline experiments.", err=True)
        raise SystemExit(1)

    # Override input if provided
    if input_path:
        exp["input"] = Path(input_path).read_text()

    system = exp["system_prompt"]
    prompt = exp["user_prompt"]
    input_text = exp.get("input", "")

    # Warn if input looks like placeholder text
    if input_text and _looks_like_placeholder(input_text):
        click.echo(
            "Warning: input text looks like a placeholder. Rules generated from\n"
            "placeholder text may not generalize to real data. Consider using:\n"
            f"  drift build {config_path} --input <real_input_file> -o {output}\n",
            err=True,
        )

    if input_text:
        prompt = prompt.replace("{input}", input_text)

    if engine_name is None:
        engine_name = Path(output).stem

    # Step 1: Probe
    click.echo(f"Step 1/3: Probing {exp['name']} ({exp['model']})...")
    result = probe_with_logprobs(
        system=system,
        prompt=prompt,
        model=exp["model"],
        base_url=exp.get("base_url"),
        top_logprobs=5,
        temperature=exp.get("temperature", 0),
    )
    stats = result.entropy_stats()
    click.echo(f"  {stats['num_tokens']} tokens, "
               f"{stats['pct_deterministic']:.0%} deterministic, "
               f"{stats['pct_high_entropy']:.0%} high entropy")

    probe_data = {
        "model": result.model,
        "stats": stats,
        "tokens": [
            {"token": t.token, "logprob": t.logprob, "entropy": t.entropy}
            for t in result.tokens
        ],
    }

    # Step 2: Synthesize
    click.echo(f"\nStep 2/3: Synthesizing rules (threshold={threshold})...")
    synth_result = synthesize_rules(probe_data, input_text, entropy_threshold=threshold)
    click.echo(format_synthesis_report(synth_result))

    # Step 3: Compile
    click.echo(f"\nStep 3/3: Compiling engine...")
    rules_for_compile = []
    for rule in synth_result.rules:
        if rule.rule_type in ("llm_required",):
            continue
        entry = {"path": rule.path, "type": rule.rule_type}
        if rule.pattern:
            entry["pattern"] = rule.pattern
        if rule.rule_type == "literal":
            entry["value"] = rule.value
        if rule.rule_type == "mapping":
            entry["source"] = rule.source
            entry["map"] = rule.map
            if rule.default is not None:
                entry["default"] = rule.default
        if rule.rule_type == "conditional":
            entry["conditions"] = rule.conditions
            if rule.default is not None:
                entry["default"] = rule.default
        rules_for_compile.append(entry)

    code = compile_engine(
        rules=rules_for_compile,
        llm_fields=synth_result.llm_fields,
        system_prompt=system,
        engine_name=engine_name,
    )
    Path(output).write_text(code)

    # Summary
    tier1 = sum(1 for r in rules_for_compile if r["type"] in ("regex", "regex_all", "literal"))
    tier2 = sum(1 for r in rules_for_compile if r["type"] in ("mapping", "conditional"))

    click.echo(f"\nEngine written to {output}")
    click.echo(f"  Tier 1 (regex/literal):      {tier1} fields")
    click.echo(f"  Tier 2 (mapping/conditional): {tier2} fields")
    click.echo(f"  Tier 3 (LLM):                {len(synth_result.llm_fields)} field groups")
    click.echo(f"\nUsage:")
    click.echo(f"  python {output} <input_file>")


def _looks_like_placeholder(text: str) -> bool:
    """Detect if input text is likely placeholder/template content."""
    lower = text.lower()
    # Strong signals — any one is enough
    if lower.count("placeholder") >= 2:
        return True
    if "lorem ipsum" in lower:
        return True
    # Weak signals — need 2+
    signals = [
        "placeholder" in lower,
        "example.com" in lower,
        "xxx" in lower or "todo" in lower,
        lower.count("{") >= 3,  # template variables
        len(text.strip()) < 50,
    ]
    return sum(signals) >= 2


def _flatten_result(obj, prefix=""):
    """Flatten a nested dict to (dotted_path, value) pairs."""
    results = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            results.extend(_flatten_result(val, new_prefix))
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            results.append((f"{prefix}[{i}]", val))
    else:
        results.append((prefix, obj))
    return results


@cli.command()
@click.argument("model")
@click.option("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
@click.option("--port", default=8080, type=int, help="Bind port (default: 8080).")
@click.option("--log", "log_file", default=None, type=click.Path(), help="Log file for request logging.")
def serve(model: str, host: str, port: int, log_file: str | None):
    """Start the drift MLX inference server.

    Run on an Apple Silicon Mac to expose MLX inference over HTTP,
    including full-vocab logprobs and attention weight probing.

    Example:
        drift serve mlx-community/Qwen2.5-32B-Instruct-4bit --port 8080
    """
    click.echo(f"Drift MLX server")
    click.echo(f"  Model: {model}")
    click.echo(f"  Host:  {host}")
    click.echo(f"  Port:  {port}")
    click.echo()
    click.echo(f"Loading model (this may take a while)...")
    from drift.server.mlx_server import start_server
    start_server(model=model, host=host, port=port, log_file=log_file)


if __name__ == "__main__":
    cli()
