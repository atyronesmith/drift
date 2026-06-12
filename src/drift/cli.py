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


def _guard_check(guard, content: str, label: str) -> dict | None:
    """Run a safety guard check, print result, return metadata."""
    if not guard or not content:
        return None
    try:
        result = guard.check(content)
        verdict = "SAFE" if result.safe else f"UNSAFE ({result.category})"
        tiered = " [tiered]" if "tiered:" in result.raw_response else ""
        click.echo(f"      guard: {verdict} ({result.latency_ms:.0f}ms){tiered}")
        return result.to_dict()
    except Exception as e:
        click.echo(f"      guard: ERROR ({e})")
        return {"safe": None, "category": None, "latency_ms": None, "error": str(e)}


def _guard_summary(all_checks: list[dict], guard=None):
    """Print guard summary statistics at end of experiment."""
    if not all_checks:
        return
    flat = []
    for c in all_checks:
        if "step_checks" in c:
            flat.extend(c["step_checks"])
        else:
            flat.append(c)

    total = len(flat)
    safe = sum(1 for c in flat if c.get("safe") is True)
    unsafe = sum(1 for c in flat if c.get("safe") is False)
    errors = total - safe - unsafe

    click.echo(f"\nGuard summary: {safe}/{total} safe, {unsafe} rejected"
               + (f", {errors} errors" if errors else ""))
    if unsafe:
        rej = unsafe / (total - errors) if (total - errors) else 0
        click.echo(f"  Rejection rate: {rej:.1%}")
        click.echo(f"  Expected retries per useful output: {1/(1-rej):.2f}x" if rej < 1 else "")

        from collections import Counter
        cats = Counter(c.get("category") for c in flat if c.get("safe") is False)
        for cat, count in cats.most_common():
            click.echo(f"  {cat}: {count}")

    # Print tiered guard stats if applicable
    if guard and hasattr(guard, "summary"):
        stats = guard.summary()
        if stats.get("total", 0) > 0:
            click.echo(f"\n  Tiered guard stats:")
            click.echo(f"    Tier 1 (keyword):    {stats['tier1']}")
            click.echo(f"    Tier 2 (classifier): {stats['tier2']}")
            click.echo(f"    Tier 3 (LLM guard):  {stats['escalated']}")
            click.echo(f"    Classifier handled:  {stats['classifier_handled_pct']}%")
            click.echo(f"    Escalation rate:     {stats['escalation_pct']}%")


async def _run_baseline(config: dict, num_runs: int, output_dir: str, guard=None) -> Path:
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

    guard_checks = []
    for i in range(num_runs):
        click.echo(f"  Run {i + 1}/{num_runs}...", nl=False)
        response = await provider.complete(system, prompt, temperature=exp.get("temperature", 0))
        click.echo(f" {response.total_ms:.0f}ms")
        result = {
            "run_index": i,
            "type": "baseline",
            "response": response.to_dict(),
            "config": exp,
        }
        gc = _guard_check(guard, response.content, f"run {i}")
        if gc:
            result["guard"] = gc
            guard_checks.append(gc)
        save_run(run_dir, i, result)

    _guard_summary(guard_checks, guard)
    click.echo(f"\nDone. Results saved to {run_dir}")
    return run_dir


async def _run_pipeline(config: dict, num_runs: int, output_dir: str, guard=None) -> Path:
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

    guard_checks = []
    for i in range(num_runs):
        click.echo(f"  Run {i + 1}/{num_runs}:")
        task_dir = (Path.cwd() / "tasks" / exp["task"]) if "task" in exp else None
        step_results = await run_pipeline(steps, initial_input, task_dir=task_dir)
        serialized = _serialize_results(step_results)

        # Run guard checks on each step output
        run_guard = {"step_checks": []}
        if guard:
            for step_data in serialized:
                if step_data.get("type") == "loop":
                    for iteration in step_data.get("iterations", []):
                        for s in iteration:
                            gc = _guard_check(guard, s.get("response", {}).get("content", ""),
                                              f"{s['name']}")
                            if gc:
                                gc["step"] = s["name"]
                                run_guard["step_checks"].append(gc)
                else:
                    gc = _guard_check(guard, step_data.get("response", {}).get("content", ""),
                                      f"{step_data['name']}")
                    if gc:
                        gc["step"] = step_data["name"]
                        run_guard["step_checks"].append(gc)

            run_guard["all_safe"] = all(c.get("safe") for c in run_guard["step_checks"])
            guard_checks.append(run_guard)

        result = {
            "run_index": i,
            "type": "pipeline",
            "steps": serialized,
            "config": exp,
        }
        if run_guard["step_checks"]:
            result["guard"] = run_guard
        save_run(run_dir, i, result)

    _guard_summary(guard_checks, guard)
    click.echo(f"\nDone. Results saved to {run_dir}")
    return run_dir


async def _run_hybrid(config: dict, num_runs: int, output_dir: str, guard=None) -> Path:
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

    guard_checks = []
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
        ms = f" {resp.total_ms:.0f}ms" if resp else ""
        click.echo(f" rules={hybrid_result.rules_matched}/{hybrid_result.rules_applied}{ms}")
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
        gc = _guard_check(guard, content, f"run {i}")
        if gc:
            result["guard"] = gc
            guard_checks.append(gc)
        save_run(run_dir, i, result)

    _guard_summary(guard_checks, guard)
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
@click.option("--safety-guard", default=None,
              help="Guard model for safety checking (e.g. llama-guard3:1b).")
@click.option("--guard-base-url", default=None,
              help="Ollama base URL for guard model.")
@click.option("--exclude-categories", default=None,
              help="Comma-separated guard categories to ignore (e.g. S6,S8).")
@click.option("--tiered-guard", default=None, type=click.Path(exists=True),
              help="Path to guard classifier weights JSON (enables tiered guard).")
@click.option("--guard-keywords", default=None, type=click.Path(exists=True),
              help="Path to guard Tier 1 keywords JSON (optional, for --tiered-guard).")
def run(config_path: str, runs: int, output_dir: str,
        safety_guard: str | None, guard_base_url: str | None,
        exclude_categories: str | None,
        tiered_guard: str | None, guard_keywords: str | None):
    """Run an experiment from a YAML config."""
    config = yaml.safe_load(Path(config_path).read_text())
    config = resolve_task_prompts(config, Path.cwd() / "tasks")

    guard = None
    if safety_guard:
        from drift.safety.guard import GuardModel
        excluded = [c.strip() for c in exclude_categories.split(",")] if exclude_categories else []
        base_guard = GuardModel(model=safety_guard, base_url=guard_base_url,
                                exclude_categories=excluded)
        if tiered_guard:
            from drift.safety.guard import TieredGuard
            guard = TieredGuard(base_guard, tiered_guard, guard_keywords)
            click.echo(f"Safety guard: {safety_guard} (tiered, weights={tiered_guard})"
                       + (f" (excluding {','.join(excluded)})" if excluded else ""))
        else:
            guard = base_guard
            click.echo(f"Safety guard: {safety_guard}"
                       + (f" (excluding {','.join(excluded)})" if excluded else ""))

    exp_type = config["experiment"].get("type", "baseline")
    if exp_type == "pipeline":
        asyncio.run(_run_pipeline(config, runs, output_dir, guard=guard))
    elif exp_type == "hybrid":
        asyncio.run(_run_hybrid(config, runs, output_dir, guard=guard))
    else:
        asyncio.run(_run_baseline(config, runs, output_dir, guard=guard))


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


@cli.command(name="probe-hidden")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Save hidden-state result to JSON.")
@click.option("--layers", default="all",
              help='Layers to capture: "all" or comma-separated indices (e.g. "20,38").')
@click.option("--positions", default="last_input",
              help='Positions: "last_input", "all_input", or comma-separated indices.')
@click.option("--server-url", default=None,
              help="Drift MLX server URL (default: DRIFT_MLX_URL or http://localhost:8080).")
@click.option("--local", "use_local", is_flag=True,
              help="Load the MLX model in-process instead of using the server.")
@click.option("--model", "model_override", default=None,
              help="MLX model ID for --local mode (default: from config or Qwen2.5-32B-4bit).")
@click.option("--input", "input_path", default=None, type=click.Path(exists=True),
              help="Override input text with this file (for batch experiments).")
def probe_hidden(config_path: str, output: str | None, layers: str, positions: str,
                 server_url: str | None, use_local: bool, model_override: str | None,
                 input_path: str | None):
    """Capture hidden states (residual stream) for a baseline experiment.

    Prefill-only: one forward pass over the input, no generation. The
    captured vectors feed Tier 4 linear-probe training and runtime.
    """
    config = yaml.safe_load(Path(config_path).read_text())
    config = resolve_task_prompts(config, Path.cwd() / "tasks")
    exp = config["experiment"]

    if input_path:
        exp["input"] = Path(input_path).read_text()

    if exp.get("type") == "pipeline":
        click.echo("Error: probe-hidden only supports baseline experiments.", err=True)
        sys.exit(1)

    system = exp["system_prompt"]
    prompt = exp["user_prompt"]
    if "input" in exp:
        prompt = prompt.replace("{input}", exp["input"])

    layer_spec = layers if layers == "all" else [int(x) for x in layers.split(",")]
    pos_spec = positions if positions in ("last_input", "all_input") \
        else [int(x) for x in positions.split(",")]

    click.echo(f"Probing hidden states: {exp['name']}")
    click.echo(f"Layers: {layers}  Positions: {positions}")

    import time as _time
    t0 = _time.perf_counter()
    if use_local:
        from drift.probing.mlx_hidden import probe_hidden_states
        model = model_override or "mlx-community/Qwen2.5-32B-Instruct-4bit"
        result = probe_hidden_states(
            system=system, prompt=prompt, model=model,
            layers=layer_spec, positions=pos_spec, prefill_only=True,
        )
    else:
        from drift.probing.mlx_hidden import probe_hidden_remote
        result = probe_hidden_remote(
            system=system, prompt=prompt, base_url=server_url,
            layers=layer_spec, positions=pos_spec, prefill_only=True,
        )
    elapsed_ms = (_time.perf_counter() - t0) * 1000

    n_vecs = sum(len(by_pos) for by_pos in result.states.values())
    dim = len(next(iter(next(iter(result.states.values())).values()))) if n_vecs else 0
    click.echo(f"Model: {result.model}")
    click.echo(f"Input tokens: {result.n_input}")
    click.echo(f"Captured: {len(result.layers)} layers x {len(result.positions)} positions "
               f"({n_vecs} vectors, d_model={dim})")
    click.echo(f"Prefill time: {elapsed_ms:,.0f} ms")

    if output:
        Path(output).write_text(json.dumps(result.to_dict(), indent=2))
        click.echo(f"\nHidden-state data saved to {output}")


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


def _run_validation(engine_path: str, input_dir: str, check_fields: list[str] | None = None) -> dict:
    """Run validation of a compiled engine against input files.

    Returns:
        {
            "field_hits": {path: count},
            "field_total": {path: count},
            "errors": [(filename, error_msg)],
            "per_file": {filename: {path: value}},
            "failing_files": [filename, ...],
            "overall_rate": float,
        }
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("engine", engine_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "process"):
        raise ValueError(f"{engine_path} has no process() function")

    input_path = Path(input_dir)
    input_files = sorted(input_path.glob("*.txt")) or sorted(input_path.glob("*.md"))
    if not input_files:
        raise ValueError(f"no .txt or .md files in {input_dir}")

    field_hits = {}
    field_total = {}
    errors = []
    per_file = {}
    failing_files = []

    for input_file in input_files:
        input_text = input_file.read_text()
        try:
            result = module.process(input_text)
        except Exception as e:
            errors.append((input_file.name, str(e)))
            failing_files.append(input_file.name)
            continue

        flat = _flatten_result(result)
        file_results = {}

        for path, value in flat:
            if check_fields and path not in check_fields:
                continue
            field_total[path] = field_total.get(path, 0) + 1
            file_results[path] = value
            if value is not None:
                field_hits[path] = field_hits.get(path, 0) + 1

        per_file[input_file.name] = file_results

    # Count missing fields: any field seen in some files but absent in others
    # is a failure (the engine should extract it from every input)
    all_fields = set(field_total.keys())
    n_processed = len(input_files) - len(errors)
    for field in all_fields:
        if check_fields and field not in check_fields:
            continue
        if field_total[field] < n_processed:
            # Some files didn't produce this field at all
            field_total[field] = n_processed

    # Identify failing files: any file missing a field that others have
    for filename, file_results in per_file.items():
        missing = all_fields - set(file_results.keys())
        if missing:
            failing_files.append(filename)

    extracted_total = sum(field_hits.values())
    attempted_total = sum(field_total.values())
    overall_rate = extracted_total / attempted_total * 100 if attempted_total > 0 else 0

    return {
        "field_hits": field_hits,
        "field_total": field_total,
        "errors": errors,
        "per_file": per_file,
        "failing_files": failing_files,
        "overall_rate": overall_rate,
        "n_files": len(input_files),
    }


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
    check_fields = [f.strip() for f in fields.split(",")] if fields else None

    try:
        vr = _run_validation(engine_path, input_dir, check_fields)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    # Load expected outputs if provided
    expected_map = {}
    if expected_dir:
        for ef in Path(expected_dir).glob("*.json"):
            expected_map[ef.stem] = json.loads(ef.read_text())

    click.echo(f"Engine:  {engine_path}")
    click.echo(f"Inputs:  {vr['n_files']} files")
    if expected_map:
        click.echo(f"Expected: {len(expected_map)} files")
    click.echo()

    # Report extraction rates
    click.echo(f"{'Field':<40} {'Extracted':>10} {'Rate':>8}", nl=True)
    click.echo(f"{'-'*40} {'-'*10} {'-'*8}")

    all_paths = sorted(vr["field_total"].keys())
    for path in all_paths:
        hits = vr["field_hits"].get(path, 0)
        tot = vr["field_total"][path]
        rate = hits / tot * 100 if tot > 0 else 0
        click.echo(f"{path:<40} {hits:>5}/{tot:<4} {rate:>6.0f}%")

    click.echo(f"\nOverall extraction rate: "
               f"{sum(vr['field_hits'].values())}/{sum(vr['field_total'].values())} "
               f"({vr['overall_rate']:.0f}%)")

    if expected_map:
        # Accuracy comparison
        field_correct = {}
        for filename, file_results in vr["per_file"].items():
            stem = Path(filename).stem
            if stem in expected_map:
                expected_flat = dict(_flatten_result(expected_map[stem]))
                for path, value in file_results.items():
                    if check_fields and path not in check_fields:
                        continue
                    if path in expected_flat and value == expected_flat[path]:
                        field_correct[path] = field_correct.get(path, 0) + 1

        click.echo(f"\n{'Field':<40} {'Correct':>10} {'Accuracy':>10}")
        click.echo(f"{'-'*40} {'-'*10} {'-'*10}")
        for path in all_paths:
            if path in field_correct:
                correct = field_correct[path]
                tot = min(vr["field_total"][path], len(expected_map))
                acc = correct / tot * 100 if tot > 0 else 0
                click.echo(f"{path:<40} {correct:>5}/{tot:<4} {acc:>8.0f}%")

        correct_total = sum(field_correct.values())
        expected_total = sum(
            min(vr["field_total"].get(p, 0), len(expected_map))
            for p in field_correct
        )
        overall_acc = correct_total / expected_total * 100 if expected_total > 0 else 0
        click.echo(f"\nOverall accuracy: {correct_total}/{expected_total} ({overall_acc:.0f}%)")

    if vr["errors"]:
        click.echo(f"\nErrors ({len(vr['errors'])}/{vr['n_files']}):")
        for name, err in vr["errors"][:10]:
            click.echo(f"  {name}: {err}")

    click.echo(f"\nProcessed {vr['n_files'] - len(vr['errors'])}/{vr['n_files']} inputs successfully")


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
        if rule.rule_type == "classifier":
            entry["vocabulary"] = rule.vocabulary
            entry["weights"] = rule.weights
            entry["bias"] = rule.bias
            entry["classes"] = rule.classes
            entry["confidence_threshold"] = rule.confidence_threshold
            if rule.structured_features:
                entry["structured_features"] = rule.structured_features
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
    tier3 = sum(1 for r in rules_for_compile if r["type"] == "classifier")

    click.echo(f"\nEngine written to {output}")
    click.echo(f"  Tier 1 (regex/literal):      {tier1} fields")
    click.echo(f"  Tier 2 (mapping/conditional): {tier2} fields")
    click.echo(f"  Tier 3 (classifier):         {tier3} fields")
    click.echo(f"  Tier 4 (LLM):                {len(synth_result.llm_fields)} field groups")
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
@click.argument("engine_path", type=click.Path(exists=True))
@click.option("--input-dir", required=True, type=click.Path(exists=True),
              help="Directory of input text files to validate against.")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True),
              help="Experiment config YAML (for model/prompt settings).")
@click.option("--max-iter", default=3, type=int,
              help="Maximum refinement iterations (default: 3).")
@click.option("--target-rate", default=100, type=int,
              help="Target extraction rate %% (default: 100).")
@click.option("--probe-sample", default=5, type=int,
              help="Max failing inputs to probe per iteration (default: 5).")
@click.option("--threshold", default=0.3, type=float,
              help="Entropy threshold for rule generation (default: 0.3).")
def refine(engine_path: str, input_dir: str, config_path: str,
           max_iter: int, target_rate: int, probe_sample: int, threshold: float):
    """Iterative refinement: validate, re-probe failures, recompile.

    Takes a compiled engine and iteratively improves it by:
    1. Validating against input files to find failures
    2. Probing failing inputs to get fresh logprob data
    3. Re-synthesizing rules from broader probe data
    4. Recompiling the engine

    Repeats until the target extraction rate is met or max iterations reached.

    Example:
        drift refine engine.py --input-dir tickets/ \\
            --config experiments/baseline-helpdesk.yaml --max-iter 3
    """
    from drift.probing.logprobs import probe_with_logprobs
    from drift.hybrid.autogen import synthesize_rules_multi, format_synthesis_report
    from drift.hybrid.compiler import compile_engine

    import random

    # Load experiment config
    config = yaml.safe_load(Path(config_path).read_text())
    config = resolve_task_prompts(config, Path.cwd() / "tasks")
    exp = config["experiment"]

    system = exp["system_prompt"]
    prompt_template = exp["user_prompt"]
    model = exp["model"]
    base_url = exp.get("base_url")
    temperature = exp.get("temperature", 0)
    engine_name = Path(engine_path).stem

    input_path = Path(input_dir)
    all_probes = []  # (probe_data, input_text) pairs accumulated across iterations
    history = []  # extraction rate per iteration

    click.echo(f"Iterative refinement: {engine_path}")
    click.echo(f"  Model:       {model}")
    click.echo(f"  Inputs:      {input_dir}")
    click.echo(f"  Max iter:    {max_iter}")
    click.echo(f"  Target rate: {target_rate}%")
    click.echo()

    for iteration in range(max_iter):
        click.echo(f"{'=' * 60}")
        click.echo(f"  Iteration {iteration + 1}/{max_iter}")
        click.echo(f"{'=' * 60}")

        # Step 1: Validate
        click.echo(f"\n  Validating...")
        try:
            vr = _run_validation(engine_path, input_dir)
        except ValueError as e:
            click.echo(f"  Error: {e}", err=True)
            raise SystemExit(1)

        rate = vr["overall_rate"]
        history.append(rate)

        # Show per-field rates
        for path in sorted(vr["field_total"].keys()):
            hits = vr["field_hits"].get(path, 0)
            tot = vr["field_total"][path]
            field_rate = hits / tot * 100 if tot > 0 else 0
            marker = "" if field_rate == 100 else " <--"
            click.echo(f"    {path:<36} {hits:>4}/{tot:<4} {field_rate:>5.0f}%{marker}")

        click.echo(f"\n  Overall: {rate:.0f}%  |  "
                   f"Failures: {len(vr['failing_files'])}  |  "
                   f"Errors: {len(vr['errors'])}")

        # Step 2: Check stopping conditions
        if rate >= target_rate:
            click.echo(f"\n  Target rate {target_rate}% reached.")
            break

        if not vr["failing_files"]:
            click.echo(f"\n  No failing files (errors only). Cannot refine further.")
            break

        # Step 3: Probe failing inputs
        sample = vr["failing_files"]
        if len(sample) > probe_sample:
            sample = random.sample(sample, probe_sample)

        click.echo(f"\n  Probing {len(sample)} failing inputs...")
        new_probes = 0
        for filename in sample:
            filepath = input_path / filename
            if not filepath.exists():
                continue

            input_text = filepath.read_text()
            prompt = prompt_template.replace("{input}", input_text)

            click.echo(f"    {filename}...", nl=False)
            try:
                result = probe_with_logprobs(
                    system=system, prompt=prompt, model=model,
                    base_url=base_url, top_logprobs=5, temperature=temperature,
                )
                probe_data = {
                    "tokens": [
                        {"token": t.token, "logprob": t.logprob, "entropy": t.entropy}
                        for t in result.tokens
                    ],
                }
                all_probes.append((probe_data, input_text))
                new_probes += 1
                click.echo(f" {len(result.tokens)} tokens")
            except Exception as e:
                click.echo(f" error: {e}")

        if new_probes == 0:
            click.echo(f"\n  No probes succeeded. Cannot refine further.")
            break

        click.echo(f"\n  Total probes: {len(all_probes)}")

        # Step 4: Synthesize rules from all probes
        click.echo(f"\n  Synthesizing rules (threshold={threshold})...")
        synth_result = synthesize_rules_multi(all_probes, entropy_threshold=threshold)

        total_rules = len([r for r in synth_result.rules if r.rule_type != "llm_required"])
        click.echo(f"    {total_rules} rules, {len(synth_result.llm_fields)} LLM field groups")

        # Step 5: Compile updated engine
        rules_for_compile = []
        for rule in synth_result.rules:
            if rule.rule_type == "llm_required":
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
        Path(engine_path).write_text(code)
        click.echo(f"\n  Engine recompiled: {engine_path}")

    # Final summary
    click.echo(f"\n{'=' * 60}")
    click.echo(f"  Refinement complete")
    click.echo(f"{'=' * 60}")
    for i, rate in enumerate(history):
        marker = " <-- final" if i == len(history) - 1 else ""
        click.echo(f"    Iteration {i + 1}: {rate:.0f}%{marker}")

    if len(history) > 1:
        improvement = history[-1] - history[0]
        click.echo(f"\n  Improvement: {improvement:+.0f}% ({history[0]:.0f}% -> {history[-1]:.0f}%)")
    click.echo(f"  Probes used: {len(all_probes)}")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--input-dir", required=True, type=click.Path(exists=True),
              help="Directory of input text files to label.")
@click.option("--output", "-o", required=True,
              help="Output path for hybrid YAML spec.")
@click.option("--n-samples", default=200, type=int,
              help="Max input files to label (default: 200).")
@click.option("--accuracy-threshold", default=0.70, type=float,
              help="Min CV accuracy to promote a classifier (default: 0.70).")
@click.option("--confidence-threshold", default=0.95, type=float,
              help="Min softmax confidence for classifier predictions (default: 0.95).")
@click.option("--labels", "labels_path", default=None, type=click.Path(),
              help="Pre-computed labels file (skip LLM calls).")
def distill(config_path: str, input_dir: str, output: str, n_samples: int,
            accuracy_threshold: float, confidence_threshold: float,
            labels_path: str | None):
    """Label inputs via LLM, train classifiers, output hybrid YAML.

    Reads input files, labels them through the LLM, runs Tier 1/2
    synthesis on the first input, then trains classifiers for remaining
    classification fields. Outputs a hybrid YAML spec with all tiers.

    Example:
        drift distill experiments/baseline-helpdesk.yaml \\
            --input-dir datasets/ticket_inputs/ \\
            --n-samples 200 -o hybrid_with_classifiers.yaml
    """
    from drift.probing.logprobs import probe_with_logprobs
    from drift.hybrid.autogen import (
        synthesize_rules_multi, render_hybrid_yaml, format_synthesis_report,
    )

    # Load config
    config = yaml.safe_load(Path(config_path).read_text())
    config = resolve_task_prompts(config, Path.cwd() / "tasks")
    exp = config["experiment"]

    system = exp["system_prompt"]
    user_prompt_tpl = exp["user_prompt"]
    model = exp["model"]
    base_url = exp.get("base_url")

    # Collect input files
    input_path = Path(input_dir)
    input_files = sorted(input_path.glob("*.txt")) + sorted(input_path.glob("*.md"))
    if not input_files:
        click.echo(f"Error: no .txt or .md files found in {input_dir}", err=True)
        raise SystemExit(1)

    input_files = input_files[:n_samples]
    click.echo(f"Distill: {exp['name']}")
    click.echo(f"  Model:   {model}")
    click.echo(f"  Inputs:  {len(input_files)} files from {input_dir}")
    click.echo(f"  Accuracy threshold: {accuracy_threshold}")
    click.echo(f"  Confidence threshold: {confidence_threshold}")
    click.echo()

    # Step 1: Probe all inputs to get LLM outputs
    click.echo(f"Step 1/3: Probing {len(input_files)} inputs...")
    probes = []
    for i, f in enumerate(input_files):
        input_text = f.read_text()
        prompt = user_prompt_tpl.replace("{input}", input_text)

        click.echo(f"  [{i+1:3d}/{len(input_files)}] {f.name[:30]}... ", nl=False)
        try:
            result = probe_with_logprobs(
                system=system, prompt=prompt, model=model,
                base_url=base_url, top_logprobs=5,
                temperature=exp.get("temperature", 0),
            )
            probe_data = {
                "model": result.model,
                "tokens": [
                    {"token": t.token, "logprob": t.logprob, "entropy": t.entropy}
                    for t in result.tokens
                ],
            }
            probes.append((probe_data, input_text))
            n_tokens = len(result.tokens)
            click.echo(f"OK ({n_tokens} tokens)")
        except Exception as e:
            click.echo(f"SKIP ({e})")

    click.echo(f"\n  Probed {len(probes)}/{len(input_files)} inputs successfully")
    click.echo()

    if len(probes) < 3:
        click.echo("Error: need at least 3 successful probes.", err=True)
        raise SystemExit(1)

    # Step 2: Synthesize rules (Tier 1 + 2 + 3 classifiers)
    click.echo(f"Step 2/3: Synthesizing rules...")
    synth_result = synthesize_rules_multi(
        probes, entropy_threshold=0.3,
        min_accuracy=accuracy_threshold,
        confidence_threshold=confidence_threshold,
    )
    click.echo(format_synthesis_report(synth_result))
    click.echo()

    # Step 3: Output hybrid YAML
    click.echo(f"Step 3/3: Writing hybrid spec...")
    yaml_str = render_hybrid_yaml(synth_result)
    Path(output).write_text(yaml_str)

    tier1 = sum(1 for r in synth_result.rules if r.rule_type in ("regex", "regex_all", "literal"))
    tier2 = sum(1 for r in synth_result.rules if r.rule_type in ("mapping", "conditional"))
    tier3 = sum(1 for r in synth_result.rules if r.rule_type == "classifier")
    llm = len(synth_result.llm_fields)

    click.echo(f"\nHybrid spec written to {output}")
    click.echo(f"  Tier 1 (regex/literal):      {tier1} fields")
    click.echo(f"  Tier 2 (mapping/conditional): {tier2} fields")
    click.echo(f"  Tier 3 (classifier):         {tier3} fields")
    click.echo(f"  Tier 4 (LLM):                {llm} field groups")
    click.echo(f"\nTo compile to engine:")
    click.echo(f"  drift compile {output} --prompt <system_prompt.md> -o engine.py")


@cli.command(name="probe-pipeline")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--model", default="qwen2.5:32b", help="Ollama model for probing (default: qwen2.5:32b).")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output directory for per-step probe JSON files.")
@click.option("--top-logprobs", default=5, help="Number of top alternatives per token.")
@click.option("--rules", is_flag=True, help="Run rule extraction analysis on JSON steps.")
def probe_pipeline(config_path: str, model: str, output: str, top_logprobs: int, rules: bool):
    """Probe token-level confidence for each pipeline step.

    Runs each step sequentially with logprobs via Ollama, chaining
    real outputs so each step sees realistic input from the prior step.
    Saves per-step probe JSON (step_0_reviewer.json, etc.) for
    downstream rule synthesis.

    Example:
        drift probe-pipeline experiments/pipeline-structured.yaml \\
            --model qwen2.5:32b -o results/pipeline-probes/
    """
    from drift.probing.logprobs import probe_with_logprobs
    from drift.pipeline.pipeline import _render_prompt

    config = yaml.safe_load(Path(config_path).read_text())
    config = resolve_task_prompts(config, Path.cwd() / "tasks")
    exp = config["experiment"]

    if exp.get("type") != "pipeline":
        click.echo("Error: probe-pipeline requires a pipeline experiment.", err=True)
        sys.exit(1)

    steps = exp["steps"]
    initial_input = exp["input"]
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Pipeline probe: {exp['name']}")
    click.echo(f"  Steps: {len(steps)}")
    click.echo(f"  Model: {model} (Ollama)")
    click.echo(f"  Output: {out_dir}")
    click.echo()

    context = {"input": initial_input}

    for i, step in enumerate(steps):
        if step.get("type") == "loop":
            click.echo(f"  Step {i}/{len(steps)} ({step.get('name', 'loop')}): "
                        f"skipping loop step (not supported for probing)")
            continue

        name = step.get("name", f"step_{i}")
        system = step["system_prompt"]

        # Determine user prompt template (same logic as pipeline runner)
        template = step.get("user_prompt", "{previous_output}")
        if i == 0 and "user_prompt" not in step:
            template = "{input}"
        prompt = _render_prompt(template, context)

        click.echo(f"  Step {i}/{len(steps)} ({name})...", nl=False)

        result = probe_with_logprobs(
            system=system,
            prompt=prompt,
            model=model,
            top_logprobs=top_logprobs,
            temperature=step.get("temperature", 0),
        )

        stats = result.entropy_stats()
        click.echo(f" {stats['num_tokens']} tokens, "
                    f"{stats['pct_deterministic']:.0%} deterministic, "
                    f"{stats['pct_high_entropy']:.0%} high entropy")

        # Build probe data
        segments = result.segment_by_entropy()
        high = result.high_entropy_tokens(threshold=0.5)

        probe_data = {
            "step_index": i,
            "step_name": name,
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

        # Save step probe
        probe_file = out_dir / f"step_{i}_{name}.json"
        probe_file.write_text(json.dumps(probe_data, indent=2))

        # Rule extraction analysis
        if rules:
            from drift.probing.synthesizer import analyze_fields, format_rules
            input_text = initial_input if i == 0 else context.get("previous_output", "")
            field_rules = analyze_fields(probe_data, input_text)
            if field_rules:
                click.echo(f"    {format_rules(field_rules)}")

        # Chain output to next step's context
        content = result.content
        context["previous_output"] = content
        context[f"step_{i}"] = content
        context[f"step_{name}"] = content

    click.echo(f"\nProbe data saved to {out_dir}/")


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


@cli.command("safety-overhead")
@click.option("--models-dir", default="security_models",
              help="Directory containing security model YAML files.")
@click.option("--model", "-m", "model_names", multiple=True,
              help="Specific model file(s) to evaluate (default: all).")
@click.option("--steps", default=3, help="Number of pipeline steps.")
@click.option("--code-steps", default=1, help="Steps that produce code.")
@click.option("--tool-calls", default=0.0, type=float,
              help="Average tool/MCP calls per step.")
@click.option("--iterations", default=1, help="Loop iterations.")
@click.option("--pipeline", "pipeline_path", default=None,
              type=click.Path(exists=True),
              help="Infer pipeline shape from experiment YAML.")
def safety_overhead(models_dir, model_names, steps, code_steps,
                    tool_calls, iterations, pipeline_path):
    """Compare safety overhead across security models.

    Loads security model definitions and projects guard check counts,
    latency, and cost for a given pipeline shape.

    Examples:

        drift safety-overhead --steps 3 --code-steps 1

        drift safety-overhead --pipeline experiments/pipeline-structured.yaml

        drift safety-overhead -m microsoft_foundry.yaml -m finos_regulated.yaml
    """
    from drift.safety.models import (
        SecurityModel, PipelineShape, project_overhead, load_all_models,
    )

    models_path = Path(models_dir)
    if not models_path.is_absolute():
        models_path = Path.cwd() / models_path

    # Load models
    if model_names:
        models = []
        for name in model_names:
            p = models_path / name
            if not p.exists():
                p = models_path / f"{name}.yaml"
            if not p.exists():
                click.echo(f"Model not found: {name}", err=True)
                continue
            models.append(SecurityModel.from_yaml(p))
    else:
        models = load_all_models(models_path)

    if not models:
        click.echo("No security models found.", err=True)
        return

    # Infer pipeline shape from config if provided
    if pipeline_path:
        config = yaml.safe_load(Path(pipeline_path).read_text())
        exp = config.get("experiment", {})
        exp_steps = exp.get("steps", [])
        steps = len(exp_steps)
        code_steps = 0
        for s in exp_steps:
            name = s.get("name", "").lower()
            if any(kw in name for kw in ("implement", "code", "refactor", "fix")):
                code_steps += 1
        # Check for loop steps
        for s in exp_steps:
            if s.get("type") == "loop":
                iterations = s.get("max_iterations", 5)
                inner = s.get("steps", [])
                steps = len(inner)
                code_steps = sum(1 for s in inner
                                 if any(kw in s.get("name", "").lower()
                                        for kw in ("implement", "code", "refactor", "fix")))
                break
        click.echo(f"Pipeline: {Path(pipeline_path).stem}")
    else:
        click.echo(f"Pipeline shape: {steps} steps, {code_steps} code, "
                    f"{tool_calls} tool calls/step, {iterations} iterations")

    pipeline = PipelineShape(
        steps=steps,
        code_steps=code_steps,
        tool_calls_per_step=tool_calls,
        iterations=iterations,
    )

    click.echo(f"  Steps: {pipeline.steps} ({pipeline.code_steps} produce code)")
    if pipeline.tool_calls_per_step:
        click.echo(f"  Tool calls/step: {pipeline.tool_calls_per_step}")
    if pipeline.iterations > 1:
        click.echo(f"  Iterations: {pipeline.iterations}")
    click.echo()

    # Project overhead for each model
    projections = []
    for model in models:
        proj = project_overhead(model, pipeline)
        projections.append(proj)

    # Comparison table
    click.echo("=" * 80)
    click.echo("SECURITY MODEL COMPARISON")
    click.echo("=" * 80)

    header = f"{'Model':<30s} {'Checks':>7s} {'LLM':>5s} {'Det':>5s} {'Latency':>9s} {'$/mo(1M)':>10s}"
    click.echo(header)
    click.echo("-" * 80)

    for proj in projections:
        click.echo(
            f"{proj.model_name:<30s} "
            f"{proj.total_checks:>7d} "
            f"{proj.llm_checks:>5d} "
            f"{proj.deterministic_checks:>5d} "
            f"{proj.total_latency_ms/1000:>8.1f}s "
            f"${proj.cost_per_month_1m:>9,.0f}"
        )

    # Detailed breakdown for each model
    for proj in projections:
        click.echo(f"\n{'─' * 80}")
        click.echo(f"  {proj.model_name} (v{next((m.version for m in models if m.name == proj.model_name), '')})")
        click.echo(f"{'─' * 80}")

        for item in proj.breakdown:
            cp = item["checkpoint"]
            guards = ", ".join(item["guards"])
            checks = item["total_checks"]
            lat = item["latency_ms"]
            click.echo(f"  {cp:<20s} {checks:>3d} checks  {lat:>8.1f}ms  [{guards}]")

        click.echo(f"  {'':20s} {'───':>3s}         {'───────':>8s}")
        click.echo(f"  {'TOTAL':<20s} {proj.total_checks:>3d} checks  {proj.total_latency_ms:>8.1f}ms")
        click.echo(f"  LLM-based checks: {proj.llm_checks}  |  Deterministic: {proj.deterministic_checks}")

    # Multiplier comparison
    if len(projections) > 1:
        baseline = projections[0]
        click.echo(f"\n{'=' * 80}")
        click.echo(f"OVERHEAD MULTIPLIERS (vs {baseline.model_name})")
        click.echo(f"{'=' * 80}")
        click.echo(f"{'Model':<30s} {'Checks':>8s} {'Latency':>9s} {'Cost':>8s}")
        click.echo("-" * 60)
        for proj in projections:
            cm = proj.total_checks / baseline.total_checks if baseline.total_checks else 0
            lm = proj.total_latency_ms / baseline.total_latency_ms if baseline.total_latency_ms else 0
            costm = proj.cost_per_month_1m / baseline.cost_per_month_1m if baseline.cost_per_month_1m else 0
            click.echo(f"{proj.model_name:<30s} {cm:>7.1f}x {lm:>8.1f}x {costm:>7.1f}x")


if __name__ == "__main__":
    cli()
