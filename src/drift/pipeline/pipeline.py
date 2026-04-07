from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import click

from drift.providers.factory import make_provider
from drift.pipeline.transforms import run_post_process
from drift.hybrid.executor import run_hybrid


@dataclass
class StepResult:
    name: str
    response: dict  # LLMResponse.to_dict()
    hybrid_metadata: dict | None = None


@dataclass
class LoopStepResult:
    name: str
    iterations: list[list[StepResult]] = field(default_factory=list)
    total_iterations: int = 0
    termination_reason: str = ""  # "condition_met" | "max_iterations" | "abort"


def _render_prompt(template: str, context: dict) -> str:
    """Render a prompt template with available context variables.

    Uses simple string replacement instead of str.format_map to avoid
    conflicts with JSON curly braces in prompt templates.

    Available variables:
        {input}              — the original pipeline input (always available)
        {previous_output}    — output from the immediately preceding step
        {step_N}             — output from step N by index (e.g. {step_0}, {step_1})
        {step_NAME}          — output from step by name (e.g. {step_reviewer})
        {loop_iteration}     — current loop iteration number (1-indexed)
        {loop_input}         — input for the current loop iteration
        {loop_previous_output} — last iteration's final step output
        {loop_step_NAME}     — output from a step within the current loop iteration
    """
    result = template
    for key, value in context.items():
        result = result.replace("{" + key + "}", str(value))
    return result


async def _run_step(
    step: dict, index: int, total: int, context: dict,
    task_dir: Path | None, prefix: str = "",
) -> tuple[StepResult, str]:
    """Execute a single pipeline step and return the result and processed content."""
    name = step.get("name", f"step_{index}")
    provider = make_provider(step)
    system = step["system_prompt"]

    template = step.get("user_prompt", "{previous_output}")
    if index == 0 and "user_prompt" not in step:
        template = "{input}"
    prompt = _render_prompt(template, context)

    hybrid_cfg = step.get("hybrid")

    if hybrid_cfg:
        # Hybrid execution: rules first, then focused LLM call
        click.echo(f"    {prefix}Step {index + 1}/{total} ({name}) [hybrid]...", nl=False)
        hybrid_result = await run_hybrid(
            provider, system, prompt,
            rules=hybrid_cfg.get("rules", []),
            llm_fields=hybrid_cfg.get("llm_fields", []),
            hybrid_prompt_template=hybrid_cfg.get("hybrid_prompt_template"),
            temperature=step.get("temperature", 0),
            merge_mode=hybrid_cfg.get("merge_mode", "llm_over_rules"),
            output_mode=hybrid_cfg.get("output_mode", "merge"),
        )
        click.echo(f" {hybrid_result.response.total_ms:.0f}ms"
                    f" (rules: {hybrid_result.rules_matched}/{hybrid_result.rules_applied})")

        hybrid_meta = {
            "rules_applied": hybrid_result.rules_applied,
            "rules_matched": hybrid_result.rules_matched,
            "extracted": hybrid_result.extracted,
        }
        result = StepResult(
            name=name,
            response=hybrid_result.response.to_dict(),
            hybrid_metadata=hybrid_meta,
        )

        if hybrid_result.merged:
            response_content = json.dumps(hybrid_result.merged)
        else:
            response_content = hybrid_result.raw_llm_response
    else:
        # Standard LLM execution
        click.echo(f"    {prefix}Step {index + 1}/{total} ({name})...", nl=False)
        response = await provider.complete(system, prompt, temperature=step.get("temperature", 0))
        click.echo(f" {response.total_ms:.0f}ms")

        result = StepResult(name=name, response=response.to_dict())
        response_content = response.content

    if "post_process" in step:
        response_content = run_post_process(
            step["post_process"], response_content, context, task_dir
        )

    return result, response_content


def _evaluate_loop_condition(condition_config: dict, step_results: list[StepResult]) -> str:
    """Evaluate whether a loop should continue based on a step's JSON output.

    Returns "continue", "stop", or "abort".
    """
    target_step = condition_config["step"]
    target_field = condition_config["field"]
    continue_values = condition_config.get("continue_values", [])
    abort_values = condition_config.get("abort_values", [])

    # Find the target step's output
    for sr in step_results:
        if sr.name == target_step:
            try:
                data = json.loads(sr.response["content"])
                value = data.get(target_field)
                if value in abort_values:
                    return "abort"
                if value in continue_values:
                    return "continue"
                return "stop"
            except (json.JSONDecodeError, KeyError):
                return "abort"

    return "abort"


async def _run_loop(
    loop_config: dict, context: dict, task_dir: Path | None,
) -> LoopStepResult:
    """Execute a loop block, repeating inner steps until condition is met or max iterations."""
    max_iterations = loop_config.get("max_iterations", 5)
    condition_config = loop_config["loop_condition"]
    inner_steps = loop_config["steps"]
    loop_name = loop_config.get("name", "loop")

    result = LoopStepResult(name=loop_name)

    # Initial loop input comes from previous_output in the parent context
    loop_input = context.get("previous_output", context.get("input", ""))

    loop_previous_output = ""

    for iteration in range(1, max_iterations + 1):
        click.echo(f"    [{loop_name}] Iteration {iteration}/{max_iterations}:")

        # Set up loop-specific context variables
        loop_context = dict(context)
        loop_context["loop_iteration"] = str(iteration)
        loop_context["loop_input"] = loop_input
        loop_context["loop_previous_output"] = loop_previous_output

        iteration_results = []

        for j, inner_step in enumerate(inner_steps):
            step_result, content = await _run_step(
                inner_step, j, len(inner_steps), loop_context,
                task_dir, prefix=f"  ",
            )
            iteration_results.append(step_result)

            # Update loop context for subsequent inner steps
            inner_name = inner_step.get("name", f"step_{j}")
            loop_context["previous_output"] = content
            loop_context[f"step_{j}"] = content
            loop_context[f"step_{inner_name}"] = content
            loop_context[f"loop_step_{inner_name}"] = content

        result.iterations.append(iteration_results)
        result.total_iterations = iteration

        # Determine loop input for next iteration
        # Use the step marked feeds_loop_input, or the last step by default
        next_loop_input = content  # last step's output
        for j, inner_step in enumerate(inner_steps):
            if inner_step.get("feeds_loop_input"):
                next_loop_input = loop_context.get(
                    f"step_{inner_step.get('name', f'step_{j}')}", content
                )
                break

        loop_previous_output = content

        # Evaluate loop condition
        decision = _evaluate_loop_condition(condition_config, iteration_results)
        if decision == "abort":
            click.echo(f"    [{loop_name}] Aborted at iteration {iteration}")
            result.termination_reason = "abort"
            break
        elif decision == "stop":
            click.echo(f"    [{loop_name}] Condition met at iteration {iteration}")
            result.termination_reason = "condition_met"
            break
        else:
            loop_input = next_loop_input
    else:
        click.echo(f"    [{loop_name}] Max iterations ({max_iterations}) reached")
        result.termination_reason = "max_iterations"

    # Update parent context with loop's final output
    context["previous_output"] = loop_previous_output
    context[f"step_{loop_name}"] = loop_previous_output

    return result


async def run_pipeline(steps_config: list[dict], initial_input: str, task_dir: Path | None = None) -> list:
    """Execute a pipeline of steps, chaining outputs via prompt templates."""
    results = []
    context = {"input": initial_input}

    for i, step in enumerate(steps_config):
        if step.get("type") == "loop":
            loop_result = await _run_loop(step, context, task_dir)
            results.append(loop_result)
        else:
            step_result, content = await _run_step(
                step, i, len(steps_config), context, task_dir,
            )
            results.append(step_result)

            context["previous_output"] = content
            context[f"step_{i}"] = content
            name = step.get("name", f"step_{i}")
            context[f"step_{name}"] = content

    return results
