from __future__ import annotations

from dataclasses import dataclass

import click

from drift.providers.factory import make_provider


@dataclass
class StepResult:
    name: str
    response: dict  # LLMResponse.to_dict()


def _render_prompt(template: str, context: dict) -> str:
    """Render a prompt template with available context variables.

    Available variables:
        {input}           — the original pipeline input (always available)
        {previous_output} — output from the immediately preceding step
        {step_N}          — output from step N by index (e.g. {step_0}, {step_1})
        {step_NAME}       — output from step by name (e.g. {step_reviewer})
    """
    return template.format_map(_SafeDict(context))


class _SafeDict(dict):
    """Dict that returns the key as-is for missing format variables."""
    def __missing__(self, key):
        return "{" + key + "}"


async def run_pipeline(steps_config: list[dict], initial_input: str) -> list[StepResult]:
    """Execute a pipeline of steps, chaining outputs via prompt templates."""
    results = []
    context = {"input": initial_input}

    for i, step in enumerate(steps_config):
        name = step.get("name", f"step_{i}")
        provider = make_provider(step)
        system = step["system_prompt"]

        # Build the user prompt from template, or default to previous output
        template = step.get("user_prompt", "{previous_output}")
        if i == 0 and "user_prompt" not in step:
            template = "{input}"
        prompt = _render_prompt(template, context)

        click.echo(f"    Step {i + 1}/{len(steps_config)} ({name})...", nl=False)
        response = await provider.complete(system, prompt, temperature=step.get("temperature", 0))
        click.echo(f" {response.total_ms:.0f}ms")

        results.append(StepResult(name=name, response=response.to_dict()))

        # Update context for subsequent steps
        context["previous_output"] = response.content
        context[f"step_{i}"] = response.content
        context[f"step_{name}"] = response.content

    return results
