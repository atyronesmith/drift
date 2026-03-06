from __future__ import annotations

import copy
from pathlib import Path


def resolve_task_prompts(config: dict, tasks_dir: Path) -> dict:
    """Resolve file-based prompt references in experiment config.

    - ``task: code_review`` loads ``tasks/code_review/input.md`` as ``input``
    - ``system_prompt_file`` / ``user_prompt_file`` load file content
    - Inline values always override file references
    """
    config = copy.deepcopy(config)
    exp = config["experiment"]

    task = exp.get("task")
    task_dir = tasks_dir / task if task else None

    # Load input from task directory
    if task_dir and "input" not in exp:
        input_file = task_dir / "input.md"
        if input_file.exists():
            exp["input"] = input_file.read_text()

    # Resolve top-level prompt file refs (baselines)
    _resolve_prompt_fields(exp, task_dir)

    # Resolve step-level prompt file refs (pipelines)
    for step in exp.get("steps", []):
        _resolve_prompt_fields(step, task_dir)

    return config


def _resolve_prompt_fields(d: dict, task_dir: Path | None) -> None:
    """Resolve system_prompt_file and user_prompt_file in a dict."""
    for field in ("system_prompt", "user_prompt"):
        file_key = f"{field}_file"
        if file_key in d and field not in d:
            path = (task_dir or Path.cwd()) / d[file_key]
            d[field] = path.read_text()
            del d[file_key]
