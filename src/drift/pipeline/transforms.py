from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path


def strip_fences(text: str) -> str:
    """Remove markdown code fences."""
    return re.sub(r"```\w*\n?(.*?)```", r"\1", text, flags=re.DOTALL).strip()


def validate_json(text: str) -> str:
    """Parse as JSON; raise on invalid input, return pretty-printed."""
    return json.dumps(json.loads(text), indent=2)


def validate_python(text: str) -> str:
    """Check Python syntax with ast.parse(); raise on syntax error."""
    ast.parse(text)
    return text


def strip_whitespace(text: str) -> str:
    """Strip leading/trailing whitespace."""
    return text.strip()


BUILTINS = {
    "strip_fences": strip_fences,
    "validate_json": validate_json,
    "validate_python": validate_python,
    "strip_whitespace": strip_whitespace,
}


def run_post_process(
    steps: list, output: str, context: dict, task_dir: Path | None
) -> str:
    """Run post-processing chain on step output."""
    for step in steps:
        if isinstance(step, str) and step in BUILTINS:
            output = BUILTINS[step](output)
        elif isinstance(step, dict) and "run" in step:
            script_path = (task_dir or Path.cwd()) / step["run"]
            output = _run_script(script_path, output, context)
        else:
            raise ValueError(f"Unknown post-process step: {step}")
    return output


def _run_script(script_path: Path, output: str, context: dict) -> str:
    """Load and execute a script's process() function."""
    spec = importlib.util.spec_from_file_location("_post_process", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.process(output, context)
