# AGENTS.md

Instructions for AI agents working on this codebase.

## Project Overview

Drift is a measurement and optimization framework for LLM output variance. It probes LLM outputs for deterministic fields, generates extraction rules, and compiles standalone Python engines that handle deterministic fields without calling the LLM.

## Architecture

- **Source code**: `src/drift/` — installed as `drift` package via `pip install -e .`
- **CLI**: `src/drift/cli.py` — Click-based, entry point is `drift`
- **Providers**: `src/drift/providers/` — Ollama, Vertex AI, DeepInfra, MLX (local + remote)
- **Hybrid framework**: `src/drift/hybrid/` — rules, executor, compiler, autogen
- **Probing**: `src/drift/probing/` — logprob entropy analysis, attention weights
- **Pipeline**: `src/drift/pipeline/` — multi-step pipeline execution
- **Tasks**: `tasks/<name>/` — each task has `input.md`, prompt files, and `functional_tests.py`
- **Experiments**: `experiments/*.yaml` — experiment configurations
- **Scripts**: `scripts/` — one-off experiment scripts (not part of the package)
- **Tests**: `tests/` — pytest test suite

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[classifiers]"
```

## Key Conventions

- **Python 3.11+** required
- **Build system**: hatchling (configured in `pyproject.toml`)
- **Dependencies**: managed with `uv`, locked in `uv.lock`
- **HTTP requests in providers**: Use `subprocess` + `curl`, not Python HTTP libraries. This avoids endpoint security software blocking direct TCP connections from Python processes.
- **Environment variables**: API keys and URLs come from `.env` (loaded via `python-dotenv`). Never hardcode credentials.
- **Results directory**: `results/` is gitignored. Experiment outputs go here.
- **Datasets directory**: `datasets/` is gitignored. Downloaded data goes here.

## Running Tests

```bash
pytest tests/
```

## CLI Commands

The main commands (run `drift --help` for full list):

| Command | Purpose |
|---------|---------|
| `drift run <config.yaml> --runs N` | Run experiment N times |
| `drift measure <results_dir>` | Compute text + functional metrics |
| `drift analyze <path>` | Generate recommendations |
| `drift compare <a> <b>` | Side-by-side comparison |
| `drift probe <config.yaml> --rules` | Token entropy analysis |
| `drift synthesize <probe.json> --input <file>` | Generate rules from probe |
| `drift compile <hybrid.yaml> --prompt <file>` | Compile to standalone engine |
| `drift build <config.yaml> -o engine.py` | One-command: probe + synthesize + compile |
| `drift validate <engine.py> --input-dir <dir>` | Test engine against inputs |
| `drift refine <engine.py> --input-dir <dir>` | Iterative refinement loop |
| `drift distill <config.yaml> --input-dir <dir>` | Train classifiers from LLM labels |
| `drift probe-pipeline <config.yaml>` | Per-step pipeline probing |
| `drift serve <model>` | Start MLX inference server |

## Adding a New Provider

1. Create `src/drift/providers/<name>.py`
2. Implement a class with `complete(system, prompt, temperature, ...)` method returning `LLMResponse`
3. Use `subprocess` + `curl` for HTTP (not `requests` or `httpx` for provider calls)
4. Register in `src/drift/cli.py` provider selection logic

## Adding a New Task

1. Create `tasks/<task_name>/` directory
2. Add `input.md` (the input document)
3. Add prompt files referenced by experiment YAML (`system.md`, `user.md`, etc.)
4. Add `functional_tests.py` with pytest-style test functions that validate LLM outputs
5. Create an experiment YAML in `experiments/`

## Rule Types

Rules in `src/drift/hybrid/rules.py`:

| Type | Description |
|------|-------------|
| `regex` | Extract via regex with capture group |
| `literal` | Fixed value |
| `regex_all` | Extract all matches into a list |
| `keyword` | Keyword presence detection |
| `classifier` | Logistic regression with confidence gating |
| `mapping` | Value lookup table |
| `conditional` | Conditional logic based on other fields |

## Common Pitfalls

- **Ollama URL**: Defaults to `localhost:11434`. Set `OLLAMA_BASE_URL` env var for remote servers.
- **Vertex AI**: Requires `gcloud auth application-default login` and project/region in experiment YAML.
- **MLX**: Only works on Apple Silicon. The `mlx` provider runs in-process; `mlx-remote` connects to a `drift serve` instance.
- **Large files**: Never commit datasets, results, or `.env` files.
