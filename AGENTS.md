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
- **Safety**: `src/drift/safety/` — guard model wrapper, security model evaluation
- **Tasks**: `tasks/<name>/` — each task has `input.md`, prompt files, and `functional_tests.py`
- **Experiments**: `experiments/*.yaml` — experiment configurations
- **Security models**: `security_models/*.yaml` — security architecture definitions for overhead projection
- **Scripts**: `scripts/` — safety audit and one-off experiment scripts
- **Tests**: `tests/` — pytest test suite

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[classifiers]"
```

Some scripts (e.g. `scripts/podlog_predigest_measure.py`) import the sibling [logdigest](https://github.com/atyronesmith/logdigest) repo. Clone it alongside drift:

```bash
git clone https://github.com/atyronesmith/logdigest.git ../logdigest
```

## Key Conventions

- **Python 3.11+** required
- **Build system**: hatchling (configured in `pyproject.toml`)
- **Dependencies**: managed with `uv`, locked in `uv.lock`
- **HTTP requests in providers**: Use `subprocess` + `curl`, not Python HTTP libraries. This avoids endpoint security software blocking direct TCP connections from Python processes.
- **Environment variables**: API keys and URLs come from `.env` (loaded via `python-dotenv`). Never hardcode credentials.
- **Results directory**: `results/` is gitignored. Experiment outputs go here.
- **Datasets directory**: `datasets/` is gitignored. Downloaded data goes here (e.g. `datasets/ansible_triage/inputs/` and `datasets/ansible_triage/digested/` for the Ansible log corpus).

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
| `drift safety-overhead [--pipeline config.yaml]` | Compare guard overhead across security models |
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

## Safety Module

- **Guard model** (`src/drift/safety/guard.py`): Wraps Llama Guard on Ollama via curl subprocess. Supports `exclude_categories` to filter false positives (e.g., S6/S8 for code workloads).
- **Security models** (`src/drift/safety/models.py`): Loads YAML security model definitions, projects guard overhead for pipeline shapes. Used by `drift safety-overhead`.
- **Batch audit** (`scripts/safety_audit.py`): Runs guard checks on all saved results. Outputs per-experiment rejection rates, category breakdown, latency stats, and retry death spiral projections.
- **Live guard** (`drift run --safety-guard`): Per-step guard checking during experiment runs. Results saved in run JSON under `"guard"` key.
- **Security model YAML format**: See `security_models/` for examples. Checkpoints: `step_input`, `step_output`, `code_output`, `tool_call`, `tool_response`, `inter_agent`, `audit`.

## Common Pitfalls

- **Ollama URL**: Defaults to `localhost:11434`. Set `OLLAMA_BASE_URL` env var for remote servers.
- **Vertex AI**: Requires `gcloud auth application-default login` and project/region in experiment YAML.
- **MLX**: Only works on Apple Silicon. The `mlx` provider runs in-process; `mlx-remote` connects to a `drift serve` instance.
- **Large files**: Never commit datasets, results, or `.env` files.
