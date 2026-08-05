# Contributing to Drift

## Getting Started

```bash
git clone https://github.com/atyronesmith/drift.git
cd drift
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[classifiers]"
```

Copy `.env.example` to `.env` and configure your provider settings.

## Development

### Project Structure

- `src/drift/` — main package
- `tests/` — test suite
- `tasks/` — task definitions (input files, prompts, functional tests)
- `experiments/` — experiment YAML configs
- `scripts/` — one-off experiment scripts
- `docs/` — documentation

### Running Tests

```bash
pytest tests/
```

### Code Style

- Python 3.11+
- No strict formatter enforced — match the style of surrounding code
- Type hints encouraged but not required
- Keep imports sorted: stdlib, third-party, local

### Providers Use curl Subprocess

All providers make HTTP calls via `subprocess` + `curl` rather than Python HTTP libraries. This is intentional — it avoids endpoint security software blocking Python's direct TCP connections. Follow this pattern when adding new providers.

## Adding Experiments

1. Create an experiment YAML in `experiments/`
2. Run it: `drift run experiments/my-experiment.yaml --runs 10`
3. Measure: `drift measure results/my-experiment/<timestamp>/`
4. Document results in `docs/RESULTS.md`

## Adding Security Models

Security models define the guard checkpoints for a particular security architecture. They live in `security_models/` as YAML files:

```yaml
name: My Security Model
version: "2026"
description: Description of the model
source: https://example.com/docs

checkpoints:
  step_input:
    - {name: content_filter, latency_ms: 290, type: llm, description: "Input safety"}
  step_output:
    - {name: content_filter, latency_ms: 290, type: llm, description: "Output safety"}
  code_output:
    - {name: code_shield, latency_ms: 50, type: static_analysis}
  tool_call:
    - {name: param_validation, latency_ms: 1, type: deterministic}
  tool_response:
    - {name: content_filter, latency_ms: 290, type: llm}
  inter_agent:
    - {name: identity_check, latency_ms: 1, type: crypto}
  audit:
    - {name: audit_log, latency_ms: 2, type: deterministic}
```

Checkpoint types: `step_input`, `step_output`, `code_output`, `tool_call`, `tool_response`, `inter_agent`, `audit`. Guard types: `llm`, `classifier`, `deterministic`, `crypto`, `static_analysis`.

Test with: `drift safety-overhead -m my_model --steps 3 --code-steps 1`

## Adding Tasks

Each task needs:

- `tasks/<name>/input.md` — the input document
- `tasks/<name>/*.md` — prompt files (system, user)
- `tasks/<name>/functional_tests.py` — pytest functions that validate outputs
- An experiment YAML referencing the task

## Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request with a clear description of what changed and why

## What Not to Commit

- `.env` files (API keys, credentials)
- `results/` (experiment outputs — large, machine-specific)
- `datasets/` (downloaded data files — e.g. `datasets/ansible_triage/inputs/` for scrubbed AAP failure logs, `datasets/ansible_triage/digested/` for pre-digested copies)
- Large binary files

## Reporting Issues

Open an issue on GitHub with:

- What you were trying to do
- What happened instead
- Steps to reproduce
- Provider/model/OS details if relevant
