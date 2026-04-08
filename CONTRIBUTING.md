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
- `datasets/` (downloaded data files)
- Large binary files

## Reporting Issues

Open an issue on GitHub with:

- What you were trying to do
- What happened instead
- Steps to reproduce
- Provider/model/OS details if relevant
