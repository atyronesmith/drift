# Drift

LLMs are non-deterministic — even at temperature=0, the same prompt can produce different outputs. In multi-agent pipelines, each agent's output becomes the next agent's input. Small variance at step 1 feeds into step 2, which feeds into step 3, and so on. When multiple agents run in parallel and their outputs are combined, the variance has even more surface area to compound.

Drift quantifies this. It runs identical multi-agent pipelines repeatedly, captures every intermediate output at each step, and measures how output variability grows with pipeline depth and breadth.

## How it works

Three independent phases connected by JSON files on disk:

1. **Collect** — run the full pipeline N times with identical inputs, saving every intermediate output at each step
2. **Measure** — compute text similarity metrics across outputs, per step and end-to-end
3. **Analyze** — statistical tests on the metrics to characterize how variance compounds (coming soon)

The baseline experiment starts simple (single LLM call, N runs) to establish that variance exists. Subsequent experiments add pipeline depth (chained agents) and breadth (parallel agents with combined outputs) to measure compounding.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configure

Copy `.env.example` to `.env` and set your Ollama URL:

```bash
cp .env.example .env
# edit .env — set OLLAMA_BASE_URL if not localhost
```

Or set `base_url` directly in the experiment YAML.

### Run an experiment

```bash
# Run the baseline code review task 30 times against Ollama
drift run experiments/baseline.yaml --runs 30

# Compute metrics on the results
drift measure results/baseline/<timestamp>/
```

### Experiment config format

Experiments are YAML files:

```yaml
experiment:
  name: baseline
  model: qwen2.5:7b
  temperature: 0
  system_prompt: |
    You are an expert code reviewer...
  user_prompt: |
    Review this Python function...
```

### Metrics output

`drift measure` outputs JSON with:

- `exact_match_rate` — fraction of outputs identical to the first
- `mean_edit_distance` — average normalized Levenshtein distance across all pairs
- `max_edit_distance` — worst-case divergence between any two outputs
- `length_cv` — coefficient of variation of output lengths
- `num_unique` — count of distinct outputs

## Project structure

```
src/drift/
├── cli.py              # CLI entry point
├── storage.py          # JSON read/write for results
├── providers/
│   ├── base.py         # LLMResponse dataclass, Provider protocol
│   └── ollama.py       # Ollama API (httpx, no SDK)
├── metrics/
│   └── text.py         # Text similarity metrics
├── pipeline/           # Multi-step pipelines (Phase 2)
├── runner/             # N-run orchestration (Phase 2)
└── analysis/           # Statistical tests (Phase 3)
```

## Dependencies

httpx, pyyaml, numpy, scipy, python-Levenshtein, click. No LLM SDKs.
