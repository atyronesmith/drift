# Drift

LLMs are non-deterministic — even at temperature=0, the same prompt can produce different outputs. In multi-agent pipelines, each agent's output becomes the next agent's input. Small variance at step 1 feeds into step 2, which feeds into step 3, and so on. When multiple agents run in parallel and their outputs are combined, the variance has even more surface area to compound.

Drift quantifies this. It runs identical multi-agent pipelines repeatedly, captures every intermediate output at each step, and measures how output variability grows with pipeline depth and breadth.

## How it works

Three independent phases connected by JSON files on disk:

1. **Collect** — run the full pipeline N times with identical inputs, saving every intermediate output at each step
2. **Measure** — compute text similarity metrics across outputs, per step and end-to-end
3. **Analyze** — statistical tests on the metrics to characterize how variance compounds

The baseline experiment starts simple (single LLM call, N runs) to establish that variance exists. Subsequent experiments add pipeline depth (chained agents) and breadth (parallel agents with combined outputs) to measure compounding.

A **hybrid execution** mode automatically separates extraction (copying from input) from reasoning (generating new content). The probe step analyzes token-level entropy and attention weights to classify fields, then generates deterministic regex rules for extraction fields — reducing latency by 50%+ while preserving reasoning quality. See [docs/HYBRID.md](docs/HYBRID.md).

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

### Providers

Drift supports multiple inference backends:

| Provider | Config value | Notes |
|----------|-------------|-------|
| Ollama | `ollama` (default) | Remote at any URL, OpenAI-compatible API |
| MLX (local) | `mlx` | In-process on Apple Silicon via mlx-lm |
| MLX (remote) | `mlx-remote` | Remote Apple Silicon Mac via `drift serve` |
| Vertex AI | `vertex` | Claude on Google Cloud |
| DeepInfra | `deepinfra` | OpenAI-compatible cloud API |

### MLX server (Apple Silicon)

The MLX provider exposes capabilities that Ollama cannot: full-vocabulary logprob distributions (exact Shannon entropy) and attention weight probing (input token attribution). Run the server on a powerful Apple Silicon Mac:

```bash
# On the remote Mac:
drift serve mlx-community/Qwen2.5-32B-Instruct-4bit --host 0.0.0.0 --port 8080

# On your local machine:
export MLX_REMOTE_URL=http://<remote-mac>:8080
drift run experiments/my-experiment.yaml --runs 10
```

Endpoints: `/v1/chat/completions` (OpenAI-compatible), `/v1/logprobs` (full-vocab entropy), `/v1/attention` (attention weights). See [docs/MLX.md](docs/MLX.md).

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
├── cli.py              # CLI entry point (run, measure, probe, serve, ...)
├── storage.py          # JSON read/write for results
├── providers/
│   ├── base.py         # LLMResponse dataclass, Provider protocol
│   ├── ollama.py       # Ollama API (curl subprocess)
│   ├── mlx.py          # In-process MLX on Apple Silicon
│   ├── mlx_remote.py   # Remote MLX server client
│   ├── vertex.py       # Vertex AI (Claude)
│   └── deepinfra.py    # DeepInfra API
├── probing/
│   ├── logprobs.py     # Token entropy via Ollama logprobs
│   ├── mlx_logprobs.py # Full-vocab entropy via MLX
│   └── mlx_attention.py# Attention weight probing via MLX
├── hybrid/
│   ├── autogen.py      # Auto-generate rules from probe data
│   ├── rules.py        # Rule execution (regex, conditional, mapping)
│   ├── executor.py     # Hybrid orchestration (rules + LLM)
│   └── compiler.py     # Compile rules to standalone Python engine
├── server/
│   └── mlx_server.py   # HTTP server for remote MLX inference
├── metrics/
│   └── text.py         # Text similarity metrics
├── pipeline/           # Multi-step pipeline execution
├── runner/             # N-run orchestration
└── analysis/           # Statistical analysis and recommendations
```

## Dependencies

Core: httpx, pyyaml, numpy, scipy, python-Levenshtein, click.
MLX (optional): mlx, mlx-lm (Apple Silicon only).
