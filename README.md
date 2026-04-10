# Drift

LLMs are non-deterministic — even at temperature=0, the same prompt can produce different outputs. In multi-agent pipelines, each agent's output becomes the next agent's input. Small variance at step 1 feeds into step 2, which feeds into step 3, and so on. When multiple agents run in parallel and their outputs are combined, the variance has even more surface area to compound.

Drift quantifies this. It runs identical multi-agent pipelines repeatedly, captures every intermediate output at each step, and measures how output variability grows with pipeline depth and breadth.

## What to target

Drift works best on AI applications with **repeating, structured workloads** — not one-off chat. Look for:

**Strong candidates:**
- **Structured extraction pipelines** — LLM reads input (logs, tickets, invoices, reports) and outputs structured JSON with defined fields. Fields that copy from the input (IDs, names, dates) can be extracted with rules; classification fields can be distilled into classifiers.
- **Multi-step LLM pipelines** — Summarize → Classify → Route → Act chains where intermediate steps have constrained outputs (categories, binary decisions). Each deterministic step you eliminate removes an LLM round-trip.
- **Large-input summarization** — Logs, traces, documents where 70-90% of the input is noise. Pre-digestion (deterministic preprocessing) can cut input tokens by 80%+ while improving output quality.
- **Batch processing** — The same prompt template applied to hundreds or thousands of inputs. Small per-unit savings compound to significant cost reduction at scale.
- **Classification with known categories** — Routing, triage, labeling tasks where the output is one of N known values. Classifier distillation can replace the LLM call entirely for high-confidence predictions.

**Weak candidates:**
- **Open-ended chat / Q&A** — Free-form conversational responses with no structured output. Nothing to extract deterministically.
- **RAG chatbots** — Quality depends on retrieval, not generation determinism. Single LLM call, variable inputs, unstructured output.
- **Creative generation** — Content writing, code generation, brainstorming. The output IS the creative work — there's nothing to shortcut.
- **Demo/quickstart repos** — Educational notebooks with one-off examples, no repeating workload.

**The test:** If you can describe the LLM's output as "fill in these fields," drift can help. If the output is "write me a paragraph," it probably can't.

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

### Safety overhead analysis

Drift measures the cost of AI safety infrastructure — guard models, content filters, and guardrail frameworks — on real workloads:

```bash
# Audit existing results against a guard model
python scripts/safety_audit.py --model llama-guard3:1b --exclude-categories S6,S8

# Run experiments with live safety guard checking
drift run experiments/pipeline-structured.yaml --runs 10 \
    --safety-guard llama-guard3:1b --exclude-categories S6,S8

# Compare overhead across security architectures
drift safety-overhead --pipeline experiments/pipeline-structured.yaml
drift safety-overhead --steps 3 --code-steps 1 --tool-calls 1 --iterations 5
```

Ships with 4 security model definitions (`security_models/`):
- **Drift Baseline** — 1 output check per step
- **Meta LlamaFirewall** — PromptGuard + AlignmentCheck + CodeShield
- **Microsoft Azure Foundry** — 4 intervention points + governance toolkit
- **FINOS Regulated FSI** — 36 controls, 7 trust boundaries (SOX/SR 11-7)

Key finding: Llama Guard 3-1B produces a 47.8% false positive rate on legitimate software engineering outputs (95% on code generation). Category overrides (excluding S6/S8) drop this to 1.3%. Production security architectures require 15-290 guard checks per pipeline run vs. the 3 we measured. See [docs/ai-safety-overhead.md](docs/ai-safety-overhead.md).

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
├── safety/
│   ├── guard.py        # Guard model wrapper (Llama Guard on Ollama)
│   └── models.py       # Security model definitions and overhead projection
├── server/
│   └── mlx_server.py   # HTTP server for remote MLX inference
├── metrics/
│   └── text.py         # Text similarity metrics
├── pipeline/           # Multi-step pipeline execution
├── runner/             # N-run orchestration
└── analysis/           # Statistical analysis and recommendations

security_models/            # Security architecture definitions (YAML)
scripts/
└── safety_audit.py     # Batch safety audit of saved results
```

## Dependencies

Core: httpx, pyyaml, numpy, scipy, python-Levenshtein, click.
MLX (optional): mlx, mlx-lm (Apple Silicon only).
Classifiers (optional): scikit-learn.

## Documentation

- [Technical paper](docs/PAPER.md) — full findings across 17 experiments
- [Experiment results](docs/RESULTS.md) — detailed per-experiment data
- [Hybrid framework](docs/HYBRID.md) — automated rule extraction pipeline
- [AI safety overhead](docs/ai-safety-overhead.md) — safety tax measurement, guard false positives, security model comparison
- [MLX provider](docs/MLX.md) — Apple Silicon inference and attention probing
- [Theory](docs/THEORY.md) — voting vs deliberation, distillation thresholds
- [Management proposal](docs/PROPOSAL.md) — dollar-cost analysis for decision makers

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

See [LICENSE](LICENSE).
