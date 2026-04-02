# MLX Provider

MLX on Apple Silicon gives drift capabilities that Ollama and cloud APIs cannot provide:

1. **Full-vocab logprobs** — exact Shannon entropy over the entire vocabulary (~150K tokens), not a top-k approximation
2. **Attention weights** — which input tokens the model attended to when generating each output token

These enable better extraction/reasoning classification and attention-based rule discovery.

## Modes

| Mode | Config | Use case |
|------|--------|----------|
| In-process | `provider: mlx` | Local Mac with enough memory for the model |
| Remote server | `provider: mlx-remote` | Powerful remote Mac, lightweight local machine |

Both modes expose the same capabilities. The remote server is a thin HTTP wrapper around the same code paths.

## Setup

### Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.11+, `mlx` and `mlx-lm` packages
- Sufficient unified memory for the model (32B 4-bit ~ 18GB)

### Install

```bash
pip install -e .
# mlx and mlx-lm are optional dependencies — install separately:
pip install mlx mlx-lm
```

## In-process usage

Set `provider: mlx` in your experiment YAML:

```yaml
experiment:
  name: my-experiment
  model: mlx-community/Qwen2.5-32B-Instruct-4bit
  provider: mlx
  temperature: 0
```

The model loads on first use and stays in memory for subsequent calls.

## Remote server

### Starting the server

On the remote Apple Silicon machine:

```bash
drift serve mlx-community/Qwen2.5-32B-Instruct-4bit --host 0.0.0.0 --port 8080
```

Or run directly:

```bash
python -m drift.server.mlx_server mlx-community/Qwen2.5-32B-Instruct-4bit --host 0.0.0.0 --port 8080
```

The model loads once at startup. The server handles requests sequentially (single-threaded, matching MLX's execution model).

### Client configuration

On the local machine, set the remote URL via environment variable or experiment YAML:

```bash
export MLX_REMOTE_URL=http://<remote-mac>:8080
drift run experiments/my-experiment.yaml --runs 10
```

Or in the YAML:

```yaml
experiment:
  name: my-experiment
  model: mlx-community/Qwen2.5-32B-Instruct-4bit
  provider: mlx-remote
  base_url: http://<remote-mac>:8080
```

### Endpoints

#### `GET /health`

Returns server status and loaded model.

```bash
curl http://localhost:8080/health
```

```json
{"status": "ok", "model": "mlx-community/Qwen2.5-32B-Instruct-4bit", "loaded": true}
```

#### `POST /v1/chat/completions`

OpenAI-compatible chat completions. Works with any OpenAI client.

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-32B-Instruct-4bit",
    "messages": [
      {"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "Capital of France?"}
    ],
    "temperature": 0,
    "max_tokens": 64
  }'
```

Response includes `usage` (token counts) and `timing` (latency).

#### `POST /v1/logprobs`

Full-vocabulary logprob probing. Returns exact Shannon entropy per token — not available from Ollama or the mlx-lm built-in server, which only provide top-k logprobs.

```bash
curl -s http://localhost:8080/v1/logprobs \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-32B-Instruct-4bit",
    "messages": [
      {"role": "system", "content": "Answer in one word."},
      {"role": "user", "content": "Capital of France?"}
    ],
    "max_tokens": 16,
    "top_k_report": 10
  }'
```

Response:

```json
{
  "content": "Paris",
  "model": "mlx-community/Qwen2.5-32B-Instruct-4bit",
  "tokens": [
    {
      "token": "Paris",
      "logprob": -0.005,
      "prob": 0.995,
      "exact_entropy": 0.06,
      "top_alternatives": [
        {"token": "London", "logprob": -5.2},
        {"token": "The", "logprob": -6.1}
      ]
    }
  ],
  "entropy_stats": {
    "mean_entropy": 0.06,
    "pct_deterministic": 1.0
  }
}
```

Parameters:
- `top_k_report` (default 10) — number of top alternative tokens to include per position

#### `POST /v1/attention`

Attention weight probing. Returns per-output-token attribution over input positions, revealing which input tokens drive each output token. Uses a two-phase approach: generate normally first, then replay with patched attention to capture weights without affecting output quality.

```bash
curl -s http://localhost:8080/v1/attention \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-32B-Instruct-4bit",
    "messages": [
      {"role": "system", "content": "You classify ticket severity."},
      {"role": "user", "content": "Priority: Medium\nCategory: Network"}
    ],
    "max_tokens": 64,
    "top_k_attend": 20
  }'
```

Response:

```json
{
  "content": "P3",
  "model": "...",
  "n_input": 45,
  "input_tokens": ["<|im_start|>", "system", "...", "Priority", ":", " Medium", "..."],
  "output_tokens": [
    {
      "token": "P",
      "token_id": 47,
      "position": 0,
      "top_attended": [[12, 0.15], [34, 0.12], [5, 0.08]]
    }
  ]
}
```

Parameters:
- `top_k_attend` (default 20) — top attended input positions per output token
- `last_n_layers` (default 1) — number of final transformer layers to average attention from
- `store_per_head` (default false) — include per-head attention weights (large)

## Why not the mlx-lm built-in server?

The `mlx_lm.server` command provides an OpenAI-compatible endpoint, but only exposes top-k logprobs — the same limitation as Ollama. It does not provide:

- Full-vocabulary log probability distributions (needed for exact Shannon entropy)
- Attention weight capture (needed for input attribution)

The drift MLX server accesses model internals directly via `generate_step()` (for logprobs) and monkey-patched attention layers (for weights), which requires running in the same process as the model.

## Architecture

```
Remote Mac (Apple Silicon, 64GB+)          Local Mac (drift CLI)
+---------------------------------+       +------------------------+
|  drift serve <model>            |       |  drift probe/run       |
|                                 | HTTP  |                        |
|  POST /v1/chat/completions     |<------|  MLXRemoteProvider     |
|  POST /v1/logprobs             |       |  (curl subprocess)     |
|  POST /v1/attention            |       |                        |
|  GET  /health                  |       +------------------------+
|                                 |
|  mlx-lm model in memory        |
+---------------------------------+
```

The client uses `curl` subprocess for HTTP (not Python sockets) to avoid endpoint security software interfering with Python's direct TCP connections.

## Models

Any model in `mlx-community` format works. Recommended:

| Model | Memory | Notes |
|-------|--------|-------|
| `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | ~1GB | Testing only |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | ~5GB | Good for development |
| `mlx-community/Qwen2.5-32B-Instruct-4bit` | ~18GB | Production probing |
