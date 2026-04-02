"""Drift MLX inference server.

Exposes MLX model capabilities over HTTP with endpoints for:
  - Standard chat completions (OpenAI-compatible)
  - Full-vocab logprob probing (exact Shannon entropy)
  - Attention weight probing (input attribution)

Usage:
    drift serve <model> [--host 0.0.0.0] [--port 8080]
    python -m drift.server.mlx_server <model> [--host 0.0.0.0] [--port 8080]
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Lock

log = logging.getLogger("drift.mlx_server")

# Model is loaded once and shared across requests
_model = None
_tokenizer = None
_model_name = ""
_model_lock = Lock()


@contextmanager
def _quiet_stderr():
    """Suppress stderr (huggingface download progress bars)."""
    old = sys.stderr
    sys.stderr = io.StringIO()
    old_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        sys.stderr = old
        os.dup2(old_fd, 2)
        os.close(old_fd)


def _ensure_model(model_name: str, quiet: bool = False):
    """Load the MLX model (called once on first request or at startup)."""
    global _model, _tokenizer, _model_name
    with _model_lock:
        if _model is not None:
            return
        from mlx_lm import load
        log.info(f"Loading model: {model_name}")
        t0 = time.time()
        if quiet:
            with _quiet_stderr():
                _model, _tokenizer = load(model_name)
        else:
            _model, _tokenizer = load(model_name)
        log.info(f"Model loaded in {time.time() - t0:.1f}s")
        _model_name = model_name


class DriftMLXHandler(BaseHTTPRequestHandler):
    """HTTP request handler for drift MLX server."""

    def log_message(self, format, *args):
        log.info(format % args)

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        return json.loads(body) if body else {}

    def _parse_messages(self, data: dict) -> tuple[str, str]:
        """Extract system and user prompts from OpenAI-style messages."""
        messages = data.get("messages", [])
        system = ""
        user = ""
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            elif msg["role"] == "user":
                user = msg["content"]
        return system, user

    def do_GET(self):
        if self.path == "/health":
            self._send_json({
                "status": "ok",
                "model": _model_name,
                "loaded": _model is not None,
            })
        else:
            self._send_json({"error": f"Unknown path: {self.path}"}, 404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            if self.path == "/v1/chat/completions":
                self._handle_completions()
            elif self.path == "/v1/logprobs":
                self._handle_logprobs()
            elif self.path == "/v1/attention":
                self._handle_attention()
            else:
                self._send_json({"error": f"Unknown path: {self.path}"}, 404)
        except Exception as e:
            log.exception("Request failed")
            self._send_json({"error": str(e)}, 500)

    def _handle_completions(self):
        """OpenAI-compatible chat completions."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        data = self._read_json()
        system, user = self._parse_messages(data)
        temperature = data.get("temperature", 0)
        max_tokens = data.get("max_tokens", 4096)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        formatted = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = _tokenizer.encode(formatted)

        sampler = make_sampler(temp=temperature if temperature > 0 else 0)
        t0 = time.perf_counter()
        output = generate(
            _model, _tokenizer, prompt=formatted,
            max_tokens=max_tokens, sampler=sampler,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        output_ids = _tokenizer.encode(output)

        self._send_json({
            "id": f"chatcmpl-drift-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(input_ids),
                "completion_tokens": len(output_ids),
                "total_tokens": len(input_ids) + len(output_ids),
            },
            "timing": {"total_ms": round(elapsed_ms, 1)},
        })

    def _handle_logprobs(self):
        """Full-vocab logprob probing with exact Shannon entropy."""
        from drift.probing.mlx_logprobs import _probe_logprobs_with_model

        data = self._read_json()
        system, user = self._parse_messages(data)

        t0 = time.perf_counter()
        result = _probe_logprobs_with_model(
            _model, _tokenizer, system, user,
            model_name=_model_name,
            max_tokens=data.get("max_tokens", 4096),
            temperature=data.get("temperature", 0),
            top_k_report=data.get("top_k_report", 10),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self._send_json({
            "content": result.content,
            "model": result.model,
            "tokens": [
                {
                    "token": t.token,
                    "logprob": t.logprob,
                    "prob": t.prob,
                    "exact_entropy": t.entropy,
                    "top_alternatives": t.top_alternatives,
                }
                for t in result.tokens
            ],
            "entropy_stats": result.entropy_stats(),
            "timing": {"total_ms": round(elapsed_ms, 1)},
        })

    def _handle_attention(self):
        """Attention weight probing with input attribution."""
        from drift.probing.mlx_attention import _probe_attention_with_model

        data = self._read_json()
        system, user = self._parse_messages(data)

        t0 = time.perf_counter()
        result = _probe_attention_with_model(
            _model, _tokenizer, system, user,
            model_name=_model_name,
            max_tokens=data.get("max_tokens", 4096),
            temperature=data.get("temperature", 0),
            top_k_attend=data.get("top_k_attend", 20),
            last_n_layers=data.get("last_n_layers", 1),
            store_per_head=data.get("store_per_head", False),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self._send_json({
            "content": result.content,
            "model": result.model,
            "n_input": result.n_input,
            "input_tokens": result.input_tokens,
            "output_tokens": [
                {
                    "token": p.token,
                    "token_id": p.token_id,
                    "position": p.position,
                    "top_attended": p.top_attended,
                    "per_head": p.per_head,
                }
                for p in result.output_tokens
            ],
            "timing": {"total_ms": round(elapsed_ms, 1)},
        })


def start_server(model: str, host: str = "0.0.0.0", port: int = 8080,
                  log_file: str | None = None):
    """Start the drift MLX server."""
    import click

    handlers: list[logging.Handler] = []
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    else:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=handlers,
    )

    _ensure_model(model, quiet=True)
    click.echo(f"Model loaded: {_model_name}")
    if log_file:
        click.echo(f"Logging to: {log_file}")
    click.echo()
    server = HTTPServer((host, port), DriftMLXHandler)
    click.echo(f"Serving on http://{host}:{port}")
    click.echo(f"Endpoints:")
    click.echo(f"  POST /v1/chat/completions  — OpenAI-compatible inference")
    click.echo(f"  POST /v1/logprobs          — full-vocab entropy probing")
    click.echo(f"  POST /v1/attention         — attention weight probing")
    click.echo(f"  GET  /health               — server status")
    click.echo()
    click.echo(f"Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Drift MLX inference server")
    parser.add_argument("model", help="HuggingFace model ID (e.g. mlx-community/Qwen2.5-32B-Instruct-4bit)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--log", dest="log_file", default=None, help="Log file path")
    args = parser.parse_args()
    start_server(args.model, args.host, args.port, log_file=args.log_file)
