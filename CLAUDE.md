# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Drift measures and reduces LLM output variance in production systems. It provides tools for running experiments, measuring text and functional metrics, probing token entropy via logprobs, synthesizing deterministic rules from probe data, compiling hybrid engines (rules + LLM), and training classifiers via distillation.

## Commands

```bash
# Install for development (uses uv, not pip)
uv pip install -e ".[dev]"

# Run all tests
python -m pytest -v

# Run an experiment
drift run <config.yaml> --runs N

# Measure results
drift measure <results_dir> -o metrics.json

# Full hybrid pipeline
drift build <config.yaml> -o engine.py

# Compare two experiments
drift compare <path_a> <path_b>
```

## Architecture

### Core Pipeline
- `src/drift/cli.py` — CLI entry point (click-based). All commands defined here.
- `src/drift/pipeline/pipeline.py` — Experiment runner: baseline, pipeline, hybrid types.
- `src/drift/providers/` — LLM providers: `ollama.py`, `deepinfra.py`, `vertex.py`, `mlx.py`. Factory in `factory.py`.
- `src/drift/storage.py` — Run directory management, result serialization.
- `src/drift/tasks.py` — Task prompt resolution from `tasks/<name>/` directories.

### Metrics
- `src/drift/metrics/text.py` — Edit distance, BLEU, exact match.
- `src/drift/metrics/functional.py` — Task-specific functional test execution.
- `src/drift/metrics/stats.py` — Bootstrap CIs, permutation tests.
- `src/drift/metrics/iteration.py` — Iteration-level metrics for loop pipelines.

### Probing & Synthesis
- `src/drift/probing/logprobs.py` — Token-level logprob extraction via provider APIs.
- `src/drift/probing/synthesizer.py` — Rule synthesis from probe data.
- `src/drift/hybrid/autogen.py` — Automated rule generation: identifies low-entropy fields, traces to input, generates regex.
- `src/drift/hybrid/compiler.py` — Compiles hybrid YAML specs into standalone Python engines.
- `src/drift/hybrid/rules.py` — Rule types: regex, literal, regex_all, keyword, classifier.
- `src/drift/hybrid/executor.py` — Runs rules, builds focused LLM prompt, merges output.
- `src/drift/hybrid/calibration.py` — Baseline confidence distributions for classifier drift detection.

### Safety
- `src/drift/safety/guard.py` — `GuardModel` (Llama Guard on Ollama via curl), `TieredGuard` (classifier-first, escalate to LLM guard).
- `src/drift/safety/classifier.py` — Tiered safe/unsafe classification: keyword scan (Tier 1) + logistic regression (Tier 2) + escalation (Tier 3). Moved here from denoise.
- `src/drift/safety/monitor.py` — JSD-based classifier drift detection. `ClassifierMonitor` tracks confidence distributions at runtime. Moved here from denoise.
- `src/drift/safety/models.py` — Security model definitions and overhead projection.
- `scripts/train_guard_classifier.py` — Trains the Tier 2 logistic regression, exports weights to `results/guard_classifier/`.
- `scripts/safety_audit.py` — Batch audit of saved results against guard models.

### Tasks
Each task lives in `tasks/<name>/` with `input.md`, prompt files, and `functional_tests.py`. Tasks: code_review, data_extraction, iterative_refactor, incident_triage, config_generation, promql_generation, metric_summary, alert_triage, pod_log_triage, ansible_classify, ansible_route, ansible_summarize, ansible_triage.

## Documentation

- `docs/PAPER.md` — Technical paper for review/outreach
- `docs/RESULTS.md` — 19 experiments documented
- `docs/HYBRID.md` — Automated rule extraction pipeline
- `docs/THEORY.md` — Voting vs deliberation, distillation viability thresholds
- `docs/PROPOSAL.md` — Management proposal with dollar-cost analysis
- `docs/ai-safety-overhead.md` — Safety infrastructure compute costs
- `docs/GUARD.md` — Guard module: tiered classification, FPR findings
- `docs/CONTINUOUS_CALIBRATION.md` — Runtime JSD monitoring, auto-retrain workflow
- `docs/LLAMA_GUARD_SE.md` — Standalone finding: Llama Guard 3-1B 47.8% FPR on SE content
- `docs/CONDITIONAL_CLASSIFIERS.md` — Conditional classifiers + continuous calibration plan

## Providers

- **Ollama** — self-hosted at 192.168.15.10:11434 (not localhost). Uses curl subprocess.
- **DeepInfra** — OpenAI-compatible API, key in .env.
- **Vertex AI** — Claude Sonnet via Google Cloud, uses anthropic SDK directly.
- Ollama/DeepInfra use `subprocess` + `curl` — endpoint security blocks Python's direct TCP sockets.

## Relationship to Denoise

Denoise (sibling repo at `../denoise`) is the runtime pre-digestion library. Drift trains, experiments, and validates.

- Drift experiments validate denoise's pre-digestion effectiveness
- `drift discover` (planned) will generate pattern sets for denoise from log corpora
- Guard classification and drift monitoring live here in `src/drift/safety/`, not in denoise
- Classifier weights are trained here (`scripts/train_guard_classifier.py`) and loaded by `drift.safety.classifier`
