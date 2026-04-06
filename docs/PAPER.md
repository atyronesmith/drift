# Drift: Reducing LLM Inference Cost Through Automated Determinism Analysis

## Executive Summary

Enterprise AI applications waste significant compute on work that doesn't require an LLM. When a model processes a support ticket, an error log, or a metrics dashboard, it spends much of its effort copying values that already exist in the input — or classifying into categories that a simple model could predict. Meanwhile, the input itself is often 70-90% noise that the LLM must wade through before it can reason about the signal.

Drift is a tool that identifies and eliminates this waste through two complementary approaches:

**Output-side optimization** — Probe the LLM's own token probabilities to find which output fields are deterministic (extraction) vs. uncertain (reasoning). Compile the deterministic fields into instant rules. Keep the LLM only for genuine reasoning. Result: 39-52% latency reduction, 100% field accuracy on extracted fields.

**Input-side optimization** — Deterministic preprocessing to strip noise, deduplicate repeated patterns, and compress known structures before the input reaches the LLM. Result: 83% input token reduction on a production corpus of 500 Ansible error logs, with quality that never degrades and improves in 53% of cases.

**Validated savings at scale:**

| Scale | GPT-4o Saved/yr | Claude Sonnet Saved/yr |
|-------|----------------|----------------------|
| 10K inputs/month | $5,800 | $7,000 |
| 50K inputs/month | $29,000 | $34,800 |
| 100K inputs/month | $57,900 | $69,500 |

These numbers are from a single pipeline step (log summarization). Applications with multiple LLM calls per unit of work see larger absolute savings.

---

## The Problem

LLM inference is the largest recurring cost in most AI applications. But not all tokens are created equal. Consider an LLM that triages support tickets into structured JSON:

```json
{
  "ticket_id": "1aiu3lrqi",        ← copied from input
  "customer_email": "jane@acme.co", ← copied from input
  "priority": "Medium",             ← copied from input
  "severity": "P3",                 ← deterministic mapping (Medium → P3)
  "root_cause": "network",          ← classification (could be learned)
  "summary": "Printer disconnects intermittently, affecting floor printing" ← reasoning
}
```

Of six fields, only one requires genuine LLM reasoning. The rest are extraction (copying from input), mapping (deterministic transformation), or classification (learnable pattern). Yet the LLM processes the entire input and generates all six fields from scratch on every call.

On the input side, the problem is equally wasteful. An Ansible error log is typically 200-3,000 lines. Of those, 71% are noise (`ok:`, `skipping:`, `changed:` status lines). The LLM must read all of it to find the 5% that contains the actual error. For large logs (100KB+), the model often fails entirely — ignoring instructions and producing rambling narratives instead of the requested one-line summary.

## How Drift Works

### Output-Side: Probe → Synthesize → Compile

Drift analyzes the LLM's own output to find optimization opportunities, then generates code to exploit them. No retraining, no prompt engineering, no hand-written rules.

**Step 1 — Probe.** Run the LLM with logprobs enabled. For each output token, capture the probability distribution over the full vocabulary. Tokens where the model is highly confident (low entropy) are deterministic — the model is copying or applying a fixed rule. Tokens where the model deliberates (high entropy) represent genuine reasoning.

**Step 2 — Synthesize.** For each deterministic field, trace the value back to its source in the input. Generate an extraction rule (regex, literal match, value mapping, or trained classifier). Validate the rule against multiple inputs.

**Step 3 — Compile.** Output a standalone Python engine. Deterministic fields are handled by rules (<1ms). Only uncertain fields are sent to the LLM. Results are merged. The engine runs as a single file with no ML dependencies at runtime.

One command (`drift build`) runs all three steps. The resulting engine handles every subsequent call faster and cheaper.

### Input-Side: Pre-Digestion

For applications with large, noisy inputs (logs, traces, documents), drift applies deterministic preprocessing to reduce what the LLM sees:

- **Level 1** — Strip noise lines (status messages, timestamps, progress indicators) while preserving a window around errors
- **Level 2** — Deduplicate repeated patterns (retry loops, item iterations) into compact descriptors
- **Level 3** — Compress known verbose structures (API responses, stack traces) into one-line summaries

This is domain-specific but automatable. The patterns are learned from corpus analysis, not hand-coded per log.

### Classifier Distillation

For classification fields (category, severity, routing decisions), drift can train a lightweight classifier on LLM-labeled data. Every LLM response is free training data. A logistic regression trained on the LLM's own outputs replaces the LLM for that field — inference in microseconds instead of seconds, with confidence-gated escalation back to the LLM for uncertain cases.

## Experiments and Results

### Experiment 1: Hybrid Extraction on Helpdesk Tickets

**Setup:** 500 IT helpdesk tickets, 12 structured output fields, qwen2.5:32b at temperature=0.

**Finding:** 7 of 12 fields are deterministic extraction (ticket ID, email, name, priority, category, creation date, subject). Drift auto-generates regex rules that extract these with 100% accuracy across all 500 tickets.

**Result:** 39-52% latency reduction. The LLM processes only 5 fields (severity, root cause, assignment group, summary, next steps) instead of 12. Zero accuracy loss.

### Experiment 2: Classifier Distillation for Classification Fields

**Setup:** Logistic regression trained on 400 LLM-labeled tickets. Features: 150-word bag-of-words + structured field one-hots. Two target fields: `root_cause_category` (8 classes) and `assignment_group` (10 classes).

**Finding:** At 0.90 confidence threshold, classifiers handle 25-30% of tickets at 95%+ accuracy. At 0.95 confidence, 100% accuracy with 14-16% coverage. Low-confidence predictions escalate to the LLM automatically.

**Result:** Eliminates LLM inference for high-confidence classification predictions. Pure-Python inference (dot product + softmax) — no sklearn or numpy needed at runtime.

### Experiment 3: Input Pre-Digestion on Ansible Error Logs

**Setup:** 500 production Ansible error logs from an internal log analysis pipeline. Three-level pre-digestion: strip noise, dedup retries, compress known patterns.

**Corpus analysis:**
- 640,000 total lines, 71.4% classified as noise
- Average log: 90-96% line reduction after pre-digestion

**Token savings (validated on full corpus):**

| Metric | Value |
|--------|-------|
| Total raw tokens | ~11.6M |
| Total digest tokens | ~1.96M |
| **Overall savings** | **83.1%** |
| Median per-log | 84.3% |
| P10–P90 range | 80.6%–93.3% |

**Quality validation (50 logs, LLM summaries compared):**

| Metric | Raw | Pre-digested |
|--------|-----|-------------|
| Format compliance | 60% | **100%** |
| LLM failures (timeout/ramble) | **40%** | **0%** |
| Digest better than raw | — | 53% of cases |
| Raw better than digest | — | 0% of cases |

Pre-digestion is not just a cost optimization — it is a quality improvement. On large logs (>50KB), the raw LLM frequently ignores instructions. Pre-digestion eliminates this failure mode entirely.

### Experiment 4: Output-Side on a Real Pipeline

**Setup:** The same Ansible log analysis pipeline has 4 LLM calls per log: Summarize → Classify (8 categories) → Route (binary) → Solution. We applied drift's classifier distillation to the Classify step.

**Finding:** Summarization consumes 92% of total pipeline tokens. Classify and Route together are 5%. Eliminating the Classify LLM call saves only ~1.5% of total pipeline tokens.

**Lesson:** Amdahl's Law applies. When one step dominates the token budget, optimizing smaller steps yields diminishing returns. For this pipeline, input-side pre-digestion (83% savings on the dominant step) vastly outperforms output-side optimization (1.5% savings on a minor step). The right optimization depends on where the tokens are.

### Experiment 5: LLM Determinism Across Infrastructure

**Setup:** Same prompt, temperature=0, measured across infrastructure tiers.

| Infrastructure | Textual Determinism | Functional Equivalence |
|---------------|--------------------|-----------------------|
| Self-hosted single GPU (Ollama) | 100% identical | 100% |
| Cloud API (Vertex AI Claude) | 30% identical | 100% |
| Shared cloud (DeepInfra) | Measurable drift | ~95% |

**Finding:** Textual drift wildly overstates functional drift. Across all tasks, 96.7% of textually different outputs are functionally equivalent. But the remaining 3.3% represent real risk in pipelines where outputs feed downstream systems.

### Experiment 6: RAG Ingestion with Pre-Digestion

**Setup:** Same 500 Ansible logs, chunked at 2048 chars (~512 tokens) with 200-char overlap. Embedded via nomic-embed-text (768-dim). 50 LLM-generated summaries used as retrieval queries.

**Ingestion results:**

| Metric | Raw | Pre-digested | Change |
|--------|-----|-------------|--------|
| Chunks | 25,367 | 4,511 | **82% fewer** |
| Tokens to embed | ~12.9M | ~2.2M | **83% fewer** |
| Embedding time | 27 min | 4 min | **6.8x faster** |

**Retrieval quality (cohesion test):** For 200 sampled chunks, measure what fraction of a chunk's top-5 nearest neighbors come from the same log. Pre-digested cohesion: **0.143** vs raw: **0.073** — nearly 2x better clustering by log identity. Raw chunks are dominated by shared Ansible boilerplate (`ok:`, `changed:`, `skipping:`) that makes every log look the same to the embedding model. Pre-digestion strips this, so chunks embed based on actual error content. Zero noise in digest neighbors vs 20 for raw.

**Finding:** Pre-digestion improves both RAG ingestion cost (82% fewer chunks) and embedding quality (2x better cohesion). It's not just cheaper — it produces more semantically meaningful vectors because the embedding model encodes signal, not noise.

**Generalization (RAGBench):** Validated on three standard retrieval benchmarks (RAGBench) using a domain-agnostic pre-digester (corpus-level boilerplate detection, dedup, separator removal):

| Dataset | Noise Level | Char Reduction | nDCG@5 Delta |
|---------|-------------|----------------|-------------|
| TechQA (IBM tech support) | 28 boilerplate lines | 11.1% | **+0.0211** |
| EManual (electronics) | Clean | 0.1% | 0.0000 |
| CuAD (legal contracts) | Clean | 0.2% | -0.0067 |

Pre-digestion helps where noise exists (TechQA: +3.7% nDCG) and is a harmless no-op on clean data. This makes it a **safe default** in any RAG pipeline — no domain knowledge required.

## What to Target

Drift is not a universal optimization. It works on specific application patterns:

**Strong candidates:**
- Structured extraction (logs → JSON, tickets → triage, invoices → line items)
- Multi-step LLM pipelines (summarize → classify → route → act)
- Large-input processing where most of the input is noise
- Batch processing where the same prompt runs on thousands of inputs
- Classification with known categories (routing, triage, labeling)
- RAG ingestion pipelines (pre-digestion before chunking and embedding)

**Weak candidates:**
- Open-ended chat and Q&A
- RAG query-answering (quality depends on retrieval, not generation)
- Creative generation (the output IS the work)

The test: if the LLM's output can be described as "fill in these fields," drift can help. If the output is "write me a paragraph," it probably can't. For RAG, drift helps the ingestion side (fewer, cleaner chunks) even when the query side is out of scope.

## How to Evaluate

```bash
# Install
pip install -e .

# Probe a task to measure determinism
drift probe experiments/my-task.yaml --rules -o probe.json

# Build a hybrid engine (probe → synthesize → compile)
drift build experiments/my-task.yaml -o engine.py

# Validate the engine against test inputs
drift validate engine.py --input-dir test_inputs/

# For classification fields: distill classifiers from LLM labels
drift distill experiments/my-task.yaml --input-dir inputs/ -o hybrid.yaml
```

The probe step takes one LLM call. The resulting engine handles every subsequent call. Break-even is 2-3 calls — after that, every call saves 39-83% depending on the optimization surface.

## Limitations

- **Domain-specific pre-digestion.** Input-side rules must be built per domain (Ansible logs, Prometheus metrics, etc.). The patterns are learnable from corpus analysis but not fully automatic yet.
- **Training data for classifiers.** Classifier distillation needs 200+ labeled examples for reliable accuracy. The LLM provides the labels, but the labeling step has a one-time cost.
- **Amdahl's Law.** If one pipeline step dominates the token budget, optimizing other steps has limited impact. Always measure where the tokens are before optimizing.
- **Not for reasoning-heavy tasks.** If every output field requires genuine deliberation, there is nothing to extract deterministically. Drift's value comes from the gap between what the LLM does and what it needs to do.

## Status

Drift is working software with validated results on real datasets. The codebase includes 15 documented experiments, standalone engine compilation, and CLI commands for the full probe-synthesize-compile workflow.

Repository: [github.com/atyronesmith/drift](https://github.com/atyronesmith/drift)
