# Drift: Measuring and Reducing LLM Output Variance in Production Systems

## Abstract

We present Drift, a measurement and optimization framework for LLM output variance in extraction-heavy production applications. Through 17 experiments across structured extraction, multi-step pipelines, and RAG ingestion, we establish three key findings: (1) edit distance is a poor proxy for functional reliability — outputs with 47-73% edit distance are 96.7% functionally equivalent on tasks with verifiable structure; (2) variance compounds through pipeline depth in free-form pipelines but can be partially absorbed by structured output constraints; (3) automated hybrid execution (deterministic rules for extractable fields, LLM for reasoning) reduces latency by 39-52% on extraction tasks, and input-side pre-digestion reduces tokens by 83% while improving quality.

Drift provides a complete CLI toolkit: `probe` identifies deterministic fields via logprob entropy analysis, `synthesize` generates extraction rules, `compile` emits standalone Python engines, and `build` runs the full pipeline in one command. Rules generated from a single input generalize at 100% across 200 inputs sharing the same format (within-format generalization, not cross-format).

---

## 1. The Problem

LLM inference is the largest recurring cost in most AI applications. But not all tokens are created equal. Consider an LLM that triages support tickets into structured JSON:

```json
{
  "ticket_id": "1aiu3lrqi",        // copied from input
  "customer_email": "jane@acme.co", // copied from input
  "priority": "Medium",             // copied from input
  "severity": "P3",                 // deterministic mapping (Medium -> P3)
  "root_cause": "network",          // classification (could be learned)
  "summary": "Printer disconnects intermittently..." // genuine reasoning
}
```

Of six fields, only one requires genuine LLM reasoning. The rest are extraction (copying from input), mapping (deterministic transformation), or classification (learnable pattern). Yet the LLM processes the entire input and generates all six fields from scratch on every call.

On the input side, the problem is equally wasteful. An Ansible error log is typically 200-3,000 lines. Of those, 71% are noise (`ok:`, `skipping:`, `changed:` status lines). The LLM must read all of it to find the 5% that contains the actual error. For large logs (100KB+), the model often fails entirely — ignoring instructions and producing rambling narratives instead of the requested one-line summary.

Beyond cost, there is a correctness problem: LLMs do not produce identical outputs across calls, even at temperature=0. This variance is usually cosmetic, but in multi-step pipelines where outputs feed downstream systems, it can compound and occasionally cross from cosmetic to functional — we observed a real, silent data-corruption bug in 1 of 30 pipeline variants.

## 2. Key Findings

### 2.1 Textual Drift Does Not Equal Functional Drift

The central finding across 17 experiments: **edit distance is the wrong metric for evaluating LLM output reliability.** Outputs that differ by 47-73% in edit distance are overwhelmingly functionally equivalent — they fix the same bugs, extract the same values, and pass the same tests. This is not a claim that LLM variance is harmless; it is a claim that text-level metrics dramatically overstate functional risk.

We define **functional equivalence** as: all variants pass an identical test suite covering correctness, security, and edge cases. For code_review, this is 10 tests against an in-memory SQLite database (parameterized queries, SQL injection blocked, no password exposure, batch queries, empty/missing input handling). For data_extraction, 12 tests covering all line items, arithmetic, and entity extraction. For iterative_refactor, 10 tests covering 4 security fixes plus correct behavior. Each variant is compiled and executed — not compared textually.

| Task | Textual Exact Match | Functional Equivalence | N |
|------|--------------------|-----------------------|---|
| code_review (pipeline, 30 variants) | 33% | **96.7%** | 30 (10 per pipeline trial) |
| data_extraction (30 runs, same input) | 83% | **100%** | 30 |
| iterative_refactor (8 runs) | 60% | **100%** | 8 |

All tasks tested are extraction or transformation-heavy. The gap between textual and functional metrics would be smaller for open-ended generation tasks where the output *is* the deliverable. Drift's applicability is strongest where outputs have verifiable structure.

But "mostly equivalent" is the dangerous word. In the code review pipeline, 1 of 30 variants contained a real bug — a duplicate dictionary key (`"id"` written twice, overwriting with the name field) that silently corrupted return data. The bug originated in the mixed pipeline (Vertex reviewer -> Ollama implementer): the cloud model's textual variance at step 1 produced a review phrasing that led the deterministic step-2 model to generate a typo in the dictionary comprehension. The bug was undetectable by text-level metrics — it passed all security tests and required data-correctness execution to catch. Whether bug probability increases with pipeline depth is unknown from this sample size.

### 2.2 Variance Compounds Through Pipeline Depth

In a 3-step pipeline (reviewer -> implementer -> verifier), textual variance increases monotonically when outputs are free-form:

| Step | Free-form ED | Structured ED |
|------|-------------|---------------|
| 1 (reviewer) | 0.47 | 0.27 |
| 2 (implementer) | 0.58 | 0.52 |
| 3 (verifier) | 0.73 | **0.34** |

Free-form pipelines show compounding drift: 0.47 -> 0.58 -> 0.73. Each step amplifies upstream variance without absorbing it. Deterministic models don't help — Ollama (100% deterministic on fixed input) produced 10/10 unique outputs when fed varying upstream output.

Structured output (JSON schema) is the only intervention that bends the curve downward. At step 3, the JSON constraint drops edit distance from 0.73 back to 0.34 — the only observed case where a downstream step reduced rather than amplified variance.

**Per-field analysis confirms this is not just boilerplate dilution.** A natural objection is that JSON keys and brackets are fixed text that mechanically lowers the *percentage* edit distance without reducing actual reasoning variance. We decomposed the structured output into structural fields (severity, category, line number) and reasoning fields (description, fix, summary) and computed edit distance on each independently:

| Metric | Reviewer | Verifier |
|--------|----------|----------|
| Free-form aggregate ED | 0.487 | 0.733 |
| Structured aggregate ED | 0.270 | 0.372 |
| Structured reasoning fields ED | 0.418 | 0.453 |
| Structured structural fields ED | 0.250 | 0.000 |

The reasoning fields in structured output drift **less** than the aggregate free-form output — 14% less at the reviewer step, 38% less at the verifier. However, this comparison has a methodological caveat: the free-form baseline measures *all* content (reasoning, examples, formatting, section headers) while the structured reasoning fields measure *only* the variable JSON values. Free-form output cannot be cleanly decomposed into "reasoning" and "structural" components — which is arguably the point: structured output *enables* per-field measurement that free-form does not. The fairer conclusion is that structured output constrains reasoning variance to a measurable degree, but part of the aggregate ED reduction is also dilution from fixed JSON boilerplate.

The per-field breakdown also reveals a priority gradient in determinism. The top-priority issues (SQL injection at issues[0], password exposure at issues[1]) are perfectly deterministic: ED=0.0 across all 10 runs for every field. Lower-priority issues (3+) show high variance (ED=0.54-0.72 on descriptions). The model agrees completely on what matters most and differs only on secondary findings.

### 2.3 Infrastructure Determines the Drift Floor

Same prompt, temperature=0, measured across three infrastructure tiers:

| Infrastructure | Model | Exact Match | Functional Eq | Latency |
|---------------|-------|-------------|---------------|---------|
| Single-GPU self-hosted (Ollama) | Qwen2.5-32B | **100%** | 100% | ~173s |
| Shared cloud (DeepInfra) | Qwen2.5-72B | 90% | 100% | ~55s |
| Cloud API (Vertex AI) | Claude Sonnet | 30% | 100% | ~20s |

**Caveat:** This comparison confounds infrastructure with model. Only the Ollama-to-DeepInfra pair isolates infrastructure effects (same model family, different serving). The Vertex result reflects both cloud infrastructure non-determinism *and* a different model architecture. The clean finding is: single-GPU Ollama is perfectly deterministic at temp=0, while DeepInfra's shared serving introduces measurable (but functionally harmless) drift on the same model family. Cloud APIs with different models show the most textual variation but maintain functional equivalence on structured extraction.

### 2.4 Temperature 0 to 0.1 Is the Cliff

On a structured extraction task (data_extraction, qwen2.5:32b):

| Temp | Exact Match | Functional Eq |
|------|-------------|---------------|
| 0.0 | **100%** | 100% |
| 0.1 | 20% | 100% |
| 0.3 | 10% | 90% |
| 0.5 | 10% | 80% |
| 1.0 | 10% | 40% |

Textual determinism collapses immediately at temp=0.1 (100% -> 20% exact match), but functional equivalence holds. Functional degradation is linear: 100% at 0, 90% at 0.3, 40% at 1.0. At temp=1.0, the model hallucinates values instead of extracting them.

For production: always use temperature=0 when determinism matters. If temp>0 is required, functional testing is mandatory.

### 2.5 Model Upgrades Are a Drift Vector

Upgrading from Qwen2.5:32b to Qwen3:30b on the same hardware introduced both textual drift and functional regression:

| Task | Qwen2.5 Functional | Qwen3 Functional |
|------|--------------------|--------------------|
| data_extraction | 100% | 100% (cosmetic drift only) |
| iterative_refactor | **100%** | **0%** (password exposure missed) |

Qwen3 consistently misses a password exposure fix that Qwen2.5 catches in every run. This is a model capability regression, not drift — it's consistent across runs. Disabling Qwen3's thinking mode (`/nothink`) partially recovers (80% functional), but the first run after model load still fails (cold-model effect).

Model version upgrades are not safe without regression testing. A task that scored 100% on one version can drop to 0% on the next.

### 2.6 Different Models Make Different Architectural Decisions

Claude Sonnet and Qwen2.5 produce functionally different solutions to the same task — not through drift, but through model personality:

- **Claude** imports bcrypt and redesigns the auth system (incompatible with test DB's plaintext passwords) — 0% functional equivalence, but arguably better security
- **Qwen2.5** makes minimal conservative fixes, preserving API compatibility — 100% functional equivalence

Claude also converges in 1 iteration on iterative refactoring tasks where Qwen2.5 takes 2-3, and runs 9x faster on extraction (19s vs 173s per run via cloud API vs self-hosted).

## 3. How Drift Works

### 3.1 Output-Side: Probe -> Synthesize -> Compile

Drift analyzes the LLM's own output to find optimization opportunities, then generates code to exploit them. No retraining, no prompt engineering, no hand-written rules.

**Step 1 — Probe.** Run the LLM with logprobs enabled. For each output token, capture the probability distribution over the full vocabulary. Tokens where the model is highly confident (low entropy) are deterministic — the model is copying or applying a fixed rule. Tokens where the model deliberates (high entropy) represent genuine reasoning.

**Step 2 — Synthesize.** For each deterministic field, trace the value back to its source in the input. Generate an extraction rule (regex, literal match, value mapping, or trained classifier). Validate the rule against the input to catch overfitting.

**Step 3 — Compile.** Output a standalone Python engine. Deterministic fields are handled by rules (<1ms). Only uncertain fields are sent to the LLM. Results are merged. The engine runs as a single file with no ML dependencies at runtime.

One command (`drift build`) runs all three steps. The resulting engine handles every subsequent call faster and cheaper.

### 3.2 Input-Side: Pre-Digestion

> **Implementation:** The pre-digestion library is the [logdigest](https://github.com/atyronesmith/logdigest) sibling repo — see logdigest's `docs/PREDIGESTION.md` for theory and results.

For applications with large, noisy inputs (logs, traces, documents), deterministic preprocessing reduces what the LLM sees:

- **Level 1** — Strip noise lines (status messages, timestamps, progress indicators) while preserving a window around errors
- **Level 2** — Deduplicate repeated patterns (retry loops, item iterations) into compact descriptors
- **Level 3** — Compress known verbose structures (API responses, stack traces) into one-line summaries

This is domain-specific. Patterns can be hand-authored or automatically discovered from a log corpus via `logdigest discover` (Drain3 template mining + frequency-based classification). See logdigest's `docs/PATTERN_GENERATION.md` for the pipeline design.

### 3.3 Classifier Distillation

For classification fields (category, severity, routing decisions), drift can train a lightweight classifier on LLM-labeled data. Every LLM response is free training data. A logistic regression trained on the LLM's own outputs replaces the LLM for that field — inference in microseconds instead of seconds, with confidence-gated escalation back to the LLM for uncertain cases.

### 3.4 Rule Tier Hierarchy

Rules are organized into tiers by complexity. Each tier is <1ms except LLM:

| Tier | Type | Example | Discovery |
|------|------|---------|-----------|
| 1 | Regex/literal | `Ticket ID:\s*(\S+)` | Single-probe entropy analysis |
| 2 | Mapping/conditional | `Medium -> P3` | Multi-probe cross-field correlation |
| 3 | Classifier | Logistic regression on BoW features | LLM-labeled training data (200+ samples) |
| 4 | LLM | Summarization, complex reasoning | Escalation target for tiers 1-3 |

When a tier fails to match (regex doesn't match, classifier confidence below threshold), the field escalates to the next tier. The LLM is the final fallback.

## 4. Validation at Scale

### 4.1 Rule Generalization (200 Tickets)

Rules generated from probing a single helpdesk ticket were tested against 200 different tickets (100 train / 100 test):

| Rule | Type | Train (100) | Test (100) |
|------|------|-------------|------------|
| ticket.id | regex | 100% | 100% |
| ticket.subject | regex | 100% | 100% |
| ticket.requester_email | regex | 100% | 100% |
| ticket.created_at | regex | 100% | 100% |
| ticket.stated_priority | regex | 100% | 100% |
| ticket.stated_category | regex | 100% | 100% |

**Train-test gap: 0.0%.** All 6 validated rules match 100% of tickets in both sets. The rules generalize because they anchor on input format structure (`Ticket ID:`, `Subject:`, `From:`) rather than content — they generalize across inputs within the same format, not across formats. If the input template changes (fields reordered, labels renamed), the rules break and must be re-probed. Re-probing takes one LLM call. One overfit rule (anchored on ticket-specific text) was caught at synthesis time by validation.

### 4.2 Classifier Calibration (400 Tickets)

Logistic regression trained on 400 LLM-labeled tickets for two classification fields:

| Field | Threshold | Coverage | Accuracy |
|-------|-----------|----------|----------|
| root_cause_category | >= 0.90 | 25% | **95.0%** |
| root_cause_category | >= 0.95 | 14% | **100%** |
| assignment_group | >= 0.90 | 30% | **95.8%** |
| assignment_group | >= 0.95 | 16% | **100%** |

The confidence gate works: high-confidence predictions are near-perfect, low-confidence escalates to the LLM with zero accuracy loss. Pure-Python inference (dot product + softmax) matches sklearn predictions exactly — no ML dependencies at runtime.

### 4.3 Hybrid Latency (10 Tickets)

| Approach | Mean Latency | Field Accuracy |
|----------|-------------|----------------|
| Pure baseline (LLM handles all 12 fields) | 21.1s (95% CI: 18.8-23.1s) | 100% |
| Auto hybrid (6 rules + LLM for 6 fields) | **12.8s** (95% CI: 11.5-14.3s) | 100% |
| **Speedup** | **39%** (range: 33-51%) | — |

The speedup mechanism is straightforward: the hybrid prompt is shorter (requesting only 6 assessment fields instead of 12) and produces fewer output tokens. The 6 extraction rules execute in microseconds. N=10 is small for absolute latency claims on cloud infrastructure, but the speedup is mechanistic (fewer tokens in, fewer tokens out) rather than statistical — it should hold at any N. The range across tickets (33-51%) reflects variation in ticket complexity, not measurement noise.

### 4.4 Input Pre-Digestion (500 Ansible Logs)

| Metric | Value |
|--------|-------|
| Total raw tokens | ~11.6M |
| Total digest tokens | ~1.96M |
| **Token savings** | **83.1%** |
| Median per-log savings | 84.3% |
| P10-P90 range | 80.6%-93.3% |

Quality validation (50 logs with LLM summaries):

| Metric | Raw | Pre-digested |
|--------|-----|-------------|
| Format compliance | 60% | **100%** |
| LLM failures (timeout/ramble) | **40%** | **0%** |
| Digest better than raw | -- | **53%** of cases |
| Raw better than digest | -- | **0%** of cases |

Format compliance and failure rate are objective metrics (did it produce a single line? did it timeout?). The "better/worse" comparison is LLM-judged (same model comparing its own outputs), which creates a circularity risk — the model may prefer pre-digested input because pre-digestion was designed for its architecture. The objective metrics alone justify the approach: eliminating the 40% raw failure rate on large logs is the primary quality gain.

### 4.5 RAG Ingestion (500 Logs + 3 RAGBench Datasets)

Pre-digestion before RAG chunking and embedding:

| Metric | Raw | Pre-digested | Change |
|--------|-----|-------------|--------|
| Chunks | 25,367 | 4,511 | **82% fewer** |
| Embedding time | 27 min | 4 min | **6.8x faster** |
| Same-log cohesion | 0.073 | 0.143 | **2x better** |

Pre-digested embeddings are more semantically meaningful because the model encodes error content, not shared Ansible boilerplate.

Validated on three RAGBench datasets (domain-agnostic pre-digestion):

| Dataset | Noise Level | Char Reduction | nDCG@5 Delta |
|---------|-------------|----------------|-------------|
| TechQA (IBM tech support) | 28 boilerplate lines | 11.1% | **+0.0211** |
| EManual (electronics) | Clean | 0.1% | 0.0000 |
| CuAD (legal contracts) | Clean | 0.2% | -0.0067 |

Pre-digestion helps where noise exists and is a harmless no-op on clean data — a safe default for any RAG pipeline.

## 5. Pipeline Hybrid: A Negative Result

We attempted to extend hybrid execution to individual pipeline steps, anchoring inter-step communication with deterministic rules. The initial results were striking: 77% drift reduction, with the implementer step achieving 0.0 edit distance despite having no hybrid rules of its own — the cascade effect from anchoring the reviewer's output.

But functional testing revealed a fundamental limitation: **literal rules on array-indexed paths constrain the output array length**. The probe discovered 2 issues, so the rules locked `issues[0]` and `issues[1]`. The LLM, told these fields were "already extracted," only filled in the remaining fields for those 2 elements — never discovering additional issues. The structured reviewer naturally finds 5-6 issues, including the performance bug that triggers batch query fixes.

Removing the offending rules and keeping only verifier-side hybrid produced a null result — the verifier's anchored fields (`status=pass`, `remaining_issues=[]`) were already consistent in structured output.

**The cascade effect is real but double-edged.** Anchoring upstream steps eliminates downstream drift, but if the anchor is incomplete, it eliminates correct behavior too. Pipeline hybrid execution needs schema-level rules (constraining structure without constraining content) rather than literal value rules. This remains open work.

## 6. Residual Model Experiments

After rules extract deterministic fields, can a smaller model handle the remaining reasoning fields?

| Model | Speedup | Classification Agreement |
|-------|---------|--------------------------|
| qwen2.5:32b (reference) | -- | -- |
| llama3.1:8b | **72%** | 50% |
| qwen3:4b | -341% (slower) | 70% |

The 8B model is genuinely fast but disagrees on classification. Adding an enum constraint (listing valid categories in the prompt) improved agreement from 10% to 50% — a cheap, high-ROI intervention that benefits all model sizes, including the reference 32B model which was itself inconsistent without it.

The remaining 50% disagreement represents genuine reasoning differences (e.g., "Access Management" vs "Account Management" for account setup tickets) — both assignments are defensible. This is the ceiling for prompt engineering on classification.

Attention weight analysis (MLX, 50 tickets) confirmed that the model doesn't classify by spotting keywords — it synthesizes meaning across the full ticket. Per-word attention weights are tiny (0.001-0.010), too diffuse for keyword extraction. Classification genuinely requires holistic reasoning, which is why keyword rules can't match LLM performance but logistic regression (which sees the full feature vector) can.

## 7. Cost Projections

**Validated savings at scale** (combining input pre-digestion + output hybrid optimization):

| Scale | GPT-4o Saved/yr | Claude Sonnet Saved/yr |
|-------|----------------|----------------------|
| 10K inputs/month | $5,800 | $7,000 |
| 50K inputs/month | $29,000 | $34,800 |
| 100K inputs/month | $57,900 | $69,500 |

These numbers are from a single pipeline (log summarization). Applications with multiple LLM calls per unit of work see larger absolute savings. Break-even is 2-3 calls after engine compilation.

**Amdahl's Law applies.** When one pipeline step (summarization) consumes 92% of the token budget, optimizing smaller steps (classify, route) yields diminishing returns. Always measure where the tokens are before optimizing. For the Ansible pipeline, input-side pre-digestion (83% savings on the dominant step) vastly outperforms output-side optimization (1.5% savings on a minor step).

## 8. What to Target

**Strong candidates:**
- Structured extraction (logs -> JSON, tickets -> triage, invoices -> line items)
- Multi-step LLM pipelines (summarize -> classify -> route -> act)
- Large-input processing where most of the input is noise
- Batch processing where the same prompt runs on thousands of inputs
- Classification with known categories (routing, triage, labeling)
- RAG ingestion pipelines (pre-digestion before chunking and embedding)

**Weak candidates:**
- Open-ended chat and Q&A
- RAG query-answering (quality depends on retrieval, not generation)
- Creative generation (the output IS the work)
- Open-ended review tasks where the output array length varies per input

The test: if the LLM's output can be described as "fill in these fields," drift can help. If the output is "write me a paragraph," it probably can't.

**A note on evaluation scope:** All experiments in this paper use extraction-heavy tasks (helpdesk triage, invoice extraction, code review with structured output, log summarization). These tasks were chosen because they represent the highest-value targets for drift optimization — not because drift's findings generalize to all LLM applications. The textual-vs-functional gap would be smaller for open-ended generation where there is no verifiable "correct" structure.

## 9. Related Work

### 9.1 Inspiration: The Busy Mom Syndrome

This work was directly inspired by Nar, Shaw, and Ibrahim's ["The Busy Mom Syndrome"](https://medium.com/enterpriseai/the-busy-mom-syndrome-e8a8829ae3db), which identifies a critical but under-studied problem: LLM output quality degrades during peak usage periods due to infrastructure resource contention, not model changes. Their central metaphor — a mother with one child is patient and thorough, but with five kids all demanding attention simultaneously, the quality of each response drops even though the intelligence hasn't changed — frames the infrastructure determinism problem clearly.

**Where we agree:**
- Cloud inference is non-deterministic even at temperature=0. Their analysis explains *why* (multi-tenant GPU sharing, KV cache eviction under memory pressure, model substitution by providers) while our experiments measure *how much* (100% exact match on single-GPU -> 90% on shared cloud -> 30% on cloud API).
- Our "cold-model first-run" pattern (Qwen3 producing different output on the first run after model load) maps directly to their KV cache warming discussion — the first request has no cached prefix and may behave differently.
- Infrastructure-level non-determinism is invisible to the application. Their finding that providers silently route to cheaper sub-models under load (documented in GPT-5's system card and Azure AI Foundry's "balanced mode") explains variance that applications cannot distinguish from model stochasticity.

**Where we diverge:**
- **Cause vs. mitigation.** Busy Mom diagnoses the infrastructure causes of quality degradation. Drift measures the application-level impact and builds mitigations — deterministic rules that bypass the LLM entirely for extractable fields, removing infrastructure variance from the equation for those fields.
- **Load-dependent vs. load-independent variance.** Busy Mom focuses on degradation under concurrent load (P99 latency tripling from 1 to 5 concurrent requests, coherence failure by turn 9-10 under cache eviction). Our experiments measure variance even under sequential, no-load conditions — single requests to an idle server still show textual drift on cloud APIs. Infrastructure variance exists at all load levels; it gets worse under contention.
- **Solution layer.** Their recommendations are infrastructure-side: cache-aware routing (87% hit rate vs 30-40% random), capacity planning formulas, off-peak scheduling. Our recommendations are application-side: extract what you can deterministically, send only genuine reasoning to the LLM. The approaches are complementary — applying both would yield the most deterministic system.
- **Input-side optimization.** Our pre-digestion (83% token reduction on Ansible logs) is an application-layer version of their recommendation to "break long contexts into modular, independently coherent chunks to reduce KV footprint." We do it deliberately before the request; without pre-digestion, the infrastructure may do it silently via context truncation or token-level KV cache compression — with unpredictable quality loss.

Vergadia's [visualization of NVIDIA's full AI stack](https://priyankavergadia.substack.com/p/nvidia-ai-stack-explained) — from Vera Rubin silicon through CMX context storage, Spectrum-X photonics, and Dynamo/TensorRT software — maps the hardware layers where these bottlenecks live. NVIDIA's CMX platform (dedicated flash tier for KV blocks, BlueField-4 DPU for cache management) is a direct hardware response to the KV cache eviction problem that Busy Mom identifies as a root cause of quality degradation. Their $20B Groq acquisition targets sequential token generation latency — the exact bottleneck that makes reasoning fields expensive. Drift's answer to the same problem operates above all five of these layers: even with perfect silicon, memory, networking, orchestration, and middleware, application-layer non-determinism persists from model stochasticity and prompt sensitivity.

The three approaches are complementary: NVIDIA optimizes the hardware platform to reduce infrastructure-induced variance, Busy Mom diagnoses the operational causes and recommends infrastructure-side mitigations, and Drift reduces the application's dependence on inference being consistent in the first place.

### 9.2 Multi-Agent Reliability

Our findings also align with recent research on multi-agent system reliability:

**Error amplification.** Xie et al. (2026) describe cascade amplification theoretically; our drift curves (0.47 -> 0.73 in free-form pipelines) provide empirical measurement. Kim et al. (2025) report up to 17.2x error amplification in unstructured agent networks. Our 3-step linear pipeline is the most vulnerable topology in both studies.

**Topology and resilience.** Huang et al. (2025) find linear pipelines are the least resilient topology (23.7% performance drop from faulty agents) compared to hierarchical (5.5%). Our structured output "sandwich pattern" is a lightweight alternative to full hierarchical governance.

**Variance as feature.** The Council of AIs study (Shaikh et al., 2025) shows response variability is a strength when harnessed through deliberation — five GPT-4 instances at temp=1 corrected errors 53% of the time through consensus. Our temperature=0 experiments on cloud infrastructure show that enough variance exists naturally for non-trivial diversity without deliberately sampling.

**Spectral risk.** Xie et al.'s formal model predicts systems become supercritical when adoption rate x spectral radius exceeds decay rate. Structured output increases the decay rate by constraining the output space — this is why the drift curve bends down.

### 9.3 The Safety Tax

Our companion analysis ([ai-safety-overhead.md](ai-safety-overhead.md)) surveys the compute cost of production safety infrastructure: guardrails (~0.5s per call), safety judge LLMs (~0.2s per call), constrained decoding (5-60% latency increase), and deterministic inference (34% throughput reduction). The key finding: stacked safety layers can reject up to 57% of legitimate outputs (LlamaFirewall benchmark), creating a retry death spiral where the effective compute cost per useful output more than doubles.

Drift's hybrid approach offers a complementary angle on this problem. Fields extracted by deterministic rules bypass not only the productive LLM but also its safety stack — a regex extracting `Ticket ID: 1aiu3lrqi` needs no guardrail check, no content moderation, and no safety judge. For the 6 of 12 fields handled by rules in our helpdesk experiment, the safety overhead is zero. The LLM safety stack only applies to the remaining 6 reasoning fields. This effectively halves the safety tax surface for this task, in addition to the 39% latency reduction from shorter prompts.

## 10. Limitations

- **Domain-specific pre-digestion.** Input-side rules must be built per domain (Ansible logs, Prometheus metrics, etc.). Corpus-level boilerplate detection is domain-agnostic, but deep pre-digestion (retry dedup, blob compression) requires domain patterns.
- **Training data for classifiers.** Classifier distillation needs 200+ labeled examples. The LLM provides the labels, but the labeling step has a one-time cost.
- **Amdahl's Law.** If one pipeline step dominates the token budget, optimizing other steps has limited impact. Always measure where the tokens are.
- **Array-constrained output.** Literal rules on array-indexed paths constrain output length, making hybrid execution unsafe for open-ended review tasks. Schema-level rules (structure without content constraints) are needed but not yet implemented.
- **Not for reasoning-heavy tasks.** If every output field requires genuine deliberation, there is nothing to extract deterministically. Drift's value comes from the gap between what the LLM does and what it needs to do.
- **Hallucinated confidence.** Drift trusts the LLM as oracle: low entropy = deterministic = safe to extract. If the model is confidently wrong (low logprob entropy but factually incorrect), `synthesize` will lock in the hallucination as a rule. For regex rules, this is partially mitigated — the pattern must match a value that exists in the input, so fabricated values fail validation. For literal and mapping rules, consistent hallucinations across probes will be encoded as rules. `drift validate` against labeled test data catches this, but requires ground truth — partially undermining the "zero human effort" claim.
- **Format-dependent rules.** Generated rules anchor on input format structure (field labels, delimiters). They generalize perfectly within a format but break when the format changes (software update, different input channel). Re-probing is a single command but requires detection that the format has changed.

## 11. Usage

```bash
# Install
pip install -e .

# Probe a task to measure determinism
drift probe experiments/my-task.yaml --rules -o probe.json

# Build a hybrid engine (probe -> synthesize -> compile)
drift build experiments/my-task.yaml -o engine.py

# Validate the engine against test inputs
drift validate engine.py --input-dir test_inputs/

# Distill classifiers from LLM labels
drift distill experiments/my-task.yaml --input-dir inputs/ -o hybrid.yaml

# Iterative refinement
drift refine engine.py --input-dir inputs/ --config experiments/my-task.yaml

# Pipeline probing
drift probe-pipeline experiments/pipeline.yaml -o results/probes/
```

## 12. Status

Drift is working software with validated results on real datasets. The codebase includes 17 documented experiments, standalone engine compilation, and CLI commands for the full probe-synthesize-compile workflow.

Repository: [github.com/atyronesmith/drift](https://github.com/atyronesmith/drift)
