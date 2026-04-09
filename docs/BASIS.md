# Inference Drift: Measuring Infrastructure-Induced Variability in Iterative Multi-Agent Systems

---

## 1. The Busy Mom Syndrome: Foundation

The observation is simple: same model, same prompt, same context — different quality outcomes depending on when you run it. Nar, Shaw et al. named it the Busy Mom Syndrome: a mother with one child asking a question is patient, thorough, coherent. That same mother with five kids all demanding attention simultaneously still has the same intelligence, but the quality of each individual answer degrades — not because she became less capable, but because resource contention became real.

The model weights did not change. The platform resources they run on did.

This is not a perception problem. It is a physics problem.

---

## 2. Prior Work: Model-Level Variance in Multi-Agent Financial Risk Assessment

This project is directly motivated by empirical results from a DTCC/FINOS financial services hackathon (2025), where a cross-company team built a multi-agent system for AI governance — evaluating the risk posture of new AI software deployments against regulatory frameworks (SR 11-7, EU AI Act, OWASP, NIST, FFIEC).

### The System

The pipeline took three inputs — a use case description, system architecture, and evaluation criteria — and ran them through an agent orchestration layer (Fluxnova) backed by LLM inference (Amazon Bedrock), with deterministic guardrails via FINOS Morphir (policy as code), CALM (architecture as code), and the FINOS AI Governance Framework risk catalog. Outputs included prioritized risks, recommended mitigations, regulatory compliance assessments, and architecture references.

The deterministic guardrails — Morphir policy checks, CALM schema validation, AIGF risk catalog lookups — were stable. The stochastic layer (LLM inference) was where variance appeared.

### The Finding

Two configurations (Config A and Config B) of the same pipeline, differing in model and prompt structure, were evaluated against a set of test cases with known planted risks:

| Metric | Config A | Config B |
|--------|----------|----------|
| Overall risk detection rate | 32% | 78% |
| Worst-case per-case detection | 0% | — |
| Best-case per-case detection | — | 100% |
| False positive accuracy | 0/3 | 0/3 |

Individual case results swung dramatically: SentinelAI detection went from 0% (Config A) to 80% (Config B). ClaimScan went from 0% to 80%. Some tools refused to produce output entirely (VulnScanner) or returned nothing (FairLens).

### The Connection

The hackathon demonstrated **model-level variance**: changing the model or prompt configuration produced wildly different output quality from the same pipeline. Detection rates ranged from 0% to 100% on the same test cases.

This project extends that finding in a more subtle and arguably more dangerous direction: **holding the model constant**, does infrastructure variance alone produce meaningful output divergence? The hackathon result is the louder signal — you changed the model, of course results changed. The drift thesis is quieter: you *didn't* change anything, and results still changed, because the platform did.

The hackathon also demonstrated a critical structural point: deterministic guardrails (policy-as-code, schema validation, risk catalogs) can only constrain what the stochastic layer produces. If the LLM's output quality degrades due to infrastructure conditions, the guardrails still apply — but they're evaluating degraded input. The governance layer doesn't fail; it faithfully governs a worse answer.

For regulated industries like financial services, where the hackathon system would need to produce consistent, auditable risk assessments, infrastructure-induced variance is not a performance problem — it is a compliance problem. The same deployment, evaluated on Tuesday afternoon vs. Sunday morning, could produce materially different risk findings from the same inputs.

---

## 3. The Engineering Reality: Why Inference Quality Degrades

### The Memory Wall

GPU compute doubles roughly every two years; memory bandwidth improves far more slowly. An H100 peaks at 1000 TFLOPs/s — enough in theory for an 8B model to generate 62,000 tokens/s. In practice: ~200. LLM decode is memory-bound. Every output token requires streaming the entire model from GPU memory to the processor. At H100's 3.2 TB/s bandwidth, the ceiling is ~200 transfers/s. The processor idles waiting for weights.

Every production serving framework — vLLM, SGLang, TensorRT-LLM — is an engineering fight against this constraint. The primary weapon is the KV cache.

### The KV Cache

Every transformer maintains a Key-Value cache during inference: the computed attention state for every token in context. Without it, every output token requires full re-attention over the entire input. With it, generation is fast and contextually coherent.

The memory cost per token:

```
KV cache per token (bytes) = 2(K+V) x n_layers x n_kv_heads x head_dim x precision_bytes
```

Real numbers (Qwen3, GQA with 8 KV heads, FP16):

| Model | Per Token | 8K Context | 32K Context | 128K Context |
|-------|-----------|------------|-------------|--------------|
| Qwen3-8B | 0.14 MB | 1.12 GB | 4.5 GB | 18 GB |
| Qwen3-32B | 0.25 MB | 2.0 GB | 8.0 GB | 32 GB |

On an H100 80GB with Qwen3-8B loaded (~16 GB weights), ~56 GB remains for KV cache. That yields:

| Config | Context | Max Concurrent (ceiling) | Eviction Threshold |
|--------|---------|--------------------------|-------------------|
| Qwen3-8B, 1xH100 | 8K | ~49 | ~46 |
| Qwen3-8B, 1xH100 | 32K | ~12 | ~11 |
| Qwen3-8B, 1xH100 | 128K | ~3 | <3 |
| Qwen3-32B, 2xH100 | 8K | ~39 | ~35 |
| Qwen3-32B, 2xH100 | 32K | ~9 | ~8 |

Ceiling = 90% gpu_memory_utilization (vLLM default). Eviction begins at ~85%.

### Numerical Stability in the Transformer Pipeline

Standard matrix stability concerns apply throughout the transformer pipeline and interact with infrastructure-level degradation.

**Attention score explosion**: Raw Q·K^T dot products scale with dimensionality. Without the √d_k correction, values push into softmax tails where gradients vanish. This is a numerical stability intervention, not a minor detail.

**Softmax instability**: Large input values cause e^x overflow. Mitigated by subtracting the maximum value before exponentiating (preserves the math, keeps exponents safe).

**Exploding/vanishing gradients**: Deep composition of matrix multiplications — if spectral radius > 1, activations grow exponentially with depth; if < 1, they decay. Mitigated by layer normalization (rescales activations at each layer) and residual connections (effective transformation is I + f(x), keeping eigenvalues near 1).

**Quantization sensitivity**: Matrices with high condition numbers (large ratio of largest to smallest singular value) are more sensitive to quantization noise. INT4 quantization degrades math and reasoning before other tasks because those tasks depend on precise numerical relationships.

**FP8 KV cache precision loss**: Storing K/V in FP8 saves memory but introduces rounding error at every decode step. Over long generations, errors compound — particularly in reasoning chains where early attention patterns condition the entire downstream path.

---

## 4. Degradation Mechanisms

### 3.1 KV Cache Eviction and Preemption (Serving Layer)

KV cache *eviction* is distinct from KV cache *precision loss*. Precision loss (FP8 rounding) is a static tradeoff made at configuration time. Eviction is a dynamic event triggered by memory pressure at runtime.

When KV cache capacity is exhausted, vLLM preempts. The server evicts the entire PagedAttention block table for a request and recomputes the full prefill from the original prompt tokens when capacity returns. Two consequences:

**Latency explosion**: Full prefill must re-run from scratch. For 4K-token context, this costs hundreds of milliseconds before a single new output token appears.

**Batch composition variance**: RECOMPUTE re-runs the prefill correctly — RoPE positional encodings are reapplied from scratch, token positions are not disrupted. However, floating-point operations across different batch compositions are not bit-identical. When a request recomputes under a different concurrent batch than the original prefill, attention score distributions shift at the margins. For most workloads this is negligible. For deterministic reproducibility requirements and long reasoning chains where early token distributions condition the entire generation path, small shifts compound across hundreds of decode steps. This output variance across batch compositions is a documented vLLM behavior (GitHub issue #27433).

### 3.2 Prefix Cache Misses (Routing Layer)

Prefix cache miss is a separate routing problem: a request lands on a pod with no cached context for that session, forcing a cold prefill regardless of whether the system has overall memory headroom.

This matters especially for agentic workflows where the same system prompt and conversation history is re-submitted with every step. Without prefix caching, the serving layer re-processes the entire prefix from scratch on every request. With it, KV blocks for the shared prefix are computed once and reused.

Benchmarks from the llm-d project (October 2025): cache-aware scheduling achieved 87% hit rate vs. 30-40% with cache-blind round-robin routing, and 88% reduction in TTFT for warm hits vs. cold recompute. Cache-blind load balancers scatter requests randomly — cold prefill every time.

### 3.3 Continuous Batching Tax (Scheduler Layer)

Continuous batching admits new prefill requests into running decode batches to maximize GPU utilization. Under high concurrency, decode phases get preempted mid-sequence to process incoming prefills from other users.

Token generation becomes bursty. ITL (Inter-Token Latency) spikes: instead of steady 20ms/token, patterns like 20ms-20ms-20ms-200ms-20ms-180ms as prefill operations for other users interrupt the decode stream.

The direct quality impact is client-side timeout truncation. When ITL spikes cause a response to exceed the client or gateway timeout, generation is cut mid-output. For thinking-mode models, this cuts the reasoning chain before completion — the output reflects an incomplete reasoning path, not a complete one that ran fast.

```
E2E Latency = TTFT + (n_output_tokens x TPOT)

TTFT = queue_delay + prefill_time + (recompute_overhead if preempted)
TPOT = generation_time / output_tokens
```

Under load, both TTFT and TPOT inflate. A TPOT spike from 20ms to 200ms reduces the safe output budget from 1,500 tokens to 150 within a 30-second timeout window.

Benchmark (vLLM community, GitHub issue #20469, Qwen3-14B-AWQ on A100 80GB, 16K input):

| Concurrency | Mean TTFT | P99 TTFT | Output tok/s | Status |
|-------------|-----------|----------|--------------|--------|
| 1 | 3,345 ms | 3,697 ms | 26.84 | Healthy |
| 5 | 5,091 ms | 11,491 ms | 46.24 | Warning |
| 25 | >34,000 ms | collapse | degraded | Critical |

High throughput does not equal high goodput. vLLM defines goodput as the fraction of requests meeting latency SLOs. A system at 90% GPU utilization can fail 40% of requests on TTFT SLOs.

### 3.4 Cloud-Specific Degradation (Platform Layer)

Cloud providers apply additional quality pressure under load — rational economic decisions under finite GPU capacity, invisible to the caller.

**Quota-triggered model downgrade**: OpenAI's GPT-5 system card states explicitly that a real-time router selects which internal sub-model handles each request, and a "mini" version handles queries once usage limits are reached. The API response carries no flag. The model name in the header does not change.

**Quality-band routing**: Azure AI Foundry's model router documentation describes its "Balanced" mode as selecting models within a 1-2% quality tolerance band below the highest-quality option. "Cost" mode widens to 5-6%. Under peak traffic, cost pressure shifts routing toward the lower end.

**Silent model drift**: Chen, Zaharia, and Zou (Stanford/UC Berkeley, arxiv:2307.09009) benchmarked GPT-3.5 and GPT-4 at the same API endpoint across March–June 2023 with no disclosed version change. GPT-4 accuracy on prime number identification dropped from 84% to 51%. Code generation formatting degraded in both models.

**Context compaction and truncation**: When conversations approach context limits under load, providers silently compress earlier history into shorter summaries or truncate entirely. The model reasons over a compressed version of the original context.

**Quantization pressure**: INT4 quantization delivers 2.7x throughput improvement but degrades performance on GSM8K math reasoning and MT-Bench coding/STEM tasks — exactly the workloads where quality drops are perceived.

**Token-level KV cache compression**: At scale, some operators score individual tokens for importance and permanently drop low-scoring tokens from cache. Unlike RECOMPUTE, dropped tokens are not recovered. Research (arxiv:2511.04686) found coherence failure by turn 9-10 in multi-turn conversations under eviction pressure; complementary benchmarks across 16 LongBench tasks (arxiv:2510.13334) confirmed reasoning, QA, and summarization accuracy all degrade as cache availability drops.

---

## 5. Why Multi-Agent Systems Amplify the Problem

The Busy Mom paper frames the issue as single-request quality degradation. Multi-agent systems amplify this in qualitatively different ways.

### Error Compounding Across Steps

In an iterative pipeline where agent A's output becomes agent B's input, degradation at any step propagates forward and compounds. Same sensitivity-to-initial-conditions problem as numerical integration of differential equations, but running through softmax nonlinearities and natural language — effectively uncharacterizable.

A simple accumulation model frames the measurement target:

```
drift(0) = 0
drift(n) = g(drift(n-1), variance(step_n))
```

Where `variance(step_n)` is the infrastructure-induced deviation at step n, and `g` is the (unknown) compounding function. The experiment measures the shape of `drift(n)` — linear, exponential, or saturating — which is itself a primary finding.

### Non-Determinism Across Agents in the Same Pipeline

Different agents in the same pipeline may experience different levels of degradation on the same run. Agent A hits a warm cache pod, Agent B hits a cold pod during a load spike, Agent C gets preempted. The orchestrator sees three 200 status codes and treats all outputs as equivalent when they aren't.

### Feedback Loops and Self-Reinforcing Drift

If the evaluator agent is subject to load-dependent quality variation, its judgment about whether to retry is unreliable. During peak hours: a degraded generator producing mediocre output and a degraded evaluator accepting it. The system's effective quality ceiling becomes a function of the worst concurrent degradation across all agents.

### Batch Inference Interference

Shared infrastructure batches requests from multiple users into the same forward pass (continuous batching). A prompt's attention computation shares GPU cycles with unrelated prompts, and the batch composition changes every request. For multi-agent pipelines, this means each agent call in the same pipeline may execute under different batch compositions, introducing per-step variance that is invisible to the orchestrator and impossible to control.

### Reproducibility is Effectively Gone

Same pipeline, same inputs, different final outputs depending on transient platform state. The degraded result cannot be reproduced because the degradation was a function of infrastructure state that no longer exists.

---

## 6. The Dedicated Hardware Thesis

The practical implication builds to a specific claim: **for iterative multi-agent systems requiring reproducible results, dedicated model on dedicated hardware may be the only viable option.**

Three tiers of infrastructure produce three qualitatively different variance profiles:

### Tier 1: Shared Cloud API

Maximum variance. Subject to all degradation mechanisms simultaneously: model substitution, KV cache eviction, prefix cache misses, continuous batching interference, context compaction, quantization pressure shifts. Variance is invisible, non-reproducible, and compounds with pipeline depth. The user has zero visibility into which mechanisms are active on any given request.

### Tier 2: Self-Hosted with Production Serving Stack (vLLM, SGLang, TGI)

Eliminates cloud-specific mechanisms (model substitution, context compaction, provider-side quantization changes). But retains serving-layer variance: KV cache eviction under local load, continuous batching across local concurrent requests, prefix cache miss on restart or memory pressure, batch composition variance on recompute. Running vLLM locally does not eliminate the class of variance — it only removes the cloud layer. If the local deployment serves multiple concurrent requests or pipelines, the same contention dynamics apply.

### Tier 3: Dedicated Sequential Inference

Single model, single GPU (or fixed multi-GPU), no batching, no KV cache eviction (single request never exceeds capacity by construction), deterministic execution path. Variance floor: sampling randomness only (eliminated at temperature=0 with fixed seed). This is the control condition and the only configuration where reproducibility is architecturally guaranteed.

The experiment must include all three tiers to isolate which layer of the stack contributes what fraction of the total variance.

---

## 7. Experiment Design

### Core Question

For a fixed multi-agent pipeline with fixed inputs, how much does the final output vary as a function of infrastructure conditions, and how does that variance amplify across pipeline depth?

### Pipeline Design

An iterative feedback loop:

- **Generator agent**: produces a technical artifact (code, analysis, plan)
- **Evaluator agent**: scores or critiques the output against defined criteria
- **Refiner agent**: takes the critique and revises
- Loop back to evaluator for N iterations or until convergence

This captures both compounding (each iteration conditions the next) and the feedback loop problem (evaluator quality variance affects convergence behavior).

### Variables

**Fixed**: prompt templates, system messages, input data, orchestration logic, model identifier, temperature (0 or fixed seed where supported)

**Varied**:
- Time of day (proxy for load on shared infrastructure)
- Infrastructure tier (shared cloud / self-hosted vLLM / dedicated sequential)

**Measured per step**:
- Wall-clock latency (TTFT where available, total response time minimum)
- Token count (input and output)
- Deterministic similarity metric against baseline run
- Evaluator accept/reject decision
- KV cache utilization (self-hosted tiers, via Prometheus `/metrics`)
- Prefix cache hit rate (self-hosted tiers)
- Preemption count (self-hosted tiers, `total_cumulative_preemption_cnt`)

**Measured per pipeline run**:
- Final output similarity to reference
- Total iterations to convergence
- Total elapsed time
- Variance across repeated runs within the same time window
- Drift curve: `drift(n)` plotted across pipeline depth

### Experimental Protocol

Run identical pipeline on identical inputs, repeatedly, across conditions:

**Time windows (shared cloud)**:
- **Off-peak**: Sunday 5-8 AM ET, N=20-30 repetitions
- **Peak**: Tuesday/Wednesday 1-4 PM ET, N=20-30 repetitions
- **Shoulder**: weekday evenings, N=20-30 repetitions

**Infrastructure tiers**:
- At least two shared cloud providers
- Self-hosted vLLM on fixed hardware (concurrent load simulated at levels matching cloud peak/off-peak)
- Self-hosted dedicated sequential inference on same hardware (single-request, no batching)

The comparison between within-window variance and between-window variance isolates the time-of-day effect. The comparison between tiers isolates the infrastructure-layer contribution.

### Similarity Metrics

Use multiple metrics to avoid single-metric blindness:

- **Exact match rate**: crude baseline — what fraction produce identical outputs?
- **Token-level edit distance**: normalized by length for comparability
- **Embedding cosine similarity**: using a fixed, locally-run embedding model (do not subject measurement to the same variance being measured)
- **Semantic equivalence via local judge**: local model scoring whether outputs are semantically equivalent, partially equivalent, or divergent
- **Task-specific correctness**: test cases for code, rubric criteria for analysis

### Analysis Targets

1. **Variance amplification curve**: plot output variance as function of pipeline depth. Shape (linear, exponential, saturating) is itself a primary finding.
2. **Convergence instability**: distribution of iterations-to-convergence across peak vs. off-peak and across infrastructure tiers.
3. **Evaluator reliability**: agreement rate of accept/reject decisions on the same input across runs. Inconsistency = feedback loop failure mode.
4. **Time-of-day effect size**: statistical comparison across time windows (Kruskal-Wallis or similar; distributions likely non-normal).
5. **Infrastructure tier effect size**: variance comparison across tiers. The delta between Tier 2 and Tier 3 isolates the serving-stack contribution; delta between Tier 1 and Tier 2 isolates the cloud-layer contribution.
6. **Leading indicators**: correlation between per-step infrastructure telemetry (KV cache utilization, prefix cache hit rate, TTFT, preemption count) and output variance. Identifies which signals predict quality degradation before it reaches the output.

---

## 8. Output Space Dimensionality as a Variance Absorber

Contradictory results in the literature suggest that output structure may determine whether infrastructure variance reaches the surface. A multi-agent incident response system (arXiv:2511.15755) reported "zero quality variance" across 348 trials, while other work (arXiv:2505.21298, arXiv:2601.06112) demonstrates that LLM non-determinism is pervasive and consequential. Both can be true simultaneously — the resolution is measurement granularity and output space structure.

### The Decision Boundary Hypothesis

When an agent's output is forced through a narrow decision boundary — yes/no, classify into K categories, select from a fixed set of options — infrastructure-induced perturbations in the internal computation are absorbed as long as they don't push the output across a boundary. The variance is still present in the model's hidden states, attention distributions, and logit scores. It simply doesn't survive the discretization into a final answer.

Widen the output space to free-form text, code, or analysis, and there is no decision boundary to absorb anything. Every internal perturbation propagates directly into the output. The variance that was invisible in a classification task becomes the dominant signal in a generation task.

### Implications for Multi-Agent Pipelines

This suggests that vulnerability to infrastructure drift is a function of output space dimensionality at each step in the pipeline:

- A pipeline of **constrained decisions** (classify → rank → select → approve) may be naturally robust to infrastructure variance, even on shared cloud during peak hours. Each step's narrow output space absorbs per-step perturbations before they can compound.
- A pipeline of **free-form generations** (draft → critique → revise → re-evaluate) is maximally exposed. Every step produces high-dimensional output where variance has room to express itself, and that variance feeds directly into the next step's input.
- A **mixed pipeline** — free-form generation constrained by structured intermediate checkpoints — may offer a practical middle ground. Force the evaluator to produce a structured rubric score rather than a free-text critique. Force the refiner to address specific scored dimensions rather than open-ended revision.

### Experimental Test

Run the same pipeline in two configurations:

- **Unconstrained**: each agent produces free-form text output
- **Constrained**: each agent produces structured output (JSON with fixed schema, categorical fields, bounded numeric scores) that gets expanded back to natural language only at the final step

The delta in output variance between these configurations, measured under the same infrastructure conditions, isolates the contribution of output space structure as a drift defense. If constrained pipelines show significantly lower variance, this is both a finding and a practical mitigation strategy: **you can build drift-resistant multi-agent systems on noisy infrastructure by constraining intermediate output spaces**, even if you can't control the infrastructure itself.

---

## 9. The Thesis

In iterative multi-agent systems, infrastructure-induced variance doesn't just affect individual responses — it changes the behavioral trajectory of the system. Same pipeline, same inputs, different emergent behavior depending on when you run it and what infrastructure serves it.

The weights are static. Everything around them is not. Inference quality is not a model property — it is a platform property. Anyone building production multi-agent systems on shared infrastructure needs to think about variance budgets the way numerical methods people think about error budgets.

Public AI services may not be usable as the basis for iterative agents where reproducibility matters. Self-hosted inference with production serving stacks (vLLM, SGLang) reduces but does not eliminate the problem — the same contention dynamics apply locally under concurrent load. Only dedicated model on dedicated hardware, running sequential inference without batching or cache eviction, provides the architectural guarantees needed for reproducible multi-agent pipelines.

This experiment measures how much that matters in practice.

---

## References

- "Agentic Workflow Guardrails: Embedding Deterministic Governance into Stochastic AI/Agentic Systems." DTCC/FINOS Financial Services Hackathon (2025).
- Nar, F.E., Shaw, R. et al. "The Busy Mom Syndrome: Why your AI reasons better on Sunday morning than Tuesday afternoon."
- Chen, L., Zaharia, M., Zou, J. "How Is ChatGPT's Behavior Changing over Time?" arXiv:2307.09009 (2023).
- arXiv:2511.04686 — Coherence failure under KV cache eviction pressure in multi-turn conversations (2025).
- arXiv:2510.13334 — LongBench task accuracy degradation under cache availability reduction (2025).
- vLLM GitHub issues #27433 (batch composition variance), #20469 (queue collapse benchmarks).
- llm-d — KV cache-aware routing benchmarks (October 2025).
- OpenAI GPT-5 System Card — quota-triggered model routing.
- Microsoft Azure AI Foundry — model router quality-band documentation.
