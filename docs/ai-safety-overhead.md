# The Cost of Making AI Safe: Measuring the Safety Tax

## Overview

As enterprises deploy LLM-powered agents into production, a growing stack of safety infrastructure is required around every inference call: guardrails, content moderation, alignment checks, constrained decoding, and audit logging. Each layer adds latency, compute cost, and operational complexity.

This document surveys available data on the overhead these layers introduce, identifies gaps in the research, and proposes how drift measurements could contribute to understanding the true cost of safe AI deployment.

## The Safety Stack

A production agentic AI call in a regulated enterprise (e.g., the the platform architecture described in [the platform-constrained-agents.md](the platform-constrained-agents.md)) may pass through some or all of the following layers:

| Layer | Example Technology | Purpose | Category |
|-------|-------------------|---------|----------|
| Input guardrails | NeMo Guardrails, PromptGuard 2 | Jailbreak detection, PII scanning, topic control | Safety |
| The actual LLM call | Qwen, Llama, Claude | The work the user wants done | Productive |
| Output guardrails | NeMo Guardrails, Llama Guard | Content safety, hallucination detection | Safety |
| Safety judge LLM | Llama Guard 3, AlignmentCheck | Full model inference to classify output safety | Safety |
| Constrained decoding | SGLang FSM, XGrammar | Logit masking at every token to enforce structure | Safety + Reproducibility |
| Deterministic inference | SGLang `--enable-deterministic-inference` | Reproducible output for audit/compliance | Reproducibility |
| Code analysis | CodeShield | Static analysis of LLM-generated code | Safety |
| Audit/compliance logging | Platform-specific | Storage and processing for regulatory compliance | Compliance |

The distinction between **safety** (preventing harmful outputs) and **reproducibility** (ensuring consistent, auditable outputs) matters. They serve different purposes: safety protects users and systems; reproducibility supports compliance, debugging, and quality assurance. Some layers (constrained decoding) serve both. The costs should be evaluated separately because an organization may need one without the other.

Each layer has measurable overhead. The question nobody has answered comprehensively is: **what is the total cost of the safety stack relative to the productive inference call?**

## Measured Overhead by Layer

### NVIDIA NeMo Guardrails

NVIDIA reports that running up to 5 GPU-accelerated guardrails in parallel adds approximately **0.5 seconds of latency** per request, with a 1.4x improvement in detection rate. The first guardrail layer incurs the most latency; additional layers add diminishing marginal cost due to parallel execution.

NeMo Guardrails supports a "single call" mode (since v0.6.0) that performs input validation, generation, and output validation in one LLM call, reducing both latency and token usage for simpler RAG scenarios.

Source: [NVIDIA Technical Blog — Measuring Effectiveness and Performance of AI Guardrails](https://developer.nvidia.com/blog/measuring-the-effectiveness-and-performance-of-ai-guardrails-in-generative-ai-applications/)

### Llama Guard / LlamaFirewall

Meta's Llama Guard 3-1B achieves a 76% threat detection rate with **0.165 seconds latency per check** on NVIDIA A30 GPUs. In practical deployments, latency is typically reported as **200–400ms** per safety check. The INT4-quantized 1B variant can run on mobile CPUs at ~30 tokens/second with a 2.5s time-to-first-token.

For comparison, API-based alternatives like Lakera Guard achieve 0.066s per prompt, Azure Prompt Shield 0.349s, and Calypso AI Moderator 0.25s.

Meta's LlamaFirewall (May 2025) composes three layers:
1. **PromptGuard 2** — lightweight BERT classifier for prompt injection (~86M parameters)
2. **AlignmentCheck** — chain-of-thought reasoning auditor (higher compute, lower ASR)
3. **CodeShield** — static analysis across 8 languages (96% precision, 79% recall)

Combined, these three layers reduced attack success rate from 17.6% to 1.75% (a 90% reduction), but at a utility drop to 42.7% — meaning the safety stack rejected nearly 60% of legitimate outputs.

**Open question**: The 42.7% utility figure is striking. If safety layers reject more than half of productive work, the effective compute cost per useful output more than doubles — but only if rejected outputs must be regenerated. The actual cost depends on the retry policy: does the system regenerate, fall back to a human, or return a canned response? No published study has measured the retry overhead or the downstream cost of false rejections.

Sources:
- [Llama Guard 3-1B-INT4 (arXiv)](https://arxiv.org/html/2411.17713v1)
- [LlamaFirewall (arXiv)](https://arxiv.org/abs/2505.03574)
- [Benchmarking Llama Model Security Against OWASP Top 10](https://arxiv.org/html/2601.19970v1)

### SGLang Constrained Decoding

SGLang's compressed finite-state machine approach enables 3x faster JSON decoding compared to naive constrained decoding. The overhead depends on schema complexity:

- **Simple schemas**: 5–15% latency increase
- **Complex schemas**: 30–60% latency increase
- **With cached compiled grammars**: <10% overhead

SGLang hides much of this cost by overlapping CPU-bound grammar computation with GPU inference. By contrast, vLLM shows significant performance degradation with guided decoding at batch sizes of 8+.

Constrained decoding serves both safety (preventing malformed output) and reproducibility (enforcing structural consistency). It operates at the token level during generation, unlike post-hoc guardrails that evaluate completed output.

Sources:
- [SqueezeBits — Guided Decoding Performance on vLLM and SGLang](https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang)
- [SGLang NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf)
- [SGLang documentation](https://docs.sglang.io/)

### Deterministic Inference

SGLang's `--enable-deterministic-inference` flag incurs approximately a **34% throughput reduction** based on their documentation. This is distinct from safety: deterministic inference does not prevent harmful outputs, it ensures that the same input produces the same output every time. The primary use cases are:

- **Auditability**: regulators or internal review can reproduce an output
- **Debugging**: developers can isolate whether a failure is due to code or model non-determinism
- **Quality assurance**: drift measurement becomes meaningful only when non-determinism is controlled

The 34% cost is significant but narrowly scoped: it only applies when bit-exact reproducibility is required. For many applications, "close enough" consistency (measurable via drift's exact match rate and edit distance) may be sufficient at lower cost.

### Guardrail Robustness

Adding overhead only makes economic sense if the guardrails actually work. Available evidence raises concerns:

- **HiddenLayer (Oct 2025)**: Bypassed OpenAI's guardrails framework using straightforward techniques. The safety judge LLM was susceptible to the same manipulation as the model it was protecting.
- **Cisco research**: Multi-turn prompt attacks achieved ~60% average success rate on open-weight models, with one model reaching 92.78% success under specific conditions.
- **arXiv:2503.24191**: Demonstrated that constrained decoding grammars can be weaponized to bypass safety mechanisms — the safety infrastructure itself becomes an attack surface.
- **Statistical tail risk (Feb 2026)**: Claims of "99% safe" obscure the reality that adversarial attacks succeed 89% of the time at 10,000 attempts on GPT-4o.

These findings do not mean guardrails are worthless — they raise the bar for most attacks and provide defense-in-depth. But they complicate the cost-benefit calculation: the safety stack's compute cost is measurable, while its protection is probabilistic and bypassable.

Sources:
- [When the Statistical Tail Wags the Average Dog](https://medium.com/@johneverettbutton/when-the-statistical-tail-wags-the-average-dog-a-warning-for-ai-risk-managers-part-1-7f0a104140c9)
- [Proof-of-Guardrail in AI Agents (arXiv)](https://arxiv.org/html/2603.05786)

## The Agentic Multiplier

Safety overhead compounds in agentic workflows because each step in an agent's execution may require its own safety checks:

- **Anthropic's internal data** shows agents consume ~4x more tokens than standard chat, and multi-agent systems push that to ~15x.
- **Hierarchical agent architectures** (e.g., ReAcTree) incur exponential increases in token consumption compared to linear chains.
- Every agent action may require input guardrails, output guardrails, and a safety judge call — potentially tripling the per-step check count.
- In iterative/looping agents, the safety stack runs on every iteration of every inner step.

**Illustrative example from drift's iterative refactor pipeline:**

A single run with 1 loop iteration executes 4 LLM calls (1 initial analysis + 3 inner steps). On DeepInfra, this took ~28 seconds of productive inference (measured: 14.6s + 7.0s + 4.5s + 2.2s). If each call requires input guardrails, output guardrails, and a safety check, the overhead estimate using published per-check latencies is:

| Component | Count | Latency per check | Total |
|-----------|-------|--------------------|-------|
| Productive inference | 4 calls | measured | 28.3s |
| Input guardrails (NeMo) | 4 checks | ~0.5s | ~2.0s |
| Output guardrails (Llama Guard) | 4 checks | ~0.2s | ~0.8s |
| **Estimated safety overhead** | | | **~2.8s (~10%)** |

This estimate is illustrative, not empirical. The per-check latencies come from vendor benchmarks run in isolation, not from a combined deployment where safety checks compete for GPU resources with productive inference. The actual overhead in production would depend on whether safety models run on dedicated GPUs, share resources with the main model, or use API-based services.

For a run that loops 5 times (16 LLM calls), the same calculation gives ~14s of safety overhead on top of productive inference. For a 30-run drift experiment, this scales to ~480 safety checks.

**Open question**: No published study quantifies the ratio of safety-compute to productive-compute across a complete agentic workflow including all layers. The data exists in fragments but has not been aggregated.

## The Economics

### The cost of not having safety

- Enterprise AI breach costs average **$10.22M** in the US (2025).
- Organizations with AI-specific controls reduced breach costs by **$2.1M** on average.
- 97% of AI-related breaches in 2025 occurred in environments without access controls.

Source: [Deploying AI Guardrails at Production Scale](https://introl.com/blog/ai-safety-infrastructure-guardrails-production-scale-2025)

### The cost of having safety

- **AI content moderation market**: $1B (2024) → projected $2.6B (2029).
- **Per-request overhead**: ~0.5s (NeMo) + ~0.2s (Llama Guard) per productive call, assuming parallel execution.
- **Throughput cost**: 34% for deterministic inference (reproducibility), 5–60% for constrained decoding (safety + reproducibility).
- **Utility loss**: up to 57.3% of legitimate outputs rejected (LlamaFirewall combined layers).

### What's missing: the ratio

A rough framework for comparing safety cost to breach cost:

- If an enterprise runs 1M inference calls/month and safety adds ~0.7s of GPU time per call, that's ~194 GPU-hours/month of safety compute.
- At $3/GPU-hour (H100 cloud), that's ~$583/month in safety compute.
- Against a $2.1M average breach cost reduction, the ROI is clear at this scale.

However, this calculation only accounts for latency-based overhead. It excludes:
- GPU memory consumed by co-located safety models
- The cost of rejected outputs and retries (the LlamaFirewall utility gap)
- Operational costs of maintaining, updating, and monitoring guardrail configurations
- The throughput reduction from deterministic inference and constrained decoding

The actual safety tax likely exceeds the latency-only estimate, but by how much remains unmeasured.

### The market context

- AI inference spending crossed **55% of AI cloud infrastructure** ($37.5B) in early 2026, projected to hit 70–80% by year's end.
- Hyperscaler AI capex hit **$600B** in 2026 (36% YoY), with 75% tied to AI infrastructure.
- Inference costs declined ~280-fold between 2022 and 2024, but usage grew faster.

If even 5% of the $37.5B inference spend goes to safety overhead, that's $1.9B — comparable to the entire content moderation market. If the actual figure is 15–20% (plausible given the agentic multiplier and utility loss), it would represent $5.6–7.5B in safety compute. These figures are speculative and included only to illustrate why measuring the actual ratio matters.

Sources:
- [AI Inference Costs: 55% of Cloud Spending in 2026](https://byteiota.com/ai-inference-costs-55-of-cloud-spending-in-2026/)
- [CES 2026: AI Compute Shift from Training to Inference](https://www.computerworld.com/article/4114579/ces-2026-ai-compute-sees-a-shift-from-training-to-inference.html)

## Gaps in the Research

The following questions do not have published answers as of March 2026:

1. **Total safety-to-productive compute ratio**: What percentage of GPU-hours in a production agentic deployment is spent on safety vs. productive work? Fragments exist (NeMo adds 0.5s, Llama Guard adds 0.2s, deterministic inference costs 34%) but no end-to-end measurement.

2. **Safety utility loss in practice**: LlamaFirewall's 42.7% utility rate was measured in a specific benchmark. What is the utility loss in real production deployments? What is the cost of re-generation when outputs are rejected? How do false rejection rates vary by domain?

3. **Diminishing returns on stacked guardrails**: NeMo shows diminishing latency per additional layer, but does detection quality improve proportionally? Is the marginal value of the 4th or 5th guardrail worth its compute?

4. **Safety overhead in iterative/looping agents**: When an agent loops 5 times with 3 inner steps, the safety stack may execute 45+ checks per user action. No study has measured this compound overhead in a real deployment.

5. **Constrained decoding vs. post-hoc guardrails**: These are fundamentally different approaches — prevention at generation time vs. detection after generation. Constrained decoding has a per-token cost but zero false rejections for structural violations. Post-hoc guardrails have a per-output cost with nonzero false rejection rates. No study has compared their compute efficiency for equivalent safety outcomes.

6. **Self-hosted vs. cloud + guardrails total cost of ownership**: Drift's data shows self-hosted Ollama produces 100% deterministic output at temp=0 with no guardrail overhead. But self-hosting has its own costs: hardware ($20K–$200K+ for GPU servers), operations staff (~$135K/year for MLOps engineers), power, cooling, and compliance infrastructure. No published comparison weighs the total cost of self-hosted deterministic inference against cloud inference plus the guardrails needed to compensate for cloud non-determinism.

## Where Drift Can Contribute

The drift project is positioned to measure several aspects of the safety tax that existing tools do not:

### 1. Quantifying the Reproducibility Cost Curve

Drift already measures output variability across identical runs. By comparing runs across different infrastructure configurations:
- **Unconstrained cloud inference** (current DeepInfra results: 40% exact match on code generation, 83.3% on structured extraction)
- **Constrained decoding** (SGLang with JSON grammar enforcement)
- **Deterministic inference** (SGLang with `--enable-deterministic-inference`)
- **Self-hosted inference** (current Ollama results: 100% exact match)

...drift could produce a cost curve: how much compute does each approach require to achieve a given level of output consistency? This would provide empirical data for gap #5 and #6 above.

This is a reproducibility measurement, not a safety measurement. Drift does not evaluate whether outputs are harmful — it evaluates whether they are consistent. But consistency is a prerequisite for meaningful safety evaluation: if outputs vary unpredictably, safety audits of sample outputs may not generalize.

### 2. Measuring Guard Layer Impact on Output Distribution

Adding a measurement mode where drift runs with and without guardrail layers would show:
- How much latency each layer adds in practice (not in isolation benchmarks)
- Whether guardrails change the output distribution (do they bias outputs toward certain patterns?)
- The retry rate: how often guardrails reject an output and force re-generation

This would require integrating NeMo Guardrails or Llama Guard into drift's pipeline executor — a concrete engineering task that could produce novel data.

### 3. Loop-Aware Overhead Measurement

Drift's iterative_refactor pipeline already measures per-iteration metrics across runs. Extending this to track:
- Total LLM calls per run (productive + safety)
- Latency breakdown per call category
- Cumulative safety overhead as iterations increase

...would provide the first published data on compound overhead in looping agents (gap #4). The infrastructure already exists; the main work is instrumenting the safety layers.

### 4. Safety-Compute Efficiency Scoring

A composite metric combining:
- Output consistency (exact match rate, edit distance)
- Latency overhead (safety layer time vs. productive inference time)
- Rejection rate (outputs blocked by guardrails)
- Detection rate (threats caught by guardrails)

This would give enterprises a way to compare safety configurations by compute efficiency rather than just detection rate. For example: "Configuration A catches 90% of threats at 15% overhead; Configuration B catches 95% at 40% overhead."

This metric does not exist today. Defining and validating it would require collaboration with security researchers who can provide realistic threat benchmarks.

## Summary

The AI safety infrastructure stack adds measurable overhead at every layer, but no published study aggregates the total cost across a complete agentic deployment. Available per-layer data suggests overhead ranges from ~10% (parallel guardrails on simple requests) to significantly more when utility loss, retries, and iterative agent workflows are factored in. The upper bound is unknown.

The economic case for safety is clear at the breach-prevention level ($2.1M average cost reduction). The open question is not whether to invest in safety, but whether the current approach — stacking LLM-based safety judges on top of LLM-based agents — is the most compute-efficient way to achieve it. Alternative approaches like constrained decoding (preventing bad outputs at generation time) or self-hosted deterministic inference (eliminating variability at the source) may offer better cost-efficiency ratios for certain use cases, but no comparative data exists.

Drift is positioned to provide empirical measurements in this space, particularly around the reproducibility cost curve and compound overhead in iterative agent workflows. These measurements would not replace safety evaluation, but would provide the cost side of the cost-benefit equation that is currently missing.
