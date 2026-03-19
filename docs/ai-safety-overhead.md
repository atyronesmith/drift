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

## Safety Layers as a System, Not Independent Taxes

The overhead figures above are measured per-layer in isolation. In practice, safety layers interact with each other and with the productive model in ways that can amplify overhead beyond simple addition:

- **Input guardrails can distort prompts.** A strict input filter that rewrites or truncates user input to remove potential injection attempts may alter the semantic content of the prompt. This can produce lower-quality outputs from the productive model, which are then more likely to trigger output guardrails — creating a feedback loop where input safety degrades output quality.

- **Constrained decoding can increase repetition and reduce quality.** Enforcing JSON structure via logit masking restricts the model's token choices at every step. There is evidence that this can increase output repetition and raise perplexity, potentially making outputs more likely to fail downstream content checks. No study has measured this interaction directly, but it follows from how constrained decoding narrows the output distribution.

- **Stacked guardrails create compound false-positive risk.** If each of 3 independent safety checks has a 10% false rejection rate, the probability that at least one rejects a legitimate output is 1 − 0.9³ ≈ 27%. With 5 checks: ~41%. This assumes independence; correlated checks would have lower compound rates, but the interaction has not been measured.

These interactions mean the total safety tax is not simply the sum of per-layer overheads. The system-level cost could be higher (due to feedback loops and compound false positives) or lower (if layers are redundant and can be pruned). No published study has modeled the safety stack as a system with feedback loops rather than a set of independent cost items.

**Research area for drift**: Running identical tasks through progressively more safety layers and measuring both output quality (task success rate, compilation rate) and output consistency (edit distance, exact match) would reveal whether adding layers degrades the productive model's output distribution — and at what point additional layers stop improving safety relative to their cost.

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

### The cost of having safety: compute

- **AI content moderation market**: $1B (2024) → projected $2.6B (2029).
- **Per-request overhead**: ~0.5s (NeMo) + ~0.2s (Llama Guard) per productive call, assuming parallel execution.
- **Throughput cost**: 34% for deterministic inference (reproducibility), 5–60% for constrained decoding (safety + reproducibility).
- **Utility loss**: up to 57.3% of legitimate outputs rejected (LlamaFirewall combined layers).

### The cost of having safety: human operations

The compute overhead is only part of the cost. Safety infrastructure requires ongoing human effort:

- **Guardrail tuning**: engineers must configure and update PII detection patterns, topic boundaries, hallucination thresholds, and prompt injection rules. These require domain expertise and ongoing maintenance as attack patterns evolve.
- **False positive investigation**: when legitimate outputs are blocked (the LlamaFirewall 57.3% rejection problem), engineers must diagnose why. Was it a misconfigured rule? An edge case in the safety model? This debugging time is a direct cost of the safety stack.
- **Human-in-the-loop overrides**: in high-stakes deployments, blocked outputs may require human review before being released or regenerated. This introduces latency measured in minutes or hours, not milliseconds.
- **Monitoring and incident response**: safety layers generate alerts, logs, and metrics that someone must monitor. False alarm fatigue is a known problem in security operations.

No published data quantifies this as a per-inference metric (e.g., person-hours per 1,000 inferences). Enterprises track these costs in aggregate staffing budgets, but the per-inference allocation is not reported. This is a significant blind spot: for some organizations, the human cost of safety may exceed the compute cost.

### What's missing: the ratio

A rough framework for comparing safety cost to breach cost:

- If an enterprise runs 1M inference calls/month and safety adds ~0.7s of GPU time per call, that's ~194 GPU-hours/month of safety compute.
- At $3/GPU-hour (H100 cloud), that's ~$583/month in safety compute.
- Against a $2.1M average breach cost reduction, the ROI is clear at this scale.

However, this calculation only accounts for latency-based overhead. It excludes:
- GPU memory consumed by co-located safety models
- The cost of rejected outputs and retries (the LlamaFirewall utility gap)
- Human operational costs (guardrail tuning, false positive investigation, monitoring)
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

## Alternative Safety Approaches

The analysis above focuses on the dominant paradigm: post-hoc LLM-based guardrails evaluating outputs from productive LLMs. This is not the only approach. Several alternatives have different cost profiles:

### Input-side sanitization

PII scrubbing, prompt reformatting, and input normalization can be done with traditional NLP tools (regex, NER models, rule engines) at near-zero latency compared to LLM-based checks. These are well-understood, deterministic, and cheap. They address a narrow class of risks (data leakage, some injection patterns) but cannot evaluate semantic safety of outputs.

### Execution-based verification

For code generation tasks, running LLM-generated code in sandboxed environments (containers, WASM, Firebase emulators) verifies safety through behavior rather than content classification. A sandbox that catches a `DROP TABLE` statement is more reliable than an LLM judge that might miss it. The cost profile is different: sandbox execution time vs. LLM inference time. For code-specific tasks, this may be both cheaper and more effective than an LLM safety judge.

### Intrinsic safety via fine-tuning

Models can be fine-tuned (via RLHF, DPO, or constitutional AI methods) to refuse harmful requests internally, reducing the need for external guardrails. The cost is borne during training rather than at inference time. The tradeoff is that intrinsic safety is harder to audit, update, and customize per-deployment compared to external guardrails. It also contributes to the utility loss problem: an overly-aligned model may refuse legitimate requests.

### The comparison gap

No published study compares the safety tax of these approaches head-to-head for equivalent safety outcomes. Key questions:

- Is the LlamaFirewall 42.7% utility loss inherent to the LLM-as-judge approach, or would sandbox-based verification achieve better utility at lower compute?
- For structured output tasks (JSON, code), does constrained decoding at generation time eliminate the need for post-hoc content checks entirely?
- What is the total cost of intrinsic safety (training-time alignment) vs. extrinsic safety (inference-time guardrails) over a model's production lifecycle?

These comparisons would reframe the question from "how much does safety cost?" to "which safety approach delivers the best return on investment?"

## Gaps in the Research

The following questions do not have published answers as of March 2026:

1. **Total safety-to-productive compute ratio**: What percentage of GPU-hours in a production agentic deployment is spent on safety vs. productive work? Fragments exist (NeMo adds 0.5s, Llama Guard adds 0.2s, deterministic inference costs 34%) but no end-to-end measurement.

2. **Safety utility loss in practice**: LlamaFirewall's 42.7% utility rate was measured in a specific benchmark. What is the utility loss in real production deployments? What is the cost of re-generation when outputs are rejected? How do false rejection rates vary by domain?

3. **The retry death spiral**: If safety layers reject 57% of outputs, and each retry has a probability of also being rejected, how many retries are needed per useful output? At what rejection rate does the system become uneconomical? This is a multiplicative cost that can turn a "10% latency tax" into a "200%+ effective overhead" problem in agentic workflows.

4. **Diminishing returns on stacked guardrails**: NeMo shows diminishing latency per additional layer, but does detection quality improve proportionally? Is the marginal value of the 4th or 5th guardrail worth its compute? And do stacked layers interact negatively — increasing false rejections through compound probability?

5. **Safety overhead in iterative/looping agents**: When an agent loops 5 times with 3 inner steps, the safety stack may execute 45+ checks per user action. No study has measured this compound overhead in a real deployment.

6. **Constrained decoding vs. post-hoc guardrails**: These are fundamentally different approaches — prevention at generation time vs. detection after generation. Constrained decoding has a per-token cost but zero false rejections for structural violations. Post-hoc guardrails have a per-output cost with nonzero false rejection rates. No study has compared their compute efficiency for equivalent safety outcomes.

7. **Self-hosted vs. cloud + guardrails total cost of ownership**: Drift's data shows self-hosted Ollama produces 100% deterministic output at temp=0 with no guardrail overhead. But self-hosting has its own costs: hardware ($20K–$200K+ for GPU servers), operations staff (~$135K/year for MLOps engineers), power, cooling, and compliance infrastructure. No published comparison weighs the total cost of self-hosted deterministic inference against cloud inference plus the guardrails needed to compensate for cloud non-determinism. A related question: do managed API providers (OpenAI, Anthropic, DeepInfra) apply their own opaque guardrails that add latency and bias outputs in ways the customer cannot measure or control?

8. **Safety-induced capability degradation**: Do safety layers make agents worse at their jobs, beyond just blocking outputs? If guardrails bias the output distribution toward "safer" but less useful responses (more verbose, more hedging, avoiding edge cases), the effective cost includes reduced task success rate — not just latency and rejection. This "capability tax" is distinct from the compute tax and may be harder to detect.

## Where Drift Can Contribute

The drift project is positioned to measure several aspects of the safety tax that existing tools do not:

### 1. Quantifying the Reproducibility Cost Curve

Drift already measures output variability across identical runs. By comparing runs across different infrastructure configurations:
- **Unconstrained cloud inference** (current DeepInfra results: 40% exact match on code generation, 83.3% on structured extraction)
- **Constrained decoding** (SGLang with JSON grammar enforcement)
- **Deterministic inference** (SGLang with `--enable-deterministic-inference`)
- **Self-hosted inference** (current Ollama results: 100% exact match)

...drift could produce a cost curve: how much compute does each approach require to achieve a given level of output consistency? This would provide empirical data for gap #6 and #7 above.

This is a reproducibility measurement, not a safety measurement. Drift does not evaluate whether outputs are harmful — it evaluates whether they are consistent. But consistency is a prerequisite for meaningful safety evaluation: if outputs vary unpredictably, safety audits of sample outputs may not generalize.

### 2. The Retry Death Spiral Metric

For a given safety configuration, drift can measure:
- The percentage of productive inferences rejected by safety layers.
- The average number of retries required to produce an accepted output.
- The edit distance between rejected and accepted outputs (revealing what the safety layer objected to).
- The total wall-clock time per useful output including retries.

**How this works in practice**: Instrument drift's pipeline executor to wrap each LLM call with a configurable safety check (e.g., Llama Guard, NeMo Guardrails). On rejection, log the rejection reason and retry with the same prompt. Track retries per step and per run.

If LlamaFirewall rejects 57% of outputs (per their benchmark), and each retry independently faces the same rejection probability, the expected retries per useful output is 1/(1 − 0.573) ≈ 2.3. In a 4-step pipeline, the expected total retries across all steps would be ~9.2 — more than doubling the productive call count. In an iterative agent with 5 loop iterations of 3 steps, the expected retries could reach ~34.

These projections assume independence between retries (the same prompt may produce similar outputs and face the same rejection). The actual retry dynamics are likely more complex and need empirical measurement. This is exactly what drift is built to do: run the same task many times and measure variance.

### 3. Safety-Induced Output Bias

Drift can compare output distributions with and without safety layers to detect the "capability tax":

- **Task success rate**: For code generation tasks, what percentage of outputs compile and pass tests — with vs. without guardrails?
- **Semantic drift**: Using embedding-based similarity, do safety-guarded outputs cluster toward "safer" but more generic responses? Do they contain more hedging language, more disclaimers, more verbose explanations?
- **Output distribution shift**: Does adding guardrails reduce the diversity of outputs (lower num_unique, higher exact_match_rate) in ways that indicate the model is being steered toward a narrow "safe" mode?

**Concrete experiment**: Run drift's `iterative_refactor` task on the same model with three configurations: (a) no guardrails, (b) Llama Guard output check, (c) full LlamaFirewall stack. Compare:
- Iteration count (does the agent take more loops with guardrails?)
- Code quality of final output (does the refactored code actually fix the issues?)
- Output consistency across runs (does adding safety reduce or increase drift?)

A system with 90% safety rate but 40% task success is worse than one with 70% safety and 75% task success — but no tool currently measures both dimensions simultaneously.

### 4. Loop-Aware Overhead Measurement

Drift's iterative_refactor pipeline already measures per-iteration metrics across runs. Extending this to track:
- Total LLM calls per run (productive + safety + retries)
- Latency breakdown per call category
- Cumulative safety overhead as iterations increase
- Whether safety overhead grows linearly with iterations or compounds

...would provide the first published data on compound overhead in looping agents (gap #5). The infrastructure already exists; the main work is instrumenting the safety layers.

### 5. Safety-Compute Efficiency Scoring

A composite metric that captures the return on safety investment:

```
SROI = Breach Cost Prevented / (Safety Compute Cost + Human Ops Cost + Capability Tax)
```

The individual components:
- **Breach cost prevented**: requires threat benchmarks and detection rate data (from security researchers, not drift)
- **Safety compute cost**: measurable by drift (latency overhead, retry count, GPU-hours)
- **Human ops cost**: requires enterprise operational data (person-hours per 1,000 inferences)
- **Capability tax**: measurable by drift (task success rate delta with vs. without safety layers)

Drift can provide the denominator — the total cost of achieving a safety configuration — while security researchers provide the numerator. Together, this would give enterprises a way to compare configurations: "Configuration A catches 90% of threats at $X per 1,000 inferences; Configuration B catches 95% at $3X. Is the marginal 5% worth it?"

This metric does not exist today. Defining and validating it would require collaboration between the drift project (cost measurement) and security research teams (threat benchmarks and detection evaluation).

## Summary

The AI safety infrastructure stack adds measurable overhead at every layer, but no published study aggregates the total cost across a complete agentic deployment. Available per-layer data suggests overhead ranges from ~10% (parallel guardrails on simple requests) to significantly more when utility loss, retries, layer interactions, human operations, and iterative agent workflows are factored in. The upper bound is unknown.

The economic case for safety is clear at the breach-prevention level ($2.1M average cost reduction). The open question is not whether to invest in safety, but whether the current approach — stacking LLM-based safety judges on top of LLM-based agents — is the most compute-efficient way to achieve it. Alternative approaches like constrained decoding (preventing bad outputs at generation time), execution-based verification (sandbox testing), intrinsic safety (training-time alignment), or self-hosted deterministic inference (eliminating variability at the source) may offer better cost-efficiency ratios for certain use cases, but no comparative data exists.

The teams that achieve the best outcomes in enterprise AI will not necessarily be those with the most safety layers. They will be those with the highest ratio of useful work to safety cost. Measuring that ratio requires tools that can observe agent behavior end-to-end — across retries, loop iterations, safety rejections, and output quality — not just per-layer latency benchmarks.

Drift is positioned to provide these measurements, particularly around the retry death spiral, safety-induced capability degradation, and compound overhead in iterative agent workflows. These measurements would not replace safety evaluation, but would provide the cost side of the cost-benefit equation that is currently missing.

The strongest implicit finding from drift's existing data may be the simplest: self-hosted inference on Ollama at temp=0 produces 100% deterministic output with zero guardrail overhead. Cloud inference on DeepInfra, using the same model family, introduces measurable drift even at temp=0. Future work should compare the total cost of self-hosted deterministic inference — where safety controls are transparent and tunable — against cloud inference with opaque, provider-managed guardrails. That comparison may redefine the conversation about what "safe AI" actually costs.
