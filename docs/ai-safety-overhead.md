# The Cost of Making AI Safe: Measuring the Safety Tax

## Overview

As enterprises deploy LLM-powered agents into production, a growing stack of safety infrastructure is required around every inference call: guardrails, content moderation, alignment checks, constrained decoding, and audit logging. Each layer adds latency, compute cost, and operational complexity.

This document surveys available data on the overhead these layers introduce, identifies gaps in the research, and proposes how drift measurements could contribute to understanding the true cost of safe AI deployment.

## The Safety Stack

A production agentic AI call in a regulated enterprise may pass through some or all of the following layers:

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

**2026 update**: The NeMo Guardrails team reports a **~40% reduction in baseline overhead** compared to the 2025 version. Per-layer latency is now estimated at **10–50ms** (down from 100–300ms typical), with optimized NVIDIA infrastructure achieving 50–150ms total across multiple layers. However, SmoothLLM-style schemes still add up to 10 extra inferences per prompt for robustness, and dialog rail complexity (Colang scripts, intent recognition) can re-introduce LLM calls that inflate the overhead beyond the per-layer estimates.

Sources:
- [NVIDIA Technical Blog — Measuring Effectiveness and Performance of AI Guardrails](https://developer.nvidia.com/blog/measuring-the-effectiveness-and-performance-of-ai-guardrails-in-generative-ai-applications/)
- [Guardrails AI vs NeMo Guardrails: Complete Comparison 2026](https://is4.ai/blog/our-blog-1/guardrails-ai-vs-nemo-guardrails-comparison-2026-352)

### Llama Guard / LlamaFirewall

Meta's Llama Guard 3-1B achieves a 76% threat detection rate with **0.165 seconds latency per check** on NVIDIA A30 GPUs. In practical deployments, latency is typically reported as **200–400ms** per safety check. The INT4-quantized 1B variant can run on mobile CPUs at ~30 tokens/second with a 2.5s time-to-first-token.

For comparison, API-based alternatives like Lakera Guard achieve 0.066s per prompt, Azure Prompt Shield 0.349s, and Calypso AI Moderator 0.25s.

Meta's LlamaFirewall (May 2025) composes three layers:
1. **PromptGuard 2** — lightweight BERT classifier for prompt injection (~86M parameters)
2. **AlignmentCheck** — chain-of-thought reasoning auditor (higher compute, lower ASR)
3. **CodeShield** — static analysis across 8 languages (96% precision, 79% recall)

Combined, these three layers reduced attack success rate from 17.6% to 1.75% (a 90% reduction), but at a utility drop to 42.7% — meaning the safety stack rejected nearly 60% of legitimate outputs.

**Open question**: The 42.7% utility figure is striking. If safety layers reject more than half of productive work, the effective compute cost per useful output more than doubles — but only if rejected outputs must be regenerated. The actual cost depends on the retry policy: does the system regenerate, fall back to a human, or return a canned response? No published study has measured the retry overhead or the downstream cost of false rejections.

**2025–2026 update**: Llama Guard 4 (12B) is now available, and independent evaluations have revealed accuracy concerns. On the AEGIS safety dataset, LlamaGuard-3 variants achieved only **22–34% in-domain accuracy** — substantially below production standards. Lightweight alternatives like Sentence-BERT classifiers (67M parameters) now achieve comparable or better moderation efficacy (AUPRC 0.946, F1 0.89) with **sub-50ms latency**, suggesting the full LLM-based guard may be over-engineered for many use cases.

**Gateway integration** is also a practical bottleneck. HAProxy testing on AWS g6.xlarge found that adding LLM-based prompt security (LlamaGuard, PromptGuard, ShieldGemma) reduced throughput from **100,000+ requests/second to ~60 requests/second** — a three-orders-of-magnitude degradation that makes LLM-based guards impractical for high-throughput gateway deployment.

**AprielGuard** (ServiceNow, Dec 2025) introduces a two-stage design with reasoning-enabled and reasoning-free modes, offering a practical latency/explainability tradeoff. The reasoning-free mode is faster but less interpretable; the reasoning-enabled mode provides chain-of-thought justification at higher cost.

Sources:
- [Llama Guard 3-1B-INT4 (arXiv)](https://arxiv.org/html/2411.17713v1)
- [LlamaFirewall (arXiv)](https://arxiv.org/abs/2505.03574)
- [Benchmarking Llama Model Security Against OWASP Top 10](https://arxiv.org/html/2601.19970v1)
- [HAProxy — Lessons Learned in LLM Prompt Security](https://www.haproxy.com/blog/lessons-learned-in-llm-prompt-security-securing-ai-with-ai)
- [AprielGuard (arXiv)](https://arxiv.org/html/2512.20293v1)

### SGLang Constrained Decoding

SGLang's compressed finite-state machine approach enables 3x faster JSON decoding compared to naive constrained decoding. The overhead depends on schema complexity:

- **Simple schemas**: 5–15% latency increase
- **Complex schemas**: 30–60% latency increase
- **With cached compiled grammars**: <10% overhead

SGLang hides much of this cost by overlapping CPU-bound grammar computation with GPU inference. By contrast, vLLM shows significant performance degradation with guided decoding at batch sizes of 8+.

Constrained decoding serves both safety (preventing malformed output) and reproducibility (enforcing structural consistency). It operates at the token level during generation, unlike post-hoc guardrails that evaluate completed output.

**2025–2026 update**: The overhead picture has improved significantly. XGrammar (now the default backend in SGLang, vLLM, and TensorRT-LLM) approaches **near-zero overhead** for cached schemas. More surprisingly, constrained decoding can actually **reduce** latency in some cases — by simplifying the next-token prediction space and skipping generation of structural scaffolding tokens.

SqueezeBits benchmarks (2025, Qwen3-8B/32B) comparing grammar backends found that **LLGuidance consistently outperforms XGrammar** when every request introduces a unique schema, because XGrammar's caching strategy is neutralized. Invalid JSON rates: XGrammar 2.21%, LLGuidance 0.12%.

Without any guided decoding, models already produce schema-valid output 90–94% of the time. Guided decoding pushes this to 96–98.2% — a meaningful improvement but one that raises the question of whether the overhead is justified when the base rate is already high. For drift's use case, the answer is clear: even a 6% structural violation rate at high volume produces thousands of malformed outputs that break downstream pipelines.

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

- **ShieldGemma prompt sensitivity (2025)**: ShieldGemma-2B showed **20.7 percentage point performance degradation** with certain prompt styles — suggesting it relies on specific prompt structures rather than understanding harmful content. Top-performing guard models like Qwen3Guard-8B varied only 1.0 percentage points across all seven prompt styles.

These findings do not mean guardrails are worthless — they raise the bar for most attacks and provide defense-in-depth. But they complicate the cost-benefit calculation: the safety stack's compute cost is measurable, while its protection is probabilistic and bypassable.

Sources:
- [When the Statistical Tail Wags the Average Dog](https://medium.com/@johneverettbutton/when-the-statistical-tail-wags-the-average-dog-a-warning-for-ai-risk-managers-part-1-7f0a104140c9)
- [Proof-of-Guardrail in AI Agents (arXiv)](https://arxiv.org/html/2603.05786)
- [Adversarial Prompt Evaluation: Systematic Benchmarking of Guardrails (arXiv)](https://arxiv.org/pdf/2502.15427)

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

The industry frames safety costs against **adversarial threats**: breaches, prompt injection, data exfiltration. The numbers are real:

- Enterprise AI breach costs average **$10.22M** in the US (2025).
- Organizations with AI-specific controls reduced breach costs by **$2.1M** on average.
- 97% of AI-related breaches in 2025 occurred in environments without access controls.

Source: [Deploying AI Guardrails at Production Scale](https://introl.com/blog/ai-safety-infrastructure-guardrails-production-scale-2025)

But this framing misses a category of damage that is arguably more common and equally costly: **self-inflicted operational damage** from agents making mistakes. No attacker is needed when an agent with tool access can:

- Execute a `DELETE FROM users` without a WHERE clause
- Deploy untested code to production
- Overwrite a configuration file with malformed output
- Send a confidential document to the wrong recipient
- Modify infrastructure in a way that causes a cascading outage
- Introduce a subtle data corruption bug that goes undetected for weeks

These are not adversarial attacks. They are the predictable consequence of giving an LLM — which produces probabilistic output — write access to production systems. The risk increases directly with the capabilities granted to the agent: more tools, more autonomy, more potential for self-inflicted damage.

**The data gap is significant.** The $10.22M breach cost figure captures externally-caused incidents. There is no equivalent published figure for the cost of agent-caused operational incidents — database restorations, rollbacks, service outages, data recovery. These costs are likely absorbed into general incident management budgets and not attributed to AI safety failures. Anecdotally, they may be more frequent than adversarial breaches: most enterprises deploying agents will experience agent errors before they experience targeted attacks.

**Different guardrail layers address different threat types:**

| Threat | Adversarial (breach) | Operational (self-inflicted) |
|--------|---------------------|------------------------------|
| Input guardrails (jailbreak detection) | Yes | No |
| Output guardrails (content safety) | Yes | Partially |
| Safety judge LLM | Yes | Partially |
| Constrained decoding | Partially | Yes (prevents malformed output) |
| Tool-level permissions (RBAC, MCP) | Yes | **Yes** |
| Sandbox/dry-run execution | Partially | **Yes** |
| Human approval for write operations | Yes | **Yes** |

The most effective defenses against self-inflicted damage are not LLM-based at all. They are traditional software engineering practices applied to agent tool access: read-only defaults, RBAC, sandboxed execution, dry-run modes, and human approval gates for destructive operations. These are cheaper than LLM safety judges and more reliable — a permission check that blocks `DROP TABLE` does not have a false negative rate.

**The connection to drift**: Output non-determinism is itself an operational risk. If an agent produces different code on different runs (as drift measures), some of those variants may be correct and others may introduce bugs. On self-hosted Ollama at temp=0, drift shows 100% identical output — the agent will always do the same thing, whether right or wrong. On DeepInfra, the 40% exact match rate on code generation means the agent's behavior varies across invocations. In a system with write access, this variability is not just a quality concern — it is a safety concern. An agent that produces correct code 80% of the time and subtly buggy code 20% of the time is more dangerous than one that is consistently wrong, because the inconsistency makes testing unreliable.

### The reviewable artifact as a zero-cost safety layer

The most effective defense against operational damage may not involve any compute overhead at all. It's the **reviewable artifact** — the playbook, the pull request, the configuration diff — that sits between agent intent and execution.

In traditional infrastructure automation (Ansible, Terraform, CI/CD pipelines), the workflow produces a declarative artifact — a playbook, a plan, a manifest — that gets committed to git, linted, reviewed, approved in a pipeline, and audited after execution. This artifact is the governance layer. It's what prevents a `user: state=absent` from deleting the wrong user, not because it's idempotent (it is), but because a human or automated check reads it before it runs.

When agents bypass this layer — executing infrastructure operations directly without producing a reviewable artifact — the governance layer disappears. The agent decides which operation to run, with which parameters, against which targets, and the only thing between intent and execution is the agent's own judgment.

**Drift data quantifies this risk.** Our functional equivalence testing (Experiment 3 in [RESULTS.md](RESULTS.md)) found that 1 in 30 code variants (3.3%) produced by an LLM pipeline contained a real bug — a duplicate dictionary key that silently returned wrong data. All 30 variants looked plausible. Only execution against a test suite revealed the defect. In an infrastructure context, that's a 3.3% chance per invocation that the agent's action is subtly wrong — not catastrophically wrong (which would be caught), but wrong in a way that corrupts data or misconfigures a system without immediate symptoms.

The reviewable artifact catches this. A playbook or PR that shows `"id": row[0], "id": row[1]` gets flagged in review. An agent that executes the equivalent operation directly does not.

**Cost profile:**

| Safety Layer | Compute Cost | False Negative Rate | Catches Operational Errors? |
|-------------|-------------|--------------------|-----------------------------|
| LLM safety judge | ~0.2s GPU per check | Nonzero (bypassable) | Partially |
| Reviewable artifact (human review) | 0 GPU | Depends on reviewer | Yes |
| Reviewable artifact (automated lint/test) | ~0 GPU | Near-zero for covered checks | Yes |
| RBAC / permission check | 0 GPU | Zero | Yes (for covered operations) |

The reviewable artifact pattern is particularly relevant for infrastructure automation frameworks that expose agent-callable tools. The right architecture is not "agent executes modules directly" but "agent produces a reviewable plan, human or CI approves, system executes." MCP servers are well-suited to this pattern — the server can require that write operations produce a plan artifact for approval rather than executing immediately.

### The LLMOps overhead multiplier

Enterprise data from 2026 puts a concrete number on the total operational overhead: the typical ratio of **total LLMOps cost to raw API spend is 2.3×–4.1×**. This multiplier includes evaluation, safety, prompt maintenance, monitoring, and reliability engineering. Retry logic and fallback routing alone add 5–15% API cost inflation from provider outages, rate limit errors, and timeout retries.

For organizations using LLM-as-judge for safety evaluation specifically, per-evaluation costs run **$0.03–$0.10**, with recommended traffic sampling rates of 1–5%. Multi-model judge panels (3–5 models with majority vote) reduce biases by 30–40% but cost 3–5× more. At 10,000 monthly evaluations, LLM judges save $50,000–$100,000 vs. human review while maintaining 80% agreement with human evaluators.

Evaluation and QA costs including judge LLM token costs run **$3,000–$15,000 per quarter** for enterprise deployments. This is a standing cost regardless of whether the safety stack prevents any incidents.

Source: [The True Cost of Running Enterprise LLMs in Production (2026 Data)](https://theneildave.in/blog/enterprise-llm-cost-2026.html)

### Cloud guardrail pricing

AWS Bedrock Guardrails provides concrete per-request pricing (post-December 2024 85% price reduction):

- **$0.15 per 1,000 text units** for content filters
- **$0.15 per 1,000 text units** for denied topics
- Sensitive information filters: no additional charge
- A text unit = up to 1,000 characters

Example: 1,000 queries/hour with average 200-char input + 1,500-char output = **$0.90/hour** for content filters + denied topics. AWS describes "small additional latency overhead," with practitioners alerting at ~500ms thresholds.

Notably, Bedrock supports an **async streaming mode** with zero latency impact — content is checked in the background, but unsafe chunks may reach the user before the scan completes. This is an explicit latency-vs-safety tradeoff that illustrates the tension.

Source: [Amazon Bedrock Guardrails reduces pricing by up to 85%](https://aws.amazon.com/about-aws/whats-new/2024/12/amazon-bedrock-guardrails-reduces-pricing-85-percent/)

### KV cache overhead from safety system prompts

Safety system prompts (alignment instructions, content policies, guardrail triggers) are typically the most static part of the prompt — and the most wasteful without caching. A 20,000-token system prompt running over 50 turns produces **1 million tokens of redundant computation**. For Llama 3.1-70B, a single 128K context prompt consumes **~40GB of HBM** just for KV cache.

**Prompt caching largely solves this**: latency reduced by up to **80%**, input token costs reduced by up to **90%** (Anthropic charges cache hits at ~90% less than misses). NVIDIA TensorRT-LLM's early KV cache reuse shares system prompt KV cache across concurrent users, accelerating inference by up to **5x** for system-prompt-heavy workloads.

However, KV cache sharing introduces a privacy tension. PROMPTPEEK attacks demonstrate cross-tenant prompt leakage via shared KV caches. Disabling sharing costs **8–38.9% TTFT overhead**. SafeKV (NDSS 2025) selectively isolates sensitive cache entries while reusing non-sensitive ones.

The safety-specific implication: the KV cache cost of safety system prompts is a solved problem for organizations using prompt caching, but remains a real cost for those who don't — and caching introduces its own security risks.

Sources:
- [ngrok — Prompt Caching](https://ngrok.com/blog/prompt-caching)
- [NVIDIA — 5x Faster TTFT with KV Cache Early Reuse](https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/)
- [SafeKV (OpenReview)](https://openreview.net/pdf?id=jhDsbd5eXL)

### The cost of having safety: compute

- **AI content moderation market**: $1B (2024) → projected $2.6B (2029).
- **Per-request overhead**: ~0.5s (NeMo) + ~0.2s (Llama Guard) per productive call, assuming parallel execution.
- **Throughput cost**: 34% for deterministic inference (reproducibility), near-zero to 60% for constrained decoding (safety + reproducibility).
- **Utility loss**: up to 57.3% of legitimate outputs rejected (LlamaFirewall combined layers).

### The cost of having safety: human operations

The compute overhead is only part of the cost. Safety infrastructure requires ongoing human effort:

- **Guardrail tuning**: engineers must configure and update PII detection patterns, topic boundaries, hallucination thresholds, and prompt injection rules. These require domain expertise and ongoing maintenance as attack patterns evolve.
- **False positive investigation**: when legitimate outputs are blocked (the LlamaFirewall 57.3% rejection problem), engineers must diagnose why. Was it a misconfigured rule? An edge case in the safety model? This debugging time is a direct cost of the safety stack.
- **Human-in-the-loop overrides**: in high-stakes deployments, blocked outputs may require human review before being released or regenerated. This introduces latency measured in minutes or hours, not milliseconds.
- **Monitoring and incident response**: safety layers generate alerts, logs, and metrics that someone must monitor. False alarm fatigue is a known problem in security operations.

No published data quantifies this as a per-inference metric (e.g., person-hours per 1,000 inferences). Enterprises track these costs in aggregate staffing budgets, but the per-inference allocation is not reported. This is a significant blind spot: for some organizations, the human cost of safety may exceed the compute cost.

### What's missing: the ratio

A rough framework for comparing safety cost to damage risk:

- If an enterprise runs 1M inference calls/month and safety adds ~0.7s of GPU time per call, that's ~194 GPU-hours/month of safety compute.
- At $3/GPU-hour (H100 cloud), that's ~$583/month in safety compute.
- Against a $2.1M average breach cost reduction, the ROI is clear at this scale.

**Break-even framing**: At $583/month for 1M calls, safety compute is negligible relative to breach risk. But this assumes the latency-only cost model. If we include the full safety tax — utility loss (57% rejection → retries doubling effective calls), human ops (~0.5 FTE at $135K/year = ~$5,600/month), and throughput reduction (34% for deterministic inference) — the monthly cost looks different:

| Component | Monthly cost estimate | Confidence |
|-----------|---------------------|------------|
| Safety compute (latency) | ~$583 | Vendor-reported (NeMo, Llama Guard benchmarks) |
| Retry overhead (2.3x calls at 57% rejection) | ~$753 | Estimated — derived from LlamaFirewall benchmark, see sensitivity analysis below |
| Deterministic inference throughput loss (34%) | ~$1,836 | Vendor-reported (SGLang documentation) |
| Human ops (0.5 FTE) | ~$5,600 | Estimated — no published per-inference data exists |
| **Total estimated safety cost** | **~$8,772/month** | **Mixed — see confidence column** |

At ~$105K/year, the break-even against a single $2.1M breach cost reduction is ~20 years — the ROI still heavily favors safety investment. But at higher call volumes (100M calls/month, common for large enterprises), the compute and retry components scale to ~$134K/month while breach risk does not scale proportionally, narrowing the margin. And if operational damage from agent errors (database restorations, rollbacks) averages even one $50K incident per month, the cost-of-not-having-safety rises substantially.

These figures are illustrative and combine vendor benchmark data, published cost estimates, and assumptions. They are included to show the structure of the calculation, not to assert precise costs. The point is that the break-even depends heavily on which cost components are included — latency-only analysis understates the safety tax by an order of magnitude.

#### Sensitivity to rejection rate

The 57.3% rejection rate (42.7% utility) from the LlamaFirewall benchmark is the most influential variable in the cost model. It drives the retry overhead directly and compounds in agentic workflows. The table below shows how the economics shift at different rejection rates:

| Rejection rate | Retries per useful output | Retry cost (1M calls) | Total safety cost/month | Break-even vs. $2.1M |
|---------------|--------------------------|----------------------|------------------------|----------------------|
| 15% | 1.18x | ~$104 | ~$8,123 | ~21 years |
| 30% | 1.43x | ~$250 | ~$8,269 | ~20 years |
| 57% (LlamaFirewall) | 2.33x | ~$753 | ~$8,772 | ~20 years |
| 70% | 3.33x | ~$1,360 | ~$9,379 | ~19 years |
| 80% | 5.00x | ~$2,332 | ~$10,351 | ~18 years |

At 1M calls/month, the rejection rate barely moves the needle — human ops dominates the cost. At 100M calls/month (where human ops is amortized), the picture changes: a 15% rejection rate adds ~$10K/month in retries while 57% adds ~$75K and 80% adds ~$233K. The retry death spiral becomes the dominant cost above ~40% rejection.

No published source reports production rejection rates outside of benchmark conditions. The 57% figure may be a worst case (LlamaFirewall with all three layers active against adversarial inputs) or a best case (production workloads may trigger different false-positive patterns). Empirical measurement of rejection rates on real workloads is the single highest-value data point missing from the cost model.

The break-even table above attempts to include these broader costs, but each line item carries uncertainty. Unmeasured factors that could shift the calculation further:
- GPU memory consumed by co-located safety models (reduces capacity for productive inference)
- Compound false-positive rates from stacked guardrails (potentially higher retry costs)
- Capability tax (reduced task success rate requiring human intervention)

Empirical measurement of these components — not just the latency line — is what would make the cost-benefit analysis actionable.

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

### Decoding-time safety

A paradigm shift is emerging: embedding safety checks directly in the decoding loop rather than running separate post-hoc models. This partially dissolves the "safety tax" concept:

- **Kelp** (ACL 2025): a 20M-parameter probe attached to a frozen LM performs streaming token-level risk detection during generation, adding **less than 0.5ms per-token latency** while outperforming post-hoc guardrails by **15.61% higher average F1**.
- **Speculative Safety-Aware Decoding (SSD)** (arXiv:2508.17739): uses a small safety-aligned model as the draft model in speculative sampling, dynamically switching decoding schemes based on jailbreak risk. Key result: **accelerates inference while adding safety** rather than the usual tradeoff.
- **ShieldHead** (ACL 2025): lightweight classification head parallelized with the LM head for token-level safety prediction during decoding, avoiding the latency of a separate safety model call entirely.

These approaches represent a shift from "safety adds overhead" to "safety can be embedded in the decoding loop at sub-millisecond cost." They are not yet production-standard but signal where the field is heading.

Sources:
- [Kelp (ACL 2025 Findings)](https://aclanthology.org/2025.findings-acl.1223.pdf)
- [Speculative Safety-Aware Decoding (arXiv)](https://arxiv.org/abs/2508.17739)

### Intrinsic safety via fine-tuning

Models can be fine-tuned (via RLHF, DPO, or constitutional AI methods) to refuse harmful requests internally, reducing the need for external guardrails. The cost is borne during training rather than at inference time. The tradeoff is that intrinsic safety is harder to audit, update, and customize per-deployment compared to external guardrails. It also contributes to the utility loss problem: an overly-aligned model may refuse legitimate requests.

**The training-time safety tax is now formalized.** Huang et al. (Georgia Tech, March 2025) demonstrated that safety alignment of Large Reasoning Models degrades reasoning capability: "You can't get a safety-aligned model and good reasoning model at the same time." This means intrinsic safety doesn't eliminate the cost — it moves it from inference-time compute to training-time capability degradation. Follow-up work explores mitigations: LoRA-based alignment (smaller weight perturbation), ThinkSafe (self-generated safety reasoning), and DGR (avoiding out-of-distribution distilled data).

Source: [Safety Tax: Safety Alignment Makes Your Large Reasoning Models Less Reasonable (arXiv:2503.00555)](https://arxiv.org/abs/2503.00555)

### The comparison gap

No published study compares the safety tax of these approaches head-to-head for equivalent safety outcomes. Key questions:

- Is the LlamaFirewall 42.7% utility loss inherent to the LLM-as-judge approach, or would sandbox-based verification achieve better utility at lower compute?
- For structured output tasks (JSON, code), does constrained decoding at generation time eliminate the need for post-hoc content checks entirely?
- What is the total cost of intrinsic safety (training-time alignment) vs. extrinsic safety (inference-time guardrails) over a model's production lifecycle?

These comparisons would reframe the question from "how much does safety cost?" to "which safety approach delivers the best return on investment?"

## Drift's Empirical Findings on the Safety Tax

Since this document was first written, drift's experiments have produced several findings directly relevant to the safety tax question:

### Hybrid execution eliminates the safety tax for deterministic fields

Drift's core finding is that many LLM output fields are deterministic — the model copies from input, applies a fixed mapping, or classifies with high confidence. For these fields, drift generates extraction rules (regex, mapping, classifier) that bypass the LLM entirely.

**The safety implication**: fields extracted by deterministic rules bypass not only the productive LLM but also the entire safety stack. A regex extracting `Ticket ID: 1aiu3lrqi` needs no guardrail check, no content moderation, and no safety judge. In the helpdesk triage experiment, 6 of 12 fields were handled by rules — effectively halving the safety tax surface for this task.

The safety-tax reduction is mechanistic, not statistical: rules produce identical output every time, cannot hallucinate, and execute in microseconds. There is nothing for a safety layer to check.

### Pre-digestion shrinks the safety scanning surface

Input pre-digestion (83% token reduction on Ansible logs) reduces what the safety stack must scan. If guardrails run on the pre-digested input rather than the raw input, the guardrail compute cost drops proportionally. For a NeMo Guardrails check that processes ~1,000 tokens of pre-digested input instead of ~6,000 tokens of raw log, the latency and token cost are both lower.

This is more than a performance optimization — it's a safety improvement. Large inputs (100KB+ logs) caused the productive LLM to fail 40% of the time (ignoring instructions, producing rambling narratives). Pre-digestion eliminated these failures entirely. A safety layer evaluating the focused pre-digested output is checking higher-quality content and is less likely to false-reject.

### Model upgrades are safety events

Upgrading from Qwen2.5:32b to Qwen3:30b on identical hardware dropped functional equivalence from **100% to 0%** on the iterative_refactor task. Qwen3 consistently missed a password exposure fix that Qwen2.5 catches every time. This is not drift — it's a model capability regression.

In a system with write access, a model upgrade that silently drops a security fix from 100% to 0% is a safety incident. No guardrail in the standard safety stack detects this: the model's output is well-formed, non-toxic, and structurally valid. It's just functionally wrong.

**The safety implication**: model version pinning is itself a safety control. The standard guardrail stack (content moderation, jailbreak detection, output format validation) evaluates the wrong dimension of safety for production agents. Functional regression testing — running the same task before and after upgrade and comparing outputs — catches what guardrails miss. Drift provides exactly this capability.

### Infrastructure non-determinism compounds safety risk

Drift measures output variability across infrastructure tiers at temperature=0:

| Infrastructure | Exact Match | Functional Eq |
|---------------|-------------|---------------|
| Single-GPU self-hosted (Ollama) | 100% | 100% |
| Shared cloud (DeepInfra) | 90% | 100% |
| Cloud API (Vertex AI) | 30% | 100% |

For safety evaluation, the 30% exact match on cloud APIs means a safety audit of a sample output may not generalize. If 70% of outputs differ textually, a safety judge that approved one variant may reject another — or vice versa. The safety stack itself is being evaluated on a moving target.

Self-hosted inference removes this variable: the output is the same every time, so safety evaluation of one output covers all future outputs on the same input. The 34% throughput cost of deterministic inference may be cheaper than the compound cost of evaluating non-deterministic outputs.

### Attention analysis reveals limits of keyword-based safety

Drift's attention weight analysis (MLX, 50 tickets) found that classification fields use **diffuse attention** — per-word weights of 0.001–0.010, spread across the entire input. The model doesn't classify by spotting keywords; it synthesizes meaning across the full ticket.

This has a direct safety implication: keyword-based guardrails (blocklists, regex filters) cannot detect harmful classification because the model's reasoning is holistic, not keyword-driven. A guardrail that blocks "DELETE" can prevent a harmful SQL statement, but a guardrail that blocks individual words cannot prevent a harmful classification decision because no single word drives the decision.

## Gaps in the Research

The following questions do not have published answers as of April 2026:

1. **Total safety-to-productive compute ratio**: What percentage of GPU-hours in a production agentic deployment is spent on safety vs. productive work? Fragments exist (NeMo adds 0.5s, Llama Guard adds 0.2s, deterministic inference costs 34%) but no end-to-end measurement.

2. **Safety utility loss in practice**: LlamaFirewall's 42.7% utility rate was measured in a specific benchmark. What is the utility loss in real production deployments? What is the cost of re-generation when outputs are rejected? How do false rejection rates vary by domain?

3. **The retry death spiral**: If safety layers reject 57% of outputs, and each retry has a probability of also being rejected, how many retries are needed per useful output? At what rejection rate does the system become uneconomical? This is a multiplicative cost that can turn a "10% latency tax" into a "200%+ effective overhead" problem in agentic workflows.

4. **Diminishing returns on stacked guardrails**: NeMo shows diminishing latency per additional layer, but does detection quality improve proportionally? Is the marginal value of the 4th or 5th guardrail worth its compute? And do stacked layers interact negatively — increasing false rejections through compound probability?

5. **Safety overhead in iterative/looping agents**: When an agent loops 5 times with 3 inner steps, the safety stack may execute 45+ checks per user action. No study has measured this compound overhead in a real deployment.

6. **Constrained decoding vs. post-hoc guardrails**: These are fundamentally different approaches — prevention at generation time vs. detection after generation. Constrained decoding has a per-token cost but zero false rejections for structural violations. Post-hoc guardrails have a per-output cost with nonzero false rejection rates. No study has compared their compute efficiency for equivalent safety outcomes.

7. **Self-hosted vs. cloud + guardrails total cost of ownership**: Drift's data shows self-hosted Ollama produces 100% deterministic output at temp=0 with no guardrail overhead. But self-hosting has its own costs: hardware ($20K–$200K+ for GPU servers), operations staff (~$135K/year for MLOps engineers), power, cooling, and compliance infrastructure. No published comparison weighs the total cost of self-hosted deterministic inference against cloud inference plus the guardrails needed to compensate for cloud non-determinism.

    A compounding factor: managed API providers (OpenAI, Anthropic, DeepInfra) apply their own guardrails that customers cannot measure, disable, or control. These provider-side guardrails add latency that appears as model inference time, reject outputs that the customer's own guardrails would have accepted, and may bias outputs in ways that are invisible to downstream measurement. When an enterprise adds NeMo Guardrails on top of a provider that already runs its own content filters, the customer is paying for two safety stacks — one visible, one opaque — with no way to measure redundancy or interaction effects. Any cost-benefit analysis of self-hosted vs. cloud must account for this: the cloud safety tax includes not just the customer's guardrails but the provider's as well, and the provider's contribution to latency, utility loss, and output bias is unmeasurable from outside.

8. **Safety-induced capability degradation**: Do safety layers make agents worse at their jobs, beyond just blocking outputs? If guardrails bias the output distribution toward "safer" but less useful responses (more verbose, more hedging, avoiding edge cases), the effective cost includes reduced task success rate — not just latency and rejection. This "capability tax" is distinct from the compute tax and may be harder to detect.

9. **Cost of agent-caused operational damage**: The $10.22M breach cost figure captures adversarial incidents. What is the equivalent figure for self-inflicted damage from agent errors — database restorations, rollbacks, bad deployments, data corruption? This data is not published or attributed to AI safety failures, but it is not invisible. Enterprises can approximate it from existing systems:
    - **Incident postmortems**: organizations using blameless postmortem processes (e.g., PagerDuty, Jeli) can tag incidents caused by automated/AI systems and extract cost-per-incident estimates.
    - **Change management records**: ITSM platforms (ServiceNow, Jira Service Management) track change failures and rollbacks. Filtering for AI-initiated changes would yield a failure rate and mean-time-to-recovery.
    - **Database restoration costs**: backup/restore frequency, RTO (recovery time objective), and the labor cost of data reconciliation after a bad write are measurable from existing infrastructure metrics.
    - **Industry benchmarks**: Gartner and Forrester publish cost-per-incident figures for IT operations ($5K–$50K for minor incidents, $100K+ for major outages). Applying these to agent-initiated incident rates would give a rough operational damage estimate.

    Without this data, the SROI of guardrails against operational damage cannot be calculated. But the data sources exist — they just have not been aggregated for AI agent deployments specifically.

10. **LLM-based vs. permission-based safety cost comparison**: For preventing operational damage (destructive tool use), how does the cost of an LLM safety judge (which must infer intent from output content) compare to a database permission check or RBAC rule (which deterministically blocks the operation)? The permission-based approach has zero false negatives and zero inference cost, but is less flexible. No study has measured where LLM-based guards add value beyond what access controls already provide.

## Where Drift Can Contribute

The drift project is positioned to measure several aspects of the safety tax that existing tools do not:

### 1. Quantifying the Reproducibility Cost Curve (partially complete)

Drift has measured output variability across three infrastructure tiers:
- **Self-hosted inference** (Ollama qwen2.5:32b): **100% exact match** at temp=0
- **Shared cloud inference** (DeepInfra qwen2.5:72b): **90% exact match** on extraction, 40% on code generation
- **Cloud API** (Vertex AI Claude Sonnet): **30% exact match** on extraction

What remains unmeasured:
- **Constrained decoding** (SGLang with JSON grammar enforcement)
- **Deterministic inference** (SGLang with `--enable-deterministic-inference`)

The cost curve so far shows that self-hosted single-GPU inference achieves perfect determinism at temp=0 with zero guardrail overhead, while cloud APIs introduce textual drift that is functionally harmless but complicates safety auditing — a sample safety evaluation may not generalize to 70% of outputs that differ textually.

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

**Integration feasibility**: The safety tools vary in how easily they can be plugged into drift's pipeline:
- **Llama Guard**: Available as a standard model on Ollama and HuggingFace. Can be integrated as a post-step check in drift's pipeline executor — call the model with the output text, parse the safe/unsafe classification. Low integration effort.
- **PromptGuard 2**: A small BERT classifier (~86M params). Available on HuggingFace. Can run locally with minimal GPU overhead. Straightforward to wrap as a pre-step input check. Low integration effort.
- **CodeShield**: Open-source static analysis tool. Runs as a subprocess on generated code — no GPU needed. Can be added to drift's `post_process` step mechanism. Low integration effort.
- **NeMo Guardrails**: Requires the NeMo Guardrails runtime and configuration. More complex integration — the framework expects to orchestrate the entire LLM call, not just evaluate an output. Would require wrapping drift's provider calls inside NeMo's execution model. Medium-high integration effort.
- **Full LlamaFirewall stack**: Combines PromptGuard 2 + AlignmentCheck + CodeShield. Available as a Python package (`llamafirewall`). AlignmentCheck requires chain-of-thought auditing, which is the most compute-intensive component. Medium integration effort — the package provides a unified API, but AlignmentCheck adds significant latency per check.

The lowest-effort experiment would compare (a) no guardrails vs. (b) Llama Guard output check vs. (c) Llama Guard + CodeShield — all of which can be integrated as post-step hooks in drift's existing pipeline executor without architectural changes.

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
SROI = Damage Prevented / (Safety Compute Cost + Human Ops Cost + Capability Tax)
```

The numerator — **damage prevented** — must account for both threat categories:
- **Adversarial damage**: breach cost reduction, data exfiltration prevention (requires security researcher threat benchmarks)
- **Operational damage**: avoided database restorations, rollbacks, service outages, data recovery from agent errors (requires enterprise incident data)

The denominator — **total safety cost** — has three components:
- **Safety compute cost**: measurable by drift (latency overhead, retry count, GPU-hours)
- **Human ops cost**: requires enterprise operational data (person-hours per 1,000 inferences)
- **Capability tax**: measurable by drift (task success rate delta with vs. without safety layers)

Drift can provide the denominator while security researchers and enterprise ops teams provide the numerator. Together, this would give enterprises a way to compare configurations: "Configuration A catches 90% of threats and prevents 95% of destructive operations at $X per 1,000 inferences; Configuration B catches 95% of threats but allows 10% of destructive operations through at $2X."

Critically, the cheapest guardrails against operational damage may not be LLM-based at all. A `REVOKE DELETE ON production.users FROM agent_role` costs zero inference compute and has zero false negatives. Measuring the SROI of permission-based controls against LLM-based content analysis would clarify where each approach is most cost-effective.

This metric does not exist today. Defining and validating it would require collaboration between the drift project (cost measurement) and security research teams (threat benchmarks and detection evaluation).

## Summary

The AI safety infrastructure stack adds measurable overhead at every layer, but no published study aggregates the total cost across a complete agentic deployment. Available per-layer data suggests overhead ranges from ~10% (parallel guardrails on simple requests) to significantly more when utility loss, retries, layer interactions, human operations, and iterative agent workflows are factored in. The upper bound is unknown.

The economic case for safety is clear at the breach-prevention level ($2.1M average cost reduction). But the breach framing understates the problem: self-inflicted operational damage from agent errors — database restorations, bad deployments, data corruption — may be more frequent than adversarial attacks and is not captured in breach cost statistics. The most effective defenses against operational damage are often traditional access controls (RBAC, read-only defaults, human approval gates) rather than LLM-based safety judges.

The open question is not whether to invest in safety, but whether the current approach — stacking LLM-based safety judges on top of LLM-based agents — is the most compute-efficient way to achieve it. Alternative approaches like constrained decoding (preventing bad outputs at generation time), execution-based verification (sandbox testing), intrinsic safety (training-time alignment), permission-based tool restrictions, or self-hosted deterministic inference (eliminating variability at the source) may offer better cost-efficiency ratios for certain use cases, but no comparative data exists.

The teams that achieve the best outcomes in enterprise AI will not necessarily be those with the most safety layers. They will be those with the highest ratio of useful work to safety cost. Measuring that ratio requires tools that can observe agent behavior end-to-end — across retries, loop iterations, safety rejections, and output quality — not just per-layer latency benchmarks.

Drift is positioned to provide these measurements, particularly around the retry death spiral, safety-induced capability degradation, and compound overhead in iterative agent workflows. These measurements would not replace safety evaluation, but would provide the cost side of the cost-benefit equation that is currently missing.

Drift's existing data already demonstrates a finding relevant to operational safety: self-hosted inference on Ollama at temp=0 produces 100% identical output across 30 runs. Cloud inference on DeepInfra, using the same model family at temp=0, produces 2 distinct code variants across 5 runs (40% exact match). In a system where the agent has write access to production, this means the agent's behavior is unpredictable on cloud infrastructure — it might produce correct code on one invocation and subtly buggy code on the next. Drift can quantify this risk directly: what percentage of output variants would cause operational damage if executed?

Future work should compare the total cost of self-hosted deterministic inference — where safety controls are transparent and tunable — against cloud inference with opaque, provider-managed guardrails. And it should compare the cost of LLM-based safety judges against simple access controls (RBAC, read-only defaults) for preventing operational damage. The cheapest guardrail may not involve a GPU at all.
