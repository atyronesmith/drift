# WEB-SEARCH.md — Literature Review: Related Work on LLM Inference Variability

Search conducted 2026-03-05.

---

## Summary

34 papers found across 8 topic areas. The space is active but the specific intersection — **infrastructure-induced variance compounding across multi-agent pipelines** — is largely unoccupied. Existing work measures either single-call nondeterminism or agent-level behavioral drift over long interactions. Nobody has systematically measured how per-step infrastructure variance compounds through a structured multi-agent pipeline, or how that compounding differs across infrastructure tiers (shared cloud vs. self-hosted serving stack vs. dedicated sequential inference).

---

## 1. LLM Inference Nondeterminism from Infrastructure / GPU / Batch Effects

**"Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference"**
Yuan et al., NeurIPS 2025. arXiv:2506.09501
Under bfloat16 with greedy decoding, DeepSeek-R1-Distill-Qwen-7B shows up to 9% accuracy variation and 9,000 tokens difference in response length due to GPU count, type, and batch size. Proposes LayerCast (FP32 compute with 16-bit weight storage) as mitigation.

**"Defeating Nondeterminism in LLM Inference"**
Thinking Machines Lab (Mira Murati), 2025. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
Identifies batch-size variation (not floating-point arithmetic alone) as the primary cause of nondeterminism. Server load determines batch size, which changes results for individual requests. Qwen3-235B at temperature=0 produced 80 distinct completions across 1,000 runs. Batch-invariant kernels achieve bitwise-identical outputs at ~60% performance cost.

**"DefensiveKV: Taming the Fragility of KV Cache Eviction in LLM Inference"**
2025. OpenReview: https://openreview.net/forum?id=nJgS06sX3O
Shows that KV cache eviction under the "stability assumption" is inherently fragile, and mean aggregation of importance scores is vulnerable in extreme cases, leading to output quality degradation.

**"NACL: A General and Effective KV Cache Eviction Framework for LLM at Inference Time"**
ACL 2024. https://aclanthology.org/2024.acl-long.428/
Proposes Proxy-Tokens Eviction with Random Eviction to address attention bias in KV cache eviction that degrades output quality.

---

## 2. Multi-Agent System Reproducibility and Drift

**"Agent Drift: Quantifying Behavioral Degradation in Multi-Agent LLM Systems Over Extended Interactions"**
Abhishek Rath, January 2026. arXiv:2601.04170
Introduces "agent drift" with three manifestations: semantic drift, coordination drift, and behavioral drift. Found semantic drift in nearly half of multi-agent workflows by 600 interactions. Focuses on emergent behavioral drift rather than infrastructure-induced variance.

**"Multi-Agent LLM Orchestration Achieves Deterministic, High-Quality Decision Support for Incident Response"**
November 2025. arXiv:2511.15755
Over 348 controlled trials, multi-agent systems achieved 100% actionable recommendation rate and zero quality variance vs. single-agent systems with 1.7% actionable rate and inconsistent outputs. Claims determinism through orchestration design.

**"Large Language Models Miss the Multi-Agent Mark"**
2025. arXiv:2505.21298
Argues that intrinsic non-determinism of LLMs fundamentally flaws multi-agent system assumptions of deterministic environments. Even temperature=0 or deterministic sampling does not solve this.

**"ReliabilityBench: Evaluating LLM Agent Reliability Under Production-Like Stress Conditions"**
January 2026. arXiv:2601.06112
Measures consistency (pass_k across repeated runs), robustness (paraphrased instructions), and fault tolerance (infrastructure failures). Finds pass@1 overestimates production reliability by 20-40%.

### Note: Contradictory Claims on Multi-Agent Determinism

The incident response paper (2511.15755) claims "zero quality variance" across 348 trials. This directly contradicts both 2505.21298 (LLMs fundamentally undermine multi-agent determinism assumptions) and 2601.06112 (pass@1 overestimates reliability by 20-40%). The contradiction is likely an artifact of measurement granularity:

- The incident response paper measures at a **binary level** — "actionable vs. not actionable." Two recommendations with completely different text, reasoning paths, and specific actions both score as "actionable." Zero variance at the binary level says nothing about variance at the text, semantic, or reasoning level.
- The orchestration design likely **constrains the output space** heavily — structured fields with limited options (severity levels, action categories). The determinism is in the constraints around the LLM, not in the LLM itself. Guardrails mask stochasticity rather than eliminate it.
- **Infrastructure conditions are uncontrolled/unreported** — if all 348 trials ran on the same hardware during the same time window, they measured the variance floor and called it the ceiling.

This tension directly motivates the measurement hierarchy in MEASURE.md. A system can show "zero variance" on Layer 4 (functional equivalence — did it produce an actionable output?) while showing significant variance on Layer 1 (edit distance) and Layer 3 (embedding similarity). Both findings are true simultaneously. The gap between coarse-grained functional metrics and fine-grained output metrics is exactly what makes binary evaluation dangerous for production systems where the specific content of the output matters, not just its category.

---

## 3. LLM Output Drift Over Time

**"How Is ChatGPT's Behavior Changing over Time?"**
Lingjiao Chen, Matei Zaharia, James Zou. 2023/2024. arXiv:2307.09009
Foundational paper. GPT-4 accuracy on prime number identification dropped from 84% (March 2023) to 51% (June 2023). Demonstrated that the "same" LLM service can change substantially in a short period with no disclosed model version change.

**"Quantifying Non-Deterministic Drift in Large Language Models"**
Claire Nicholson, January 2026. arXiv:2601.19934
Empirically quantifies baseline behavioral drift in gpt-4o-mini and llama3.1-8b across five prompt categories. Non-determinism persists even at temperature 0.0, with distinct variability patterns by model size, deployment type, and prompt type.

**"LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows"**
November 2025. ACM ICAIF '25 (AI4Finance workshop). arXiv:2511.07585
Financial-sector focus. Tier 1 models (Granite-3-8B, Qwen2.5:7B) maintain perfect consistency; GPT-OSS-120B exhibits severe nondeterminism.

**"Evaluating Performance Drift from Model Switching in Multi-Turn LLM Systems"**
March 2026. arXiv:2603.03111
Introduces a switch-matrix benchmark measuring handoff-induced drift when one model continues another's conversation in production routing/fallback scenarios.

---

## 4. Measuring Variability in Agentic Pipelines

**"A Practical Guide for Evaluating LLMs and LLM-Reliant Systems"**
June 2025. arXiv:2506.13023
Addresses error compounding in multi-turn agentic workflows. Variability arises from stochastic computations, chained model calls, and grounding data/API inconsistencies.

**"Sherlock: Reliable and Efficient Agentic Workflow Execution"**
2025. arXiv:2511.00330
Explores trade-offs between cost, accuracy, and latency in agentic workflows. Verification incurs up to 28.9x latency and 53.2x cost overhead.

**"Evaluation and Benchmarking of LLM Agents: A Survey"**
KDD 2025. arXiv:2507.21504
Comprehensive taxonomy of LLM agent evaluation. Notes that agents are inherently probabilistic and behave dynamically, requiring new evaluation approaches beyond deterministic software testing.

**SWE-rebench**
2025. https://swe-rebench.com/about
Runs each model five times on the full benchmark and reports standard error of the mean (SEM) and pass@5, directly addressing run-to-run variance in agentic coding benchmarks.

---

## 5. Infrastructure-Induced Quality Degradation in LLM Serving

**"BurstGPT: Towards Efficient and Reliable LLM Serving — A Real-World Workload Study"**
2024. arXiv:2401.17644
Uncovers vulnerability of LLM serving systems to short-term burstiness. GPU memory limitations from fluctuating burstiness lead to significant performance degradation, resource contention, and massive request failures.

**"When LLMs Get Significantly Worse: A Statistical Approach to Detect Model Degradations"**
February 2025. arXiv:2602.10144
Even theoretically lossless optimizations (changing hardware or serving configuration) cause "flips" due to 16-bit activations introducing harmless but consequential numerical errors.

**"Niyama: Breaking the Silos of LLM Inference Serving"**
March 2025. Microsoft Research. https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/niyama.pdf
QoS-driven serving system with hybrid prioritization and eager relegation for graceful degradation during overload. Increases capacity by 32% over siloed vLLM deployments.

**"MorphServe: Efficient and Workload-Aware LLM Serving via Runtime Layer Swapping and KV Cache Resizing"**
May 2025. arXiv:2506.02006
Static quantization causes persistent accuracy degradation; dynamic workloads with load spikes cause memory exhaustion and SLO violations in vLLM and TGI.

**"Efficient Mixed-Precision LLM Inference with TurboMind"**
August 2025. arXiv:2508.15601
Shows TensorRT-LLM fails to fully leverage the hardware memory hierarchy. Achieves up to 61% lower latency and 156% higher throughput than TensorRT-LLM and vLLM+MARLIN.

---

## 6. Stochastic Variability in Reasoning Chains

**"Self-Consistency Improves Chain of Thought Reasoning in Language Models"**
Wang et al., 2022. arXiv:2203.11171
Foundational paper. Complex reasoning tasks admit multiple valid reasoning paths. Sampling diverse paths and marginalizing yields +17.9% accuracy on GSM8K. Implicitly demonstrates that CoT outputs are stochastic even for the same input.

**"Thought Branches: Interpreting LLM Reasoning Requires Resampling"**
October 2025. arXiv:2510.27484
Reasoning models perform computations stochastically. Interpreting a single CoT trace is fundamentally inadequate — the distribution of possible CoTs must be studied.

**"Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens"**
2025. arXiv:2508.01191
CoT reasoning reproduces training-distribution patterns but suffers significant degradation under distributional deviations, revealing brittleness.

**"Prompt-Reverse Inconsistency: LLM Self-Inconsistency Beyond Generative Randomness and Prompt Paraphrasing"**
April 2025. arXiv:2504.01282
Documents up to 10% variation in output accuracy even under deterministic settings across repeated identical inputs.

---

## 7. Reproducibility of LLM-Based Systems

**"Reflections on the Reproducibility of Commercial LLM Performance in Empirical Software Engineering Studies"**
Florian Angermeir et al., October 2025. ICSE 2026. arXiv:2510.25506
Of 18 LLM-centric studies from ICSE/ASE 2024 with research artifacts, only 5 were executable. None were fully reproducible. Extrinsic time-related factors (silent model updates) critically impact performance.

**"Large Language Models for Software Engineering: A Reproducibility Crisis"**
Mohammed Latif Siddiq et al., November 2025. arXiv:2512.00651
First large-scale study of reproducibility practices across 640 papers (2017-2025). Defines a taxonomy of seven "reproducibility smell" categories.

**"Non-Determinism of 'Deterministic' LLM Settings"**
2024-2025. arXiv:2408.04667
First systematic investigation of LLM instability under maximally deterministic hyperparameters. Introduces TARr@N and TARa@N metrics for quantifying determinism.

**"Measuring Determinism in Large Language Models for Software Code Review"**
February 2025. arXiv:2502.20747
Tests GPT-4o, GPT-4o mini, Claude, and LLaMA at temperature=0 for code review. LLMs produce different outputs for identical inputs. LLM inconsistency may be comparable to human reviewer variability.

**"Replication for Language Models"**
Barrie, Palmer & Spirling, 2025. https://arthurspirling.org/documents/BarriePalmerSpirling_TrustMeBro.pdf
LLM choices have downstream effects on substantive results and statistical significance. Open-weight models may provide a path forward for replication.

---

## 8. Floating-Point Nondeterminism in GPU Inference

**"Impacts of Floating-Point Non-Associativity on Reproducibility for HPC and Deep Learning Applications"**
Shanmugavelu et al. (Oak Ridge National Laboratory), 2024. arXiv:2408.05148
Systematic analysis of variability from non-deterministic kernels for parallel reductions, PyTorch functions, and full DL pipelines. Evaluates deterministic hardware (Groq LPU) as a mitigation strategy.

**"On the Structure of Floating-Point Noise in Batch-Invariant GPU Matrix Multiplication"**
October 2025. arXiv:2511.00025
Challenges the assumption that GPU floating-point errors are i.i.d. Gaussian noise. For float16, nearly 50% of total error variance is in off-diagonal (correlated) terms — the noise acts as a coordinated directional perturbation, not random static. Errors don't average out; they push.

**"Solving Reproducibility Challenges in Deep Learning and LLMs: Our Journey"**
Ingonyama, 2024-2025. https://www.ingonyama.com/post/solving-reproducibility-challenges-in-deep-learning-and-llms-our-journey
Traced non-determinism down to PTX (CUDA machine code) level. Even quantized models reintroduce non-determinism through unquantized LayerNorm layers. Rewrote GEMM CUDA kernels using only CUDA Cores (avoiding Tensor Cores) for determinism.

**"Robustness of Deep Learning Classification to Adversarial Input on GPUs: Asynchronous Parallel Accumulation Is a Source of Vulnerability"**
Shanmugavelu et al., Euro-Par 2025. https://link.springer.com/chapter/10.1007/978-3-031-99857-7_22
Connects GPU non-determinism from asynchronous parallel accumulation directly to adversarial robustness vulnerabilities.

---

## Where This Project Sits

Existing work covers the pieces individually:
- Single-call nondeterminism is well-documented (papers 1, 2, 25, 28)
- Agent-level behavioral drift over long interactions has been named (paper 5)
- Infrastructure serving degradation is understood (papers 17, 18, 19, 20)
- The reproducibility crisis is acknowledged (papers 26, 27, 30)
- Floating-point nondeterminism is traced to hardware (papers 31, 32, 33)

Nobody has connected these into a single experimental measurement: **how does per-step infrastructure-induced variance compound through a structured multi-agent pipeline, and how does that compounding differ across infrastructure tiers?** That is the gap this project fills.
