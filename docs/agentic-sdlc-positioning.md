# Drift: Quality Assurance for the Agentic SDLC

## The Problem We Must Solve Before We Scale

the vendor is transitioning to an Agentic Software Development Lifecycle. AI agents will operate across the entire pipeline — planning, coding, code review, testing, security, documentation, and support. This is the right direction, and the urgency is real.

But there is a foundational quality risk that must be addressed before we scale: **Large Language Models are non-deterministic.** Given the same input, the same model, and the same configuration, an LLM can produce different outputs on successive calls. This behavior — which we call **drift** — is not a bug in any single model or platform. It is an inherent property of how modern inference infrastructure operates.

If we are building an organization where "agents are the execution engine" and "humans provide oversight," then we need to be able to answer a basic question: **when I run this agent workflow twice with the same inputs, do I get the same result?**

Today, for many inference configurations, the answer is no.

## What We Measured

We built a testing framework to quantify output variability across LLM providers and configurations. Our test case is a structured data extraction task: given a complex commercial invoice (12 line items, multiple tax jurisdictions, discounts, credits, and payment terms), extract all fields into structured JSON.

This task was chosen because it is representative of the kind of work agents will perform across the SDLC — taking unstructured input and producing structured, actionable output. Code review findings, test case generation, security vulnerability assessments, and documentation all follow this pattern.

We ran each configuration 30 times, with identical inputs, at temperature 0 (the setting that should produce maximum determinism).

### Results

| Environment | Model | Exact Match Rate | Unique Outputs | Mean Latency |
|-------------|-------|-----------------|----------------|--------------|
| Self-hosted (Ollama, single GPU) | Qwen2.5-32B | **100%** | 1 | 173s |
| Self-hosted, 2-stage pipeline | Qwen2.5-32B | **100%** | 1 | 270s |
| Cloud inference (DeepInfra) | Qwen2.5-72B-Instruct | **83.3%** | 3 | 50s |

The self-hosted environment produced identical output across all 30 runs, on both single-stage and multi-stage pipelines. The cloud inference provider produced three distinct output variants in a 25/3/2 distribution.

### What Drifted

On the cloud provider, the structural extraction was perfect across all 30 runs — every line item, every dollar amount, every field was identical. The drift appeared in how the model chose to format description strings:

- **Majority (25 runs):** `"TX State Sales Tax (6.25%)"` — rate included in the description and the rate field
- **Variant 2 (3 runs):** `"TX State Sales Tax"` — rate only in the dedicated field
- **Variant 3 (2 runs):** Mixed — some descriptions normalized, others not

These are small differences. In an invoice extraction task, they are cosmetic. But the implications for SDLC agent workflows are significant.

## Why This Matters for the Agentic SDLC

### 1. Reproducibility Is a Requirement, Not a Feature

The Global Engineering memo states: *"human ownership of the output, ensuring our teams remain the drivers of innovation."*

Human ownership requires reproducibility. When a security agent triages a CVE, when a code review agent flags a potential vulnerability, when a testing agent generates a test suite — the humans overseeing these workflows need to know that the output is stable. If an agent produces a different security assessment on Monday than it does on Wednesday for the same code, the human reviewer cannot meaningfully own that output. They are not overseeing a deterministic tool; they are auditing a stochastic process.

Consider the implications across the SDLC stages the memo identifies:

- **Code review:** An agent flags 4 issues on one run and 5 on the next. Which review does the developer act on? The developer would have to run the review multiple times and compare, defeating the purpose of automation.
- **Security scanning:** An agent classifies a dependency vulnerability as "medium" severity in one run and "high" in another. The response playbook differs between these classifications. Which classification is correct?
- **Testing:** An agent generates 12 test cases on one run and 14 on the next. Are the extra two tests valuable edge cases or hallucinated scenarios? The developer must review every test anyway to determine which set to trust.
- **Documentation:** An agent generates API documentation that describes a parameter as "optional" in one run and "required" in another. The documentation ships. A customer files a support case.

In each case, non-determinism transforms agent oversight from a review task into an investigation task. This is the opposite of the efficiency gain the agentic SDLC is designed to deliver.

### 2. The Infrastructure Choice Matters

Our data shows that drift is not primarily a model quality issue — it is an infrastructure issue. The same model family produced zero drift on self-hosted infrastructure and measurable drift on shared cloud infrastructure.

The root cause is well-understood: on shared inference servers, requests from multiple users are batched together for GPU efficiency. The batch composition varies from request to request, which changes the order of floating-point operations in attention and normalization layers. Because floating-point addition is non-associative (a + b + c ≠ a + c + b at the precision boundary), different batch compositions produce different numerical results, which can tip the model's token selection at decision boundaries.

This has direct implications for infrastructure decisions:

- **Self-hosted, single-GPU inference** (e.g., Ollama): Deterministic at temperature 0. No batching means consistent floating-point operation order. Slower, but reproducible.
- **Shared cloud inference** (e.g., DeepInfra, managed API endpoints): Non-deterministic due to batch interleaving. Faster and cheaper, but outputs vary across runs.
- **Deterministic inference engines** (e.g., SGLang with `--enable-deterministic-inference`): Replace standard GPU kernels with batch-invariant implementations. Achieves determinism at approximately 34% throughput cost on shared infrastructure.
- **Constrained decoding** (e.g., SGLang with JSON schema enforcement): Eliminates structural drift (missing fields, wrong types, inconsistent keys) by masking invalid tokens at each decoding step. Value drift within free-form fields can still occur.

Teams selecting inference infrastructure for agent workflows need to evaluate this tradeoff explicitly. The fastest, cheapest inference option may not be appropriate for workflows where reproducibility is a quality requirement.

### 3. Measurement Must Be Part of the Workflow

The memo establishes a clear measurement philosophy: *"Measure at the workflow level, not the activity level. We measure whether workflows are hitting their targets: cycle time, defect rate, throughput, and resolution time."*

Drift measurement fits directly into this framework. It is a workflow-level quality metric that answers: **is this agent workflow producing consistent results?** It complements the other proposed indicators:

| Proposed Metric | What It Measures | Drift's Role |
|----------------|-----------------|--------------|
| Cycle time | How fast we deliver | Drift testing adds a quality gate that prevents rework from inconsistent agent output |
| Defect rate | How often we ship bugs | Non-deterministic agents can introduce defects that appear intermittently and are difficult to reproduce |
| Code coverage | How well we test | Agent-generated tests that vary across runs produce inconsistent coverage metrics |
| PR cycle time | How fast we review | Inconsistent agent reviews require additional human review time, increasing PR cycle time |

A drift score should be a prerequisite for deploying an agent workflow into production. If the workflow cannot meet a minimum exact match rate on representative inputs, it is not ready for the "all-in product scope" the memo envisions.

## What We Recommend

### Establish Drift Testing as a Standard Practice

Before a team deploys an agent workflow, they should run a drift test: execute the workflow N times (we recommend 30) with identical representative inputs and measure the exact match rate. This takes hours, not days, and produces a concrete quality number.

We propose the following thresholds as starting points for discussion:

| Workflow Type | Minimum Exact Match Rate | Rationale |
|--------------|------------------------|-----------|
| Security-critical (CVE triage, vulnerability scanning) | 100% | Non-deterministic security assessments are unacceptable |
| Quality-critical (code review, test generation) | ≥95% | Minor formatting drift is tolerable; structural drift is not |
| Content generation (documentation, summaries) | ≥80% | Some variation is expected and acceptable |

These thresholds should be calibrated per team based on the downstream impact of inconsistency.

### Include Infrastructure Determinism in Architecture Decisions

When teams architect their agent workflows, the inference infrastructure choice should be an explicit architectural decision with documented tradeoffs, not a default. Questions teams should answer:

1. Does this workflow require reproducible output? (Security, compliance, and audit workflows almost certainly do.)
2. What is the acceptable drift rate for this workflow?
3. What is the latency budget? (Deterministic inference is slower.)
4. Can constrained decoding (JSON schema enforcement) eliminate the categories of drift that matter for this workflow?

### Build Drift Testing into CI/CD

As agent workflows mature, drift tests should run as part of the CI/CD pipeline, alongside unit tests and integration tests. A model update, infrastructure change, or prompt modification that increases drift beyond the established threshold should block deployment, just as a failing test suite would.

### Contribute Upstream

The memo states: *"We'll work with our communities to build the frameworks, establish best practices, and define the standards for responsibly integrating AI into the open source development lifecycle."*

Drift measurement is a natural upstream contribution. The problem is universal — every organization adopting agentic workflows faces the same reproducibility challenge. By open-sourcing our drift testing framework and publishing our findings, the vendor can lead the conversation on quality standards for agentic development, consistent with our mission to be "the catalyst for communities of customers, contributors, and partners to build better technology the open source way."

## Conclusion

The agentic SDLC transition is the right move. The competitive pressure is real, and the technology is ready. But "all-in" means all-in on quality, not just velocity. We have the data showing that LLM non-determinism is a measurable, infrastructure-dependent phenomenon that will affect every agent workflow we deploy.

The investment required is modest: a testing framework (which exists today), a measurement practice (30 runs per workflow), and a threshold policy (team-specific, data-driven). The cost of not making this investment is higher: intermittent defects, unreproducible agent behavior, and erosion of the human oversight that the agentic SDLC depends on.

We should ship fast. We should also ship the same thing twice.
