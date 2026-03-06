# Results

## Experiment 1: Single-Call Baseline

**Date**: 2026-03-06

**Goal**: Establish the variance floor for a single LLM call with identical inputs at temperature=0.

### Setup

- **Task**: Code review of a Python function with known bugs (SQL injection, password exposure, N+1 queries, fragile column access)
- **Prompt**: Fixed system prompt + user prompt (see `experiments/baseline.yaml`)
- **Runs**: 10 per configuration

### Configuration A: Dedicated Ollama (Tier 3)

- **Model**: qwen2.5:32b (Q4_K_M)
- **Infrastructure**: Dedicated machine (single GPU), sequential inference, no concurrent load

| Metric | Value |
|--------|-------|
| Exact match rate | 1.0 |
| Mean edit distance | 0.0 |
| Max edit distance | 0.0 |
| Num unique outputs | 1 / 10 |
| Length CV | 0.0 |
| Mean latency | 44,776 ms |

**Finding**: Perfect determinism. 10 identical outputs. This is the variance floor.

### Configuration B: Claude via Vertex AI (Tier 1 — Shared Cloud)

- **Model**: claude-sonnet-4-6
- **Infrastructure**: Google Cloud Vertex AI, shared multi-tenant

| Metric | Value |
|--------|-------|
| Exact match rate | 0.1 |
| Mean edit distance | 0.47 |
| Max edit distance | 0.64 |
| Num unique outputs | 10 / 10 |
| Length CV | 0.067 |
| Mean latency | 12,810 ms |

**Finding**: Every run produced a unique output. ~47% average text divergence at temperature=0.

### Analysis: Text Variance vs. Semantic Equivalence

Despite high textual variance, all 10 Vertex outputs are **semantically equivalent**. Every run:

- Identified the same bugs in the same priority order (SQL injection, password exposure, N+1 queries, fragile column access, missing type hints)
- Proposed the same fixes (parameterized queries, exclude password, batch with IN clause, use row_factory)
- Used the same severity classification (critical → performance → reliability → style)

The variance is entirely in **surface-level generation**: phrasing, section organization, code example style, whether a summary table is included, and level of detail. Output lengths ranged from 2,135 to 2,705 characters.

This confirms the **output space dimensionality** hypothesis from BASIS.md: the underlying decisions are stable, but free-form text generation gives variance room to express itself. A functional equivalence metric (Layer 4) would show near-zero variance while text metrics (Layer 1) show ~47% edit distance. Both are true simultaneously.

### Open Question

Does this textual variance matter when outputs are **consumed by another LLM**? Two semantically equivalent but textually different code reviews, fed as input to a downstream agent, may or may not produce different downstream outputs. This is the subject of Experiment 2.

---

## Experiment 2: Chained Pipeline

**Date**: 2026-03-06

**Goal**: Measure whether textual variance from a single LLM call compounds, stabilizes, or is absorbed when outputs are chained through a multi-step pipeline.

### Pipeline Design

A 3-step pipeline simulating a realistic code improvement workflow. Each step has a **distinct role** with its own system prompt — this is not the same prompt repeated, but three independent agents each doing different work:

1. **Reviewer** — receives the original code and produces a code review (same task as Experiment 1)
2. **Implementer** — receives the review output and implements all recommended fixes, returning corrected code with inline comments
3. **Verifier** — receives the corrected code and verifies it for correctness, security, and best practices

Each step's output becomes the next step's entire user prompt. The system prompts are fixed and distinct per step. This means variance at step 1 (different review text) feeds into step 2 as different instructions (which fixes to implement), and step 2's output (different corrected code) feeds step 3 as different code to verify.

### Trial A: Homogeneous Pipeline (vertex × 3)

All three steps use the same model on the same infrastructure.

- **Model**: claude-sonnet-4-6 at every step
- **Infrastructure**: Google Cloud Vertex AI (shared, multi-tenant)
- **Runs**: 10

| Step | Role | Mean Edit Dist | Max Edit Dist | Unique | Length CV | Mean Latency |
|------|------|---------------|---------------|--------|-----------|-------------|
| 1 | reviewer | 0.47 | 0.64 | 10/10 | 0.084 | 13,240 ms |
| 2 | implementer | 0.58 | 0.74 | 10/10 | 0.158 | 8,113 ms |
| 3 | verifier | 0.73 | 0.81 | 10/10 | 0.095 | 13,386 ms |

**Drift curve (mean edit distance)**: 0.47 → 0.58 → 0.73

**Finding**: Variance compounds at every step. The reviewer introduces ~47% textual variance. The implementer, receiving 10 different reviews as instructions, amplifies to ~58%. The verifier, receiving 10 different implementations, amplifies further to ~73%. No convergence — each step adds divergence.

The length CV spike at step 2 (0.158, nearly 2x the other steps) suggests the implementer is particularly sensitive to input phrasing — different review text leads to meaningfully different amounts of code being produced, not just different wording.

### Trial B: Mixed-Model Pipeline (vertex → ollama → ollama)

Expensive cloud model for the initial generation, cheaper dedicated model for downstream processing. Reflects real-world cost optimization.

- **Step 1**: claude-sonnet-4-6 via Vertex AI (shared cloud)
- **Steps 2–3**: qwen2.5:32b via Ollama (dedicated hardware, sequential inference)
- **Runs**: 10

| Step | Role | Model | Mean Edit Dist | Max Edit Dist | Unique | Length CV | Mean Latency |
|------|------|-------|---------------|---------------|--------|-----------|-------------|
| 1 | reviewer | claude-sonnet-4-6 | 0.48 | 0.67 | 10/10 | 0.060 | 12,390 ms |
| 2 | implementer | qwen2.5:32b | 0.61 | 0.77 | 10/10 | 0.334 | 31,183 ms |
| 3 | verifier | qwen2.5:32b | 0.67 | 0.76 | 10/10 | 0.132 | 52,073 ms |

**Drift curve (mean edit distance)**: 0.48 → 0.61 → 0.67

**Finding**: The deterministic Ollama model (which produced 0.0 edit distance in Experiment 1 when given identical input) produced 10 unique outputs at both steps 2 and 3 when fed varying input from Vertex. Ollama is perfectly deterministic given the same input, but it faithfully propagates upstream variance — it does not normalize, absorb, or converge the differences.

The length CV at step 2 is extremely high (0.334) — the dedicated model is even more sensitive to input variation in terms of output length than the cloud model was in Trial A (0.158). Same semantic content in different words produced dramatically different code lengths.

### Comparative Analysis

**Drift curve comparison**:

| Step | Homogeneous (vertex×3) | Mixed (vertex→ollama×2) |
|------|----------------------|------------------------|
| 1 | 0.47 | 0.48 |
| 2 | 0.58 | 0.61 |
| 3 | 0.73 | 0.67 |

- **Step 1 is nearly identical** (~0.47 vs 0.48) — same model, same prompt, expected.
- **Step 2 divergence is comparable** — both models amplify step 1 variance by a similar amount. The dedicated model doesn't dampen variance despite being deterministic on fixed input.
- **Step 3 diverges between trials** — the homogeneous pipeline (0.73) drifted more than the mixed pipeline (0.67). The cloud model adds its own variance at every step (nondeterministic infrastructure noise on top of input variance), while the dedicated model amplifies only what it receives without adding new noise.

### Trial C: Structured Output Pipeline (JSON Schema Sandwich)

Tests whether constraining output format at key pipeline steps absorbs variance. Steps 1 and 3 require JSON schema output; step 2 (implementer) produces raw code.

- **Model**: claude-sonnet-4-6 at every step
- **Infrastructure**: Google Cloud Vertex AI (shared, multi-tenant)
- **Runs**: 10

| Step | Role | Output Format | Mean Edit Dist | Max Edit Dist | Unique | Length CV | Mean Latency |
|------|------|--------------|---------------|---------------|--------|-----------|-------------|
| 1 | reviewer | JSON schema | 0.27 | 0.40 | 10/10 | 0.132 | 8,682 ms |
| 2 | implementer | raw code | 0.52 | 0.73 | 10/10 | 0.289 | 4,150 ms |
| 3 | verifier | JSON schema | 0.34 | 0.49 | 10/10 | 0.102 | 2,285 ms |

**Drift curve (mean edit distance)**: 0.27 → 0.52 → 0.34

**Finding**: The drift curve **bends down** at step 3. This is the only trial where variance decreased between pipeline steps. The JSON schema at step 3 forces the verifier to produce output in a constrained format, absorbing the variance accumulated at step 2 (raw code). Step 1's JSON constraint also reduced initial variance from 0.47 (free-form in Trial A) to 0.27.

### Comparative Analysis

**Drift curve comparison**:

| Step | A: Homogeneous (vertex×3) | B: Mixed (vertex→ollama×2) | C: Structured (JSON sandwich) |
|------|----------------------|------------------------|------------------------------|
| 1 | 0.47 | 0.48 | 0.27 |
| 2 | 0.58 | 0.61 | 0.52 |
| 3 | 0.73 | 0.67 | 0.34 |

- **Step 1**: Structured output (0.27) nearly halves the variance of free-form output (0.47). JSON schema constrains the output space, leaving less room for textual variation.
- **Step 2**: All three trials converge around 0.52–0.61. The implementer produces raw code in all cases, and code has moderate textual variability regardless of how the upstream review was formatted.
- **Step 3**: This is where the pattern diverges dramatically. Free-form pipelines continue climbing (0.67–0.73) while structured output drops back down (0.34). The JSON schema acts as a **variance absorber** — a bottleneck that forces convergence regardless of input diversity.

### Key Takeaways

1. **Variance compounds across pipeline depth in free-form pipelines**. Trials A and B show monotonically increasing edit distance from step 1 → 3.

2. **Structured output absorbs variance**. Trial C's JSON schema at step 3 drops edit distance from 0.52 back to 0.34 — the only case where a downstream step reduced variance rather than amplifying it.

3. **Deterministic models don't absorb variance**. Ollama produced identical output 10/10 times on fixed input (Experiment 1) but 10/10 unique outputs when fed varying input. A downstream model cannot "normalize" upstream variance — semantically equivalent but textually different input produces textually different output.

4. **Cloud infrastructure adds variance at every step**. The homogeneous pipeline drifted more at step 3 (0.73 vs 0.67) because each Vertex call introduces its own nondeterminism on top of the input variance. Dedicated hardware only amplifies existing variance; cloud adds new variance.

5. **Output length is highly sensitive to input phrasing**. Length CV at the implementer step was 0.158 (homogeneous) and 0.334 (mixed) — meaning the same semantic review, phrased differently, caused 15–33% variation in how much code the model produced. This has practical implications for token budgets and cost estimation.

6. **The open question remains semantic**. All measurements so far are Layer 1 (text-level). Whether the 10 different verified implementations are functionally equivalent (do they all pass the same tests?) is unanswered. The textual variance is established; the semantic impact needs Layer 4 measurement.

---

## Literature Context

Our experimental findings align with and extend recent research on multi-agent system reliability. Six papers in our [knowledge base](knowledge/INDEX.md) provide theoretical grounding for our observations.

### Our Results in Context

**Error amplification through pipeline depth** — Our drift curves (0.47 → 0.73 in free-form pipelines) are an empirical demonstration of the cascade amplification described theoretically by Xie et al. (2026) and measured at scale by Kim et al. (2025, via Moran's analysis). Where they study intentional errors propagating through agent networks, we measure accidental variance from infrastructure. The underlying dynamics are identical: each step amplifies rather than corrects upstream variation.

**Topology determines resilience** — Kim et al. report up to 17.2× error amplification in unstructured "bag of agents" networks vs 4.4× with centralized coordination. Huang et al. (2025) find linear pipelines are the *least* resilient topology (23.7% performance drop from faulty agents) compared to hierarchical (5.5%). Our 3-step pipeline is a linear chain — the most vulnerable structure in both studies. This suggests our measured drift rates represent a worst-case topology for variance containment.

**Structured output as a circuit breaker** — Our JSON schema "sandwich pattern" reducing variance from 0.73 to 0.34 at step 3 maps to the governance mechanisms in the literature:
- Xie et al.'s genealogy-based governance decomposes outputs into verifiable atomic claims, achieving 0.89–0.94 defense rates against error cascades
- The MASC framework (ICLR 2026 submission) detects anomalous steps via reconstruction error and triggers correction before propagation
- Our approach is simpler — constraining the output format rather than verifying content — but achieves the same structural effect: reducing the dimensionality of the output space so variance has fewer degrees of freedom

**The spectral risk criterion** — Xie et al.'s formal model (ℛ ≈ βρ(A)/δ) predicts that systems become supercritical when the adoption rate × spectral radius exceeds the decay rate. In our pipeline, β (how much each step adopts upstream variance) is ~1.0 (full adoption), ρ(A) scales with pipeline length, and δ (self-correction) is effectively 0 in free-form pipelines. Structured output increases δ by constraining output space — this is why the drift curve bends down.

**Variance as feature vs. bug** — The Council of AIs study (Shaikh et al., 2025) takes the opposite perspective: response variability is a *strength* when harnessed through deliberation. Their five GPT-4 instances at temperature=1 corrected errors 53% of the time through iterative consensus, achieving 94.7% accuracy on USMLE exams. Our temperature=0 experiments on Vertex show that cloud infrastructure alone produces enough variance for non-trivial diversity (10/10 unique outputs). The question is not whether variance exists, but whether your architecture exploits or suffers from it.

**Semantic vs. syntactic errors** — Huang et al. find that systems handle syntactic errors far better than semantic ones. This maps to our observation: our pipeline outputs are semantically equivalent (same bugs identified, same fixes proposed) but textually different. The variance we measure is predominantly syntactic/surface-level — and yet it still compounds through the pipeline, producing dramatically different final outputs. If even syntactic variance compounds, semantic errors (the kind infrastructure noise could occasionally introduce) would be far more damaging.

### Design Implications for Future Experiments

Based on the literature, several architectural patterns could improve variance containment in our pipelines:

1. **Assurance Plane feedback loop** — Add an Evaluator step with structured pass/fail output after the implementer, recycling failures back to step 2. This is the single biggest performance differentiator in the DeepMind study.

2. **Fan-out with synthesizer** — Instead of a linear chain, fan out to N instances at step 1, then synthesize. The Council of AIs shows this corrects errors 53% of the time, but at 3–7 minutes per question and N× cost.

3. **Hierarchical topology** — Add a coordinator node that aggregates and verifies step outputs before passing downstream. Both Kim et al. and Huang et al. converge on hierarchical > flat > linear for resilience.

4. **Anomaly detection layer** — MASC's prototype-guided reconstruction could detect when a step's output deviates from expected patterns, triggering correction before propagation. This is a more principled version of our JSON schema constraint.

5. **Hybrid fast-path/slow-path** — The Council of AIs found 78% of questions needed no deliberation. A pipeline could fast-track outputs with low variance and only trigger correction/deliberation when variance exceeds a threshold — balancing cost against reliability.

6. **Saturation awareness** — Kim et al. find agent coordination yields highest returns below 45% single-agent accuracy. Our code review task likely exceeds this threshold, meaning the pipeline structure adds more coordination overhead than value for *this specific task*. Testing on harder tasks (where the base model struggles) should show larger benefits from pipeline structure.
