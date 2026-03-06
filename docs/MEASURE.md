# MEASURE.md — Quantifying Variability in Iterative Agent Systems

---

## 1. The Measurement Problem

We want to answer: **does an iterative, multi-agent system produce meaningfully different outputs under different infrastructure conditions, and does that difference grow with pipeline depth?**

This requires solving two subproblems:

1. **What to compare**: choosing output representations and metrics that capture real variability without overcounting cosmetic differences or undercounting semantic drift.
2. **How to avoid contaminating the measurement**: any measurement tool that itself runs on variable infrastructure introduces the same class of noise it's trying to detect.

### The Meta-Measurement Trap

If we use an LLM to judge whether two outputs are "the same," that judgment is subject to the same infrastructure variance we're measuring. An evaluator running during peak hours might judge two outputs as equivalent that an off-peak evaluator would flag as divergent. Every measurement method must either be:

- **Deterministic**: computed by fixed algorithm with no model in the loop
- **Locally grounded**: run on dedicated local hardware with sequential inference, no batching, no cache eviction

This constraint shapes the entire measurement plan.

---

## 2. Factors That Produce Variability

Each factor below contributes to output variance. The experiment must either control or measure each one.

| Factor | Type | How It Contributes |
|--------|------|--------------------|
| Model infrastructure provider | Environmental | Different providers have different routing, batching, caching, and substitution behavior |
| Model identity | Controlled | Fixed per experiment run; but provider may silently substitute |
| KV cache state | Environmental | Eviction forces recompute under different batch compositions; prefix cache misses force cold prefill |
| Quantization level | Environmental/Controlled | INT4 vs FP8 vs FP16 produce different numerical paths; provider may change dynamically |
| Infrastructure load | Environmental | Contention for GPU memory, compute, and scheduling priority |
| Iteration depth | Structural | Each iteration compounds variance from previous steps |
| Agent fan-out (parallelism) | Structural | Multiple agent outputs combined means multiple independent draws from a noisy distribution |
| Temperature / sampling | Controlled | Fixed at 0 or fixed seed to isolate infrastructure variance from sampling variance |

---

## 3. Measurement Layers

Measurements are organized into **primary metrics** (Layers 1-5) that directly measure output variability, and **diagnostic telemetry** recorded alongside each run to help explain *why* variability occurred.

### Diagnostic Telemetry (recorded per run, not a variability metric)

Infrastructure signals recorded alongside each pipeline execution. These do not measure output variability — they provide after-the-fact context for explaining it. If a run diverges from the reference, telemetry may reveal that a preemption event or cache miss coincided with the divergence point.

| Signal | Source | Why Record It |
|--------|--------|---------------|
| TTFT | API response timing | Latency spike hints at preemption or cold prefill |
| TPOT | Token timing | Bursty decode suggests batching interference |
| Total latency | API response timing | Gross indicator of serving pressure |
| Output token count | API response | Detects truncation (generation cut short) |
| KV cache utilization % | vLLM Prometheus `/metrics` | Eviction pressure indicator (self-hosted only) |
| Prefix cache hit rate | vLLM metrics | Cold vs. warm prefill (self-hosted only) |
| Preemption count | vLLM `total_cumulative_preemption_cnt` | Direct eviction event count (self-hosted only) |

These are not primary measurements. They are collected opportunistically and used in post-hoc correlation analysis: "runs where TTFT exceeded X also showed higher edit distance." The correlation may or may not be significant — that's a finding, not an assumption.

For cloud providers, most of these signals are unavailable. Latency and token count are the only universally observable signals, which is itself part of the problem: degradation on shared infrastructure is invisible by design.

### Layer 1: Deterministic Text Comparison

Character-by-character and token-by-token comparison of raw output. Fully deterministic. No model in the loop.

**Exact match rate**: What fraction of repeated runs produce byte-identical output? At temperature=0, any deviation from 100% is infrastructure-induced. This is the single most important baseline metric.

**Normalized edit distance**: Levenshtein distance divided by max output length. Captures the magnitude of text-level differences. Use character-level and token-level variants:

```
char_edit_distance = levenshtein(output_a, output_b) / max(len(a), len(b))
token_edit_distance = levenshtein(tokenize(a), tokenize(b)) / max(tokens(a), tokens(b))
```

Token-level is more meaningful because a single-character change that crosses a token boundary looks like a larger difference at the character level than it is semantically.

**N-gram overlap (BLEU/ROUGE family)**: BLEU-4 and ROUGE-L against a reference baseline run. Well-understood metrics with known limitations. Useful because they're standard and comparable to other work, not because they're ideal.

**Output length variance**: Standard deviation of output token count across runs with identical input. Length variation is a coarse but informative signal — if the model produces 200 tokens on one run and 800 on another from the same prompt at temperature=0, something changed.

**Longest common subsequence ratio**: LCS length / max output length. Captures how much of the core content survives across runs even when surface ordering changes.

### Layer 2: Structural Comparison

For outputs with parseable structure, compare the structure rather than the text.

**AST diff (code outputs)**: Parse both outputs into abstract syntax trees. Compare tree edit distance. Two code snippets that differ in variable names or whitespace but have identical ASTs are structurally equivalent. This separates cosmetic variation from logical variation.

**JSON/YAML structural diff**: For structured data outputs, compare the parsed data structure. Ignore key ordering, whitespace, formatting. Flag value differences, missing keys, added keys, type changes.

**Section/heading structure diff**: For document outputs, extract the outline (headings, bullet structure, numbered lists) and compare. Captures whether the organization of the response changed even if individual paragraphs differ.

**Dependency graph diff**: For outputs that describe a plan, architecture, or workflow, extract the dependency graph (what depends on what) and compare graph isomorphism. Two plans that describe the same dependencies in different words are structurally equivalent.

### Layer 3: Embedding Similarity

Capture semantic proximity without using a generative model. Deterministic once the embedding model is fixed.

**Requirements**: The embedding model MUST run locally on dedicated hardware. Using a cloud embedding API subjects the measurement to the same variance being measured.

**Whole-output cosine similarity**: Embed the full output as a single vector, compute cosine similarity between run pairs. Use a fixed, locally-run embedding model (e.g., `all-MiniLM-L6-v2`, `bge-large`, `nomic-embed-text`). Choose one model and fix it for all experiments.

**Chunk-level similarity**: Split outputs into fixed-size chunks (e.g., 256 tokens), embed each, compute pairwise similarity matrix. This catches cases where the overall output is similar but a specific section diverged.

**Sentence-level alignment**: Split into sentences, embed each, find best-match alignment between two outputs, report unmatched sentences. Captures insertions, deletions, and substitutions at the sentence level.

**Calibration requirement**: Before using embedding similarity as a variability metric, establish the scale. Run the embedding model on known-equivalent paraphrases and known-different texts to map cosine similarity values to "same/different/partially different" categories. Without calibration, a cosine similarity of 0.92 is uninterpretable.

### Layer 4: Functional Equivalence

Does the output *do* the same thing? This is the gold standard but only applies to outputs with testable behavior.

**Test suite pass rate (code outputs)**: If the agent produces code, run a fixed test suite. Binary: pass or fail. No measurement variance. Two code outputs that both pass all tests are functionally equivalent regardless of textual differences. Track which specific tests pass/fail across runs to detect partial degradation.

**Assertion checking (structured outputs)**: Define assertions over the output. "The plan must include step X before step Y." "The analysis must mention factor Z." Binary per assertion. Percentage of assertions satisfied across runs.

**Numerical answer comparison (math/analysis)**: If the output includes numerical results, extract and compare. Exact match or within-tolerance. No model needed.

**Decision consistency**: If the agent makes binary decisions (accept/reject, yes/no, include/exclude), track the decision across runs. Agreement rate is a clean, deterministic metric.

### Layer 5: Semantic Equivalence Judgment

Full semantic comparison using a judge model. Most informative, but introduces its own variance.

**Local judge model**: Run a judge model on dedicated local hardware (Tier 3 — sequential inference, no batching). Present output pairs and ask: "Are these semantically equivalent, partially equivalent, or divergent?" Use structured output (JSON) to force categorical judgment.

**Judge calibration**: Before using in the experiment, measure the judge's own consistency. Run the same pair through the judge N times at temperature=0. If the judge itself gives inconsistent answers, its verdicts cannot be trusted. Report judge self-consistency alongside all Layer 5 results.

**Multiple judge models**: If feasible, run 2-3 different local judge models and report agreement. Agreement across judges increases confidence. Disagreement is itself a data point about the difficulty of the comparison.

**Rubric-based scoring**: Rather than open-ended "are these the same?", provide a specific rubric. E.g., "Score 1-5 on: completeness, accuracy, coherence, consistency with input requirements." Structured scoring is more reproducible than binary same/different judgments.

---

## 4. What Specifically to Measure on the Output

Given a set of N runs of the same pipeline with the same input, compute:

### Within-Run Measurements (per pipeline execution)

| Measurement | Layer | What It Captures |
|-------------|-------|------------------|
| Per-step output token count | L1 | Truncation, generation length variance |
| Per-step exact match vs. reference | L1 | Binary: did this step produce identical output? |
| Per-step edit distance vs. reference | L1 | Magnitude of text-level divergence |
| Per-step embedding similarity vs. reference | L3 | Semantic distance from reference at each step |
| Evaluator decision (accept/reject) | L4 | Feedback loop consistency |
| Iteration count to convergence | L4 | Behavioral stability |
| Drift curve: cumulative divergence vs. step number | L1/L3 | Shape of variance amplification |

Diagnostic telemetry (TTFT, TPOT, preemption count, cache hit rate) is recorded alongside each step but analyzed separately — see correlation analysis in Section 7.

### Across-Run Measurements (comparing N runs)

| Measurement | Layer | What It Captures |
|-------------|-------|------------------|
| Exact match rate across runs | L1 | Baseline reproducibility |
| Pairwise edit distance distribution | L1 | Width of text-level output distribution |
| Pairwise embedding similarity distribution | L3 | Width of semantic output distribution |
| Output length coefficient of variation | L1 | Normalized length variability |
| Final output functional equivalence rate | L4 | Do outputs behave the same? |
| Convergence iteration count distribution | L4 | Behavioral consistency |
| Mean and variance of drift curve across runs | L1/L3 | How consistently does drift accumulate? |
| Evaluator agreement rate on same intermediate output | L4 | Feedback loop reliability |

### Across-Condition Measurements (comparing run sets between conditions)

| Measurement | What It Captures |
|-------------|------------------|
| Between-condition effect size on final output similarity | Does time-of-day / infra tier change the output distribution? |
| Between-condition effect size on convergence iterations | Does infrastructure affect behavioral convergence? |
| Between-condition drift curve shape comparison | Does variance amplify differently under load? |

Post-hoc diagnostic analysis correlates telemetry signals (TTFT, cache utilization, preemption counts) against the primary output metrics above. This is exploratory — it may reveal which infrastructure events predict output degradation, but the telemetry is not itself a variability measurement.

---

## 5. The Test System: Amplifying Individual Variation

To detect infrastructure-induced variance, the test system must have enough depth and breadth to accumulate small per-step deviations into measurable final-output differences.

### Design Principles

**Depth (iteration/recursion)**: Each step conditions the next. A 10-step pipeline where each step's output becomes the next step's input gives variance 10 opportunities to compound. The drift curve across steps is the primary measurement.

**Breadth (parallelism/fan-out)**: Multiple agents produce intermediate results that get combined. If 4 agents each contribute a section and a combiner agent merges them, variance in any single agent's output propagates to the final result. Fan-out means more independent draws from a noisy distribution.

**Combination (depth x breadth)**: A tree structure where parallel agents produce intermediate outputs, those outputs get combined, and the combined result feeds the next round of parallel agents. This is the architecture that maximally amplifies per-step variance.

### Candidate Pipeline Architecture

```
Round 1:  [Agent A1] [Agent A2] [Agent A3] [Agent A4]   (fan-out = 4)
              \          \          /          /
Round 2:        [Combiner B1]    [Combiner B2]           (fan-in = 2)
                    \                /
Round 3:           [Evaluator C]                          (evaluate)
                        |
Round 4:           [Refiner D]                            (revise)
                        |
                   Loop → Round 1 with revised input      (iterate)
```

Each round is a measurement point. Each edge is an opportunity for variance to propagate. The full pipeline runs for K iterations or until convergence.

### Task Selection Criteria

The task given to the pipeline must be:

- **Complex enough** that the model cannot solve it in a single token — forces multi-step reasoning where variance has room to compound
- **Deterministic enough** that a correct answer exists and can be verified — enables Layer 4 functional equivalence checking
- **Decomposable** into parallel subtasks — enables fan-out
- **Iteratively refinable** — the evaluator must have real criteria to evaluate against, and the refiner must have real work to do

Candidate tasks:
- Generate a multi-module software system from a specification (code outputs, test-suite verifiable)
- Produce a structured analysis document from a data set (assertions on required sections/conclusions)
- Solve a multi-step mathematical derivation (numerical answer verifiable)

---

## 6. Establishing Baselines

Before measuring infrastructure-induced variance, establish the variance floor.

### Baseline 1: Tier 3 Sequential Inference

Same pipeline, same model, dedicated hardware, sequential inference, no batching, temperature=0. Run N=30 times. Any variance in this baseline is the irreducible floor — model nondeterminism at the hardware level (floating-point nondeterminism in CUDA kernels, etc.). All other measurements are compared against this floor.

### Baseline 2: Trivial Pipeline

Single-step, single-agent, short-context prompt on Tier 3 hardware. Measures the variance floor for a single model call without pipeline amplification. The delta between this and the full pipeline on the same hardware isolates the amplification effect of pipeline depth/breadth.

### Baseline 3: Known-Equivalent Inputs

Feed the pipeline two inputs that are semantically identical but phrased differently. The output difference establishes the model's sensitivity to input paraphrase — a confound that must be separated from infrastructure-induced variance.

---

## 7. Statistical Analysis Plan

### Distribution Characterization

Do not assume normality. For each metric, report:
- Median and IQR (robust central tendency)
- Full distribution (histogram or kernel density estimate)
- Shapiro-Wilk test for normality

### Between-Condition Comparisons

- **Kruskal-Wallis H-test**: non-parametric comparison across time windows (off-peak / shoulder / peak) and infrastructure tiers
- **Mann-Whitney U**: pairwise tier comparisons (Tier 1 vs. 2, Tier 2 vs. 3, Tier 1 vs. 3)
- **Effect size**: Cliff's delta (non-parametric effect size) rather than Cohen's d

### Variance Comparison

- **Levene's test**: are the variances significantly different between conditions? (This is the core question — not "are the means different?" but "is the spread different?")
- **Brown-Forsythe test**: robust variant of Levene's for non-normal distributions
- **Coefficient of variation ratio**: CV(peak) / CV(off-peak) as a simple amplification factor

### Drift Curve Analysis

- **Mixed-effects model**: drift ~ step * condition + (1|run), to estimate how drift grows with step number and whether the growth rate differs between conditions
- **Breakpoint detection**: does the drift curve have an inflection point where variance accelerates? Change-point analysis on the per-step variance series

### Diagnostic Correlation (secondary, exploratory)

Where telemetry is available (self-hosted tiers, latency on cloud), test whether infrastructure signals correlate with output variability:

- **Spearman rank correlation**: between telemetry (TTFT, KV cache util, prefix cache hit rate) and primary output metrics. May identify which infrastructure events coincide with output degradation — or may show no useful correlation, which is itself a finding (degradation is invisible even with telemetry)
- **Lag analysis**: does a telemetry spike at step N predict output divergence at step N, N+1, or later? Tests whether variance effects are immediate or propagate through the pipeline

This analysis is exploratory and not required to demonstrate the thesis. The primary finding is output variability itself; telemetry correlation is a bonus that helps explain mechanism.

### Sample Size

With N=30 runs per condition and 3 conditions (off-peak, shoulder, peak) across 3 infrastructure tiers, the total matrix is 270 pipeline executions. Power analysis for Kruskal-Wallis with expected medium effect size (Cliff's delta ~0.3) at alpha=0.05 requires approximately N=25-30 per group.

---

## 8. Reporting Metrics Summary

The minimum viable measurement set (in priority order):

1. **Exact match rate** (L1) — did the same input produce the same output? Binary, deterministic, no ambiguity.
2. **Normalized edit distance** (L1) — how different is the text? Deterministic magnitude.
3. **Output length CV** (L1) — how much does output length vary? Single number, immediately interpretable.
4. **Convergence iteration count distribution** (L4) — does the pipeline converge consistently? Behavioral metric.
5. **Drift curve slope** (L1/L3) — does variance grow with depth? The central question.
6. **Embedding cosine similarity** (L3) — semantic proximity, locally computed.
7. **Functional equivalence rate** (L4) — do outputs do the same thing? Gold standard where applicable.
8. **Between-condition variance ratio** (statistical) — is variance wider under load? The core finding.

Metrics 1-5 require no model in the loop. They should be computed first and may be sufficient to demonstrate the thesis without requiring the more complex semantic measurements.
