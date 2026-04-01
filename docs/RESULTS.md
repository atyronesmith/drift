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

6. **The open question remains semantic**. All measurements so far are Layer 1 (text-level). Whether the 10 different verified implementations are functionally equivalent (do they all pass the same tests?) is unanswered. The textual variance is established; the semantic impact needs Layer 4 measurement. *(Answered in Experiment 3 below.)*

---

## Experiment 3: Functional Equivalence

**Date**: 2026-03-23

**Goal**: Determine whether the textual drift measured in Experiment 2 corresponds to functional drift — do the 30 different code variants actually behave differently?

### Method

Wrote a test suite (`tests/test_functional_equivalence.py`) targeting the 5 planted bugs from the original code. Each of the 30 implementer step outputs (10 per pipeline trial) was extracted, compiled, and executed against an in-memory SQLite database with known test data.

**10 core tests per variant:**

| Test | What It Checks |
|------|---------------|
| correct_users | Returns correct users for given IDs |
| correct_data | Returns correct field values (id, name, email) |
| parameterized_queries | Uses `?` placeholders, not string interpolation |
| sql_injection_blocked | Injection payload doesn't return all users |
| no_password_in_result | No `password` key in returned dicts |
| password_value_not_leaked | No password values in any field |
| batch_query | Single query with IN clause, not N+1 |
| empty_input | Returns `[]` for empty input |
| nonexistent_user | Returns `[]` for missing IDs |
| mixed_existing_nonexistent | Returns only existing users |

**1 robustness test:** Does the code work without `row_factory` pre-set by the caller?

### Results

| Trial | Core Pass Rate | Variants with Bugs | Robustness (self-contained) |
|-------|---------------|-------------------|---------------------------|
| Homogeneous (Vertex ×3) | **100%** (10/10) | 0 | 5/10 |
| Mixed (Vertex → Ollama ×2) | **90%** (9/10) | 1 | 3/10 |
| Structured (JSON sandwich) | **100%** (10/10) | 0 | 6/10 |

**Overall: 29/30 variants (96.7%) are functionally equivalent on all core tests.**

Despite 47–73% textual edit distance, nearly all variants fix the same bugs, produce the same data, and handle the same edge cases identically.

### The One Bug

`pipeline-mixed/run_005` contains a subtle defect — a duplicate dictionary key:

```python
return [{"id": row[0], "id": row[1], "email": row[2]} for row in rows]
#                       ^^^^^^^^^^^
#                       Should be "name": row[1]
```

Python silently overwrites the first `"id"` with `row[1]` (the name), so `user["id"]` returns `"Alice"` instead of `1`, and the `"name"` key is missing entirely. This variant:

- Returns wrong data for every query
- Would silently corrupt any downstream system consuming the results
- Passes SQL injection, password, and batch query tests — the security fixes are correct, the data mapping is wrong

The bug was generated by the Ollama model (qwen2.5:32b, deterministic at temp=0) faithfully implementing a Vertex-generated code review. The review text varied across runs due to cloud infrastructure non-determinism; this particular variant's review phrasing led the implementer to produce a typo in the dictionary comprehension.

### The row_factory Split

50% of variants (15/30) use named column access (`row["id"]`) without setting `row_factory = sqlite3.Row` themselves. They assume the caller has already configured it — a contract assumption that is invisible in the code review and varies by drift.

| Access Pattern | Count | Self-contained? |
|---------------|-------|----------------|
| Sets `row_factory` internally | 9/30 | Yes |
| Uses `row["col"]`, assumes caller set factory | 6/30 | No — crashes without setup |
| Uses positional `row[0]`, works either way | 14/30 | Yes |
| Handles both patterns (try named, fall back to positional) | 1/30 | Yes |

This is not a correctness bug (the code works when called correctly), but it creates **integration fragility** — two variants that pass all functional tests may still fail differently when integrated into a larger system depending on how the database connection is configured.

### Key Takeaways

1. **Textual drift ≠ functional drift** (mostly). 47–73% edit distance produced 96.7% functional equivalence. For this task, drift is overwhelmingly cosmetic.

2. **But "mostly" is the dangerous word.** 1 in 30 variants (3.3%) contains a real bug that silently returns wrong data. In a system with write access, this variant would corrupt a database. The bug is undetectable by text-level metrics — it requires execution.

3. **The bug came from the mixed pipeline.** The deterministic Ollama model produced the bug because it faithfully implemented a subtly different instruction from a drifted Vertex review. Cloud variance at step 1 caused a functional failure at step 2, even though step 2's model is perfectly deterministic. This is the compounding effect predicted in BASIS.md — variance at one step can cross the threshold from cosmetic to functional at the next.

4. **Structured output may protect against this.** The structured pipeline (Trial C) produced 0 functional bugs across 10 runs. JSON schema constraints at steps 1 and 3 limited the reviewer's output space, which may have prevented the kind of phrasing variation that caused the mixed pipeline's bug. Sample size is too small to be conclusive, but the direction is consistent with the variance absorption finding from Experiment 2.

5. **Functional equivalence testing is a viable drift defense.** A test suite that takes seconds to run caught a bug that 30 runs of text-level drift analysis missed. For code generation pipelines, running generated code against tests is both cheaper and more informative than computing edit distances.

## Experiment 4: Multi-Task Functional Equivalence

**Date**: 2026-03-27

**Goal**: Validate the distillation viability threshold table from [THEORY.md](THEORY.md) across multiple tasks with different complexity profiles.

### Method

Extended functional equivalence testing to the `data_extraction` and `iterative_refactor` tasks using the same test module interface (`extract()` + `tests` dict) as Experiment 3.

**data_extraction** (12 tests, 30 runs): Validates all 12 line items (part numbers, quantities, unit prices, totals), arithmetic consistency (qty × price = total, sum of lines = subtotal), financial totals (subtotal, invoice total, amount due), and entity extraction (seller/buyer names, EINs, invoice metadata).

**iterative_refactor** (10 tests, 5 DeepInfra + 3 Ollama runs): Tests the refactored UserService class against a test SQLite database. Validates the 4 core security fixes (no hardcoded password, parameterized queries in all methods, password excluded from response, no bare except) plus correct behavior (authenticate, search, delete).

### Results

| Task | Provider | Runs | Exact Match | Functional Equivalence |
|------|----------|------|-------------|----------------------|
| data_extraction | DeepInfra | 30 | 83.3% (25/30) | **100%** (30/30) |
| iterative_refactor | DeepInfra | 5 | 60% (3/5) | **100%** (5/5) |
| iterative_refactor | Ollama | 3 | 100% (3/3) | **100%** (3/3) |

For comparison (Experiment 3):

| Task | Provider | Runs | Exact Match | Functional Equivalence |
|------|----------|------|-------------|----------------------|
| code_review | Mixed | 30 | 33% (10/30) | **96.7%** (29/30) |

### Where the Drift Was

**data_extraction**: The 5 textually unique variants differed only in description string formatting — whether rate/percentage text was included in description fields vs. placed in dedicated rate fields. All numeric values, line items, and financial totals were identical across all 30 runs.

**iterative_refactor (DeepInfra)**: The 2 textually unique variants differed in `delete_user` exception handling style — single `sqlite3.Error` handler vs. dual handler (`sqlite3.Error` + generic `Exception`). Both approaches are correct; the difference is defensive depth, not correctness. All 4 security fixes (parameterized queries, password removal, no hardcoded secrets, specific exceptions) were applied identically in all 5 runs.

### Distillation Viability Validation

All four task configurations land in the top two bands of the threshold table:

| Band | Criteria | Tasks |
|------|----------|-------|
| >99% | Distill to code | data_extraction (100%), iterative_refactor (100%) |
| 95-99% | Distill common path, LLM for edge cases | code_review (96.7%) |

The threshold table holds across tasks of different types (JSON extraction, code refactoring, code review) and different complexity (12 line items with financial math vs. 4 security bugs vs. 5 planted bugs in a 3-stage pipeline). The key finding: **textual drift is a poor proxy for functional drift.** A task with 33% exact match can still have 96.7% functional equivalence; a task with 60% exact match can have 100%.

## Experiment 5: Qwen3 Model Comparison

**Date**: 2026-03-30

**Goal**: Compare drift characteristics of Qwen3 (30b) and Qwen3.5 (35b) against the Qwen2.5 (32b) baseline on the same tasks and infrastructure (single-GPU Ollama, temp=0).

### Results

| Task | Model | Runs | Exact Match | Functional Eq | Mean Latency |
|------|-------|------|-------------|---------------|-------------|
| data_extraction | qwen2.5:32b | 30 | 100% | 100% | 173s |
| data_extraction | qwen3:30b | 10 | 90% (2 unique) | 100% | 172s |
| data_extraction | qwen3.5:35b | 1 | — | 0% | 369s |
| iterative_refactor | qwen2.5:32b | 5 | 60% | 100% | 38s |
| iterative_refactor | qwen3:30b | 5 | 20% (2 unique) | 0% | 89s |

### Key Findings

**1. Qwen3 introduces drift where Qwen2.5 was deterministic.** On the same single-GPU Ollama instance at temp=0, Qwen2.5:32b produced 100% exact match on data_extraction. Qwen3:30b produces 90% — the first run differs from subsequent runs (likely KV cache warming). The drift is cosmetic: `184.5` vs `184.50` in JSON numeric formatting. All functional tests pass.

**2. Qwen3 has a model capability regression on security tasks.** On iterative_refactor, all 5 Qwen3 runs expose the password field in the `authenticate` response — a bug that Qwen2.5 fixed in all runs. Qwen3 fixes parameterized queries, removes the hardcoded password constant, and uses specific exception handling, but consistently misses the password exposure issue. This is not drift (it's consistent across runs) — it's a model-level blind spot.

**3. Qwen3.5 is unusable for structured extraction without schema constraints.** A single run returned `$`-formatted strings (`"$184.50"`, `"$2,214.00"`) instead of numeric values, causing all arithmetic tests to fail. This is exactly the failure mode that SGLang's constrained decoding would prevent — and a concrete argument for schema enforcement at the inference layer.

**4. Functional tests catch model regressions that text metrics miss.** The iterative_refactor Qwen3 results show 20% exact match and the analysis module would classify the text drift as "significant." But the real problem isn't drift — it's that the model consistently produces incorrect output. Without functional tests, this would be invisible: the output *looks* reasonable and has the right structure.

### Implications for Model Upgrades

These results demonstrate that upgrading a model version (Qwen2.5 → Qwen3) is not safe without regression testing. A model that scored 100% functional equivalence on a task can drop to 0% on the same task with a newer version. The `drift analyze` workflow provides this safety net: run the same experiment on the new model, compare functional equivalence, and only deploy if the new model maintains or improves the equivalence rate.

### Qwen3 /nothink Mode

Disabling Qwen3's thinking mode with `/nothink` partially addresses the password exclusion regression:

| Config | Functional Eq | First-run bug | Mean implementer latency |
|--------|--------------|---------------|-------------------------|
| qwen3:30b (think) | 0% (0/5) | password exposed in all 5 | 89s |
| qwen3:30b /nothink | 80% (4/5) | password exposed in run 0 only | 89s |
| qwen2.5:32b (baseline) | 100% (5/5) | none | 38s |

On data_extraction, `/nothink` is 24% faster (131s vs 172s) but introduces a new failure: the first run returns `$`-formatted strings instead of numbers (same type error as Qwen3.5).

**The cold-model first-run pattern**: across both tasks and both thinking modes, Qwen3's first run after model load produces different (and often worse) output than subsequent runs. This is consistent with KV cache initialization effects. The practical mitigation: discard the first inference after loading a new model.

## Experiment 6: Temperature Sweep

**Date**: 2026-03-31

**Goal**: Quantify how drift scales with temperature on a structured extraction task (data_extraction) using the same model and infrastructure that is deterministic at temp=0.

### Setup

- **Model**: qwen2.5:32b on dedicated Ollama (single-GPU, sequential inference)
- **Task**: data_extraction (complex industrial invoice, 12 line items)
- **Temperatures**: 0, 0.1, 0.3, 0.5, 1.0
- **Runs**: 10 per temperature

### Results

| Temp | Exact Match | Unique | Edit Distance | Functional Eq | Bugs |
|------|-------------|--------|---------------|---------------|------|
| 0.0  | 100%        | 1      | 0.0000        | 100% (10/10)  | 0    |
| 0.1  | 20%         | 4      | 0.0006        | 100% (10/10)  | 0    |
| 0.3  | 10%         | 10     | 0.0184        | 90% (9/10)    | 1    |
| 0.5  | 10%         | 10     | 0.0348        | 80% (8/10)    | 2    |
| 1.0  | 10%         | 10     | 0.1326        | 40% (4/10)    | 6    |

### Key Findings

**1. Temp 0 → 0.1 is the cliff for textual determinism.** Exact match drops from 100% to 20% with a tiny temperature nudge. Even temp=0.1 produces 4 unique outputs. However, all are functionally equivalent — the differences are cosmetic (JSON field ordering, number formatting).

**2. Functional equivalence degrades gradually.** While text metrics collapse immediately, functional equivalence holds through temp=0.1 (100%) and degrades linearly: 90% at 0.3, 80% at 0.5, 40% at 1.0. This is the clearest evidence yet that textual drift overstates functional drift.

**3. Temp 1.0 is catastrophic for structured tasks.** 6 of 10 outputs contain numerical errors (wrong prices, quantities, or totals). At high temperatures, the model hallucinates values rather than extracting them from the input.

**4. The distillation viability threshold shifts with temperature.** At temp≤0.1, this task is firmly in the "distill to code" band (100% functional eq). At temp=0.3, it drops to "distill + fallback" (90%). At temp=0.5, "keep with gates" (80%). At temp=1.0, it's "unreliable" (40%). Temperature is the single most important knob for drift control.

### Implications

For production deployments: **always use temp=0 when determinism matters.** Even temp=0.1 introduces enough sampling noise to defeat textual caching. If temp>0 is required (for creativity/diversity), functional testing becomes mandatory — text metrics alone will dramatically overstate the problem.

## Experiment 7: Claude Sonnet via Vertex AI

**Date**: 2026-03-31

**Goal**: Compare drift characteristics of Claude Sonnet (cloud API) against Qwen2.5 (single-GPU Ollama) on the same tasks at temp=0.

### Setup

- **Model**: claude-sonnet-4-6 via Vertex AI (project: your-gcp-project-id, region: your-gcp-region)
- **Tasks**: data_extraction (10 runs), iterative_refactor (5 runs)
- **Temperature**: 0

### Results

| Task | Model/Provider | Exact Match | Unique | Edit Distance | Functional Eq |
|------|---------------|-------------|--------|---------------|---------------|
| data_extraction | qwen2.5:32b / Ollama | 100% | 1 | 0.0000 | 100% |
| data_extraction | Claude Sonnet / Vertex | 30% | 4 | 0.0030 | 100% |
| iterative_refactor | qwen2.5:32b / Ollama | 60% | 3 | 0.0054 | 100% |
| iterative_refactor | Claude Sonnet / Vertex | 20% | 5 | 0.3968 | 0%* |

*Claude's 0% functional eq on iterative_refactor is not drift — all 5 runs consistently import `bcrypt` (not installed) and use bcrypt password hashing (incompatible with the test DB's plaintext passwords). This is an architectural choice difference, not a quality issue.

### Key Findings

**1. Cloud APIs drift more than single-GPU at temp=0.** Claude Sonnet on Vertex produces 4 unique outputs on data_extraction where Qwen2.5 on a dedicated GPU produces 1. Cloud infrastructure (load balancing, batch processing, different hardware) introduces sampling noise even at temp=0. Despite this, all 4 variants are functionally equivalent.

**2. Different model families make different architectural decisions.** On iterative_refactor, Qwen2.5 makes minimal, conservative fixes (keeps plaintext comparison, parameterizes queries). Claude introduces a new dependency (bcrypt) and redesigns the auth system. Both approaches address the security issues, but Claude's is incompatible with the existing database schema. This is a model personality difference, not drift.

**3. Claude is 9x faster on extraction.** ~19s per run on Vertex vs ~173s on Ollama (qwen2.5:32b). The pipeline task shows similar speedup: ~50s total per run on Vertex vs ~240s+ on Ollama.

**4. Pipeline convergence is immediate on Claude.** All 5 iterative_refactor runs terminated at iteration 1 (verifier found no remaining issues). Qwen2.5 typically runs 2-3 iterations. Claude's first implementer pass is more thorough.

### Infrastructure Trifecta (data_extraction at temp=0)

Adding DeepInfra (Qwen2.5-72B-Instruct, shared cloud inference) completes the infrastructure comparison:

| Infrastructure | Model | Exact Match | Unique | Edit Dist | Functional | Latency |
|----------------|-------|-------------|--------|-----------|------------|---------|
| Ollama single-GPU | qwen2.5:32b | 100% | 1 | 0.0000 | 100% | ~173s |
| DeepInfra cloud | Qwen2.5-72B | 90% | 2 | 0.0010 | 100% | ~55s |
| Vertex AI cloud | Claude Sonnet | 30% | 4 | 0.0030 | 100% | ~20s |

**Finding**: Textual determinism follows a clear gradient — dedicated GPU > shared cloud (same model family) > cloud API (different model). All three maintain 100% functional equivalence. The drift is entirely cosmetic: JSON field ordering, whitespace, and number formatting differences. DeepInfra's 90% exact match (1 of 10 runs differs) is consistent with shared infrastructure batch non-determinism.

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

---

## Experiment 8: Batch Rule Generalization

**Date**: 2026-03-31

**Goal**: Validate that auto-synthesized rules generalize beyond the single ticket used for probing. Probe one helpdesk ticket, generate rules, then test those rules against 200 different tickets (100 train / 100 test split).

### Setup

- **Dataset**: Console-AI/IT-helpdesk-synthetic-tickets (500 tickets, MIT license, HuggingFace)
- **Model**: qwen2.5:32b on dedicated Ollama (single-GPU, temp=0)
- **Pipeline**: `drift probe` (1 ticket) → `drift synthesize` → validate rules against 200 tickets
- **Input format**: Labeled fields (Ticket ID, Subject, From, Created, Priority, Category) + free-text Description

```
  Ticket ID:    1aiu3lrqi
  Subject:      Hey IT! Our network printer keeps disconnecting.
  From:         jane.doe@acme.co
  Created:      2024-10-28T18:36:55.004Z
  Priority:     Medium
  Category:     Network

  Description:
  Hey IT! Our network printer keeps disconnecting intermittently...
```

### Probe Results (1 ticket)

The probe found 77% deterministic tokens. The synthesizer produced 7 rules from a 12-field output:

| Field | Entropy | Classification |
|-------|---------|---------------|
| ticket.id | 0.000 | rule (regex) |
| ticket.subject | 0.001 | rule (regex) |
| ticket.requester_email | 0.000 | rule (regex) |
| ticket.created_at | 0.000 | rule (regex) |
| ticket.stated_priority | 0.000 | rule (regex) |
| ticket.stated_category | 0.000 | rule (regex) |
| assessment.root_cause_category | 0.001 | rule (regex) — **failed validation** |
| assessment.severity | 0.214 | LLM required |
| assessment.assignment_group | 0.564 | LLM required |
| assessment.summary | 0.666 | LLM required |
| assessment.next_steps[0] | 0.662 | LLM required |
| assessment.next_steps[1] | 1.066 | LLM required |

6 rules validated. 1 rule (`assessment.root_cause_category`) failed validation — it anchored on ticket-specific text ("Hey IT! Our") instead of a format label. 5 fields correctly classified as LLM-required (all assessment/reasoning fields).

### Generated Rules

```
  ticket.id               Ticket\ ID:\s*(\S+)\s*$
  ticket.subject           Subject:\s*(.+?)\s*$
  ticket.requester_email   From:\s*([\w.+-]+@[\w.-]+)\s*$
  ticket.created_at        Created:\s*(\S+)\s*$
  ticket.stated_priority   Priority:\s*(\S+)\s*$
  ticket.stated_category   Category:\s*(\S+)\s*$
```

All 6 patterns anchor on the input format labels (`Ticket ID:`, `Subject:`, `From:`, etc.) — not on ticket-specific values.

### Generalization Results

| Rule | Type | Train (100) | Test (100) | Total (200) | Status |
|------|------|-------------|------------|-------------|--------|
| ticket.id | regex | 100% | 100% | 100% | GENERALIZES |
| ticket.subject | regex | 100% | 100% | 100% | GENERALIZES |
| ticket.requester_email | regex | 100% | 100% | 100% | GENERALIZES |
| ticket.created_at | regex | 100% | 100% | 100% | GENERALIZES |
| ticket.stated_priority | regex | 100% | 100% | 100% | GENERALIZES |
| ticket.stated_category | regex | 100% | 100% | 100% | GENERALIZES |
| assessment.root_cause_category | regex | 0% | 0% | 0% | OVERFIT |

**Train-test gap: 0.0%**

All 6 validated rules match 100% of tickets in both the training and test sets. The overfit rule was already flagged at synthesis time (failed validation). Excluding it, the generalization rate is **6/6 = 100%**.

### Why This Works

The rules generalize because they anchor on **input format structure** rather than content:

```
  ┌──────────────────────────────────────────────────────────────┐
  │  What the rules anchor on:                                    │
  │                                                               │
  │  "Ticket ID:    1aiu3lrqi"      →  Ticket\ ID:\s*(\S+)\s*$  │
  │  "Ticket ID:    fs7blkcux"      →  matches ✓                 │
  │  "Ticket ID:    l0nhq0pys"      →  matches ✓                 │
  │                                                               │
  │  The label "Ticket ID:" is constant across all tickets.       │
  │  The value changes, but the regex captures any \S+ after it. │
  │                                                               │
  │  What the overfit rule anchored on:                           │
  │                                                               │
  │  "Hey IT! Our network..."       →  Hey\ IT!\ Our\s*(\S+)    │
  │  "I'm having trouble with..."   →  NO MATCH ✗               │
  │                                                               │
  │  The description text varies across tickets.                  │
  │  Rules anchored on content don't generalize.                  │
  └──────────────────────────────────────────────────────────────┘
```

### Security Note

The 6 extraction rules are **immune to prompt injection**. Even if a ticket's Description field contains adversarial text designed to manipulate an LLM, the regex rules extract metadata fields without involving the LLM. Only the 5 assessment fields (severity, root_cause, summary, next_steps, assignment_group) pass through the LLM — reducing the attack surface by 50%.

### Baseline vs Hybrid Latency (10 tickets)

With generalization confirmed, we ran both approaches on 10 tickets to measure actual speedup:

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        Pure Baseline                                    │
  │  Input ─────────────────────────▶ LLM ─────────────────▶ Output        │
  │                                   21.1s mean                            │
  │  All 12 fields produced by the LLM                                     │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                     Auto Hybrid (zero human effort)                     │
  │  Input ──┬──▶ Auto rules (6) ───▶ 6 fields ──┐                        │
  │          │    (< 1ms)                          ├──▶ merge ──▶ Output   │
  │          └──▶ LLM ──────────────▶ 6 fields ───┘                       │
  │               12.8s mean                                                │
  │  6 ticket fields via regex, 6 assessment fields via LLM                │
  └─────────────────────────────────────────────────────────────────────────┘
```

| Ticket | Baseline (ms) | Hybrid (ms) | Speedup |
|--------|--------------|-------------|---------|
| 1aiu3lrqi | 26,170 | 12,697 | 51% |
| kz5mjjpox | 23,129 | 14,233 | 38% |
| 86eza0fwq | 19,576 | 12,343 | 37% |
| jtw509e3n | 21,306 | 12,780 | 40% |
| tso616mbn | 19,094 | 11,917 | 38% |
| 6bqwkmxi1 | 18,410 | 11,091 | 40% |
| cythbyl5p | 23,342 | 15,564 | 33% |
| u43q44ro8 | 19,434 | 11,532 | 41% |
| nay58n9jf | 19,872 | 12,296 | 38% |
| 2apbmy56h | 20,302 | 13,432 | 34% |
| **Mean** | **21,064** | **12,789** | **39%** |

- **Field accuracy**: 100% for both baseline and hybrid across all 10 tickets
- **Rules matched**: 6/6 on every ticket
- **Speedup**: 39% mean (33-51% range)

The hybrid approach is faster because the LLM prompt is shorter (only asking for assessment fields) and produces fewer output tokens. The 6 extraction rules execute in microseconds.

### Key Takeaways

1. **Rules generated from 1 ticket generalize to 200.** The auto-synthesis pipeline produces rules that anchor on input format labels, not content. Zero train-test gap.

2. **Validation catches overfit rules at synthesis time.** The `_validate_pattern` step flagged the one bad rule before deployment. No surprises at test time.

3. **The input format determines generalizability.** Labeled fields (`Key: Value`) produce perfect rules. Free-text fields (descriptions, narratives) correctly stay with the LLM.

4. **39% latency reduction with zero human effort.** The hybrid approach cuts mean latency from 21.1s to 12.8s per ticket while maintaining 100% field accuracy. The LLM only handles reasoning (severity, root cause, summary, next steps); extraction is deterministic.

5. **50% of output fields extracted without the LLM.** 6 of 12 fields come from regex rules. This reduces LLM cost, eliminates extraction errors on those fields, and makes them immune to prompt injection.
