# Theoretical Notes

## Voting vs. Deliberation: Is Simple Good Enough?

The Council of AIs study (Shaikh et al., 2025) achieved 94.7% accuracy through multi-round deliberation among five GPT-4 instances. Simple majority voting hit 91.7% — only 3 percentage points lower.

**The cost asymmetry is stark.** Majority voting is one round: 5 parallel calls, pick the mode. Deliberation required an average of 3.6 rounds per disputed question, with a Facilitator synthesizing and re-prompting each round. That's roughly 5-10x the token cost for 3% accuracy improvement.

Where deliberation shines is the **correction asymmetry**: when the initial majority was wrong (19 questions), deliberation corrected 10 of them (53%). Voting would have gotten all 19 wrong. The 5.0x odds ratio favoring correction over corruption is the real value — but it only applies to the ~7% of questions where the majority initially errs.

**Practical implication**: For most use cases, voting is probably good enough. Reserve deliberation for high-stakes decisions where the cost of error exceeds the cost of multiple rounds.

## Same Model, Same Blind Spots

The Council used exclusively GPT-4 instances — all five council members are the same model, same temperature (1), same top_p (1). The "diversity" they exploit is purely stochastic: different rolls of the same dice. They are not getting fundamentally different training data, architectures, or reasoning strategies.

This makes the 94.7% result both impressive (variance alone can self-correct) and limiting (there's a ceiling when all instances share the same blind spots). The authors acknowledge this and suggest future work with heterogeneous councils (Claude, Bard, etc.) to get genuinely different reasoning approaches.

**The persistent error floor**: When all 5 instances got it wrong initially, deliberation *never* corrected it. The error floor is set by the model's knowledge, not the coordination strategy. No amount of topology, voting, or deliberation fixes a gap in the underlying model.

## Implications for Pipeline Design

This maps directly to our pipeline experiments:

1. **Step 1 dominance**: If step 1 produces fundamentally wrong output (not just textually variant), no downstream structure will recover it. This supports spending on a high-quality step 1 model — the DeepMind finding of "spend on workers, not orchestrator" applies here too.

2. **Infrastructure variance is stochastic diversity**: Our temperature=0 experiments on Vertex produce 10/10 unique outputs from infrastructure noise alone. The Council deliberately sets temperature=1 to maximize diversity. Cloud infrastructure gives you diversity for free — the question is whether your architecture exploits or suffers from it.

3. **Fan-out voting as a cheap defense**: Instead of a linear pipeline (our current approach, the least resilient topology), a fan-out step that runs N instances and takes the majority vote could absorb variance at a fraction of the cost of full deliberation. The 91.7% vs 94.7% result suggests this is usually sufficient.

4. **Heterogeneous models for real diversity**: Using different model families at different pipeline steps (our mixed pipeline: Claude -> Ollama) introduces genuinely different reasoning, not just stochastic sampling. This could break through the error floor that homogeneous councils can't penetrate — but our experiments show it also introduces different variance characteristics (length CV of 0.334 vs 0.158).

## Distillation Viability: When Can You Eliminate the LLM?

A recurring question in enterprise AI deployment: for a given task, do you actually need an LLM at every invocation, or can you extract the model's decision-making into deterministic code and run it without inference cost?

The chain-of-thought traces from reasoning models look like a window into the model's logic, but they aren't. They're generated text — a post-hoc narrative the model produces because it's been trained to write reasoning-like prose before answering. The actual decision-making happens in the forward pass through billions of weights and attention heads. The chain-of-thought correlates with better outputs (more token-steps to work through a problem) but it's a reconstruction, not a recording.

That said, for narrow, well-defined tasks, you can observe the model's outputs across many runs and extract the stable patterns. If the model consistently applies the same rules — parameterize queries, exclude sensitive fields, batch operations — those rules are codifiable regardless of whether the model "knows" it's applying them.

### Drift as a Distillation Viability Metric

Our functional equivalence data across three tasks validates this signal:

| Task | Textual Exact Match | Functional Equivalence | Drift Source |
|------|-------------------|----------------------|--------------|
| code_review (30 runs, DeepInfra) | 33% | 96.7% | 1 variant with duplicate dict key bug |
| data_extraction (30 runs, DeepInfra) | 83.3% | 100% | Description formatting, not values |
| iterative_refactor (5 runs, DeepInfra) | 60% | 100% | Exception handler style, not security fixes |
| iterative_refactor (3 runs, Ollama) | 100% | 100% | No drift (single GPU, deterministic) |

The pattern is consistent: **textual drift wildly overstates functional drift.** Models vary in surface expression while applying the same underlying rules. The one genuine failure (code_review variant 5, duplicate dictionary key) produced a real bug — a 3.3% defect rate from drift alone.

This suggests a measurable threshold:

| Functional Equivalence Rate | Interpretation | Recommendation |
|----------------------------|----------------|----------------|
| >99% | Decisions are deterministic | Distill to code. LLM adds cost without value. |
| 95-99% | Decisions are stable with rare edge cases | Distill the common path, use LLM as fallback for edge cases. |
| 80-95% | Decisions vary meaningfully | LLM is doing real reasoning. Keep it, but add review gates. |
| <80% | Decisions are inconsistent | LLM is unreliable for this task. Improve the prompt, model, or task decomposition before deploying. |

The pattern: **use the LLM to discover the rules, codify them as deterministic code, use the LLM only for cases that fall outside the rules.** This gives you reproducibility and cost savings on the common path with the model as a fallback for genuinely ambiguous cases.

### A Concrete Example: Infrastructure Automation

The AnsibleClaw project (2026) converts Ansible modules into "skills" for AI agents. The skill generation turns out to be a deterministic transformation — Jinja2 templates populated from `ansible-doc --json` output. No LLM inference required. This works because the task (parsing structured documentation and generating wrapper code) is well-defined enough that the "rules" are fully extractable.

The interesting question is the next layer: should an agent use an LLM to decide *which* Ansible module to run with *which* parameters? Drift measurements on that decision process would tell you. If the agent consistently selects the same module and parameters for a given request (high functional equivalence), the decision logic can be distilled into a deterministic dispatcher. If it varies, the LLM is doing real reasoning and the variance itself is a risk — especially when the agent has write access to infrastructure.

### The Cost Implication

The distillation question reframes drift from a quality metric to a cost optimization tool:

- **High functional equivalence** → the LLM is a constant-function generator with expensive overhead. Replace it with code.
- **Low functional equivalence** → the LLM is doing non-trivial work. Keep it, measure the variance, and gate the output.

This gives `drift measure` a prescriptive output beyond "here's how much your outputs vary." It answers: **should you even be using an LLM for this task?**

## Drift as Task-Aware Compression

Drift follows the same fundamental rules as lossless text compression. Both identify and eliminate redundancy — the difference is that drift is *lossy* and *task-aware*: the discarded bits are guaranteed to be noise for the downstream task, so reconstruction isn't needed.

### The Entropy Bound

Shannon's source coding theorem establishes that no lossless compressor can encode data below its entropy rate. Drift faces the same limit: **you cannot reduce a document below its information content.**

A document's Shannon entropy measures the average information per symbol. Low entropy means high redundancy — predictable, repetitive content that a compressor (or drift) can eliminate. High entropy means every token carries unique information — nothing to remove.

This maps directly to drift's observed results:

| Corpus | Entropy Profile | Redundancy | Drift Impact |
|--------|----------------|------------|-------------|
| Ansible logs | Very low — `ok:`, `changed:`, `skipping:` repeated thousands of times | ~80% | 82% chunk reduction, 6.8x faster |
| TechQA (IBM support) | Moderate — template headers/footers across documents | ~11% | 11% reduction, +3.7% nDCG |
| EManual (electronics) | High — unique product descriptions | ~0% | No-op |
| CuAD (legal contracts) | High — unique contract clauses | ~0% | No-op |

**Drift impact is inversely proportional to document entropy.** High-entropy documents (unique natural language) pass through unchanged. Low-entropy documents (repetitive logs, template-heavy support docs) compress dramatically.

### Unifying Input and Output

This entropy framing unifies all of drift's techniques under one principle:

**Input side** (pre-digestion):
- Strips low-entropy content from documents — boilerplate lines, repeated patterns, separator noise
- Impact bounded by the fraction of input tokens below the entropy floor
- Equivalent to a compressor that discards redundancy before the data enters the pipeline

**Output side** (rule extraction):
- Replaces low-entropy LLM output tokens with deterministic rules (regex, literals)
- Zero-entropy tokens (always the same output) → perfect candidates for rule extraction
- Equivalent to a compressor that recognizes constant functions and replaces them with lookup tables

**Classification** (classifier distillation):
- Replaces low-entropy categorical decisions with small classifiers (logistic regression)
- Impact bounded by how predictable the classification is from input features alone
- Equivalent to a compressor that recognizes a mapping function and encodes it directly

In all three cases, drift exploits the gap between the data's actual entropy and the capacity being spent to process it. An LLM is a maximally expressive processor — using it on zero-entropy tokens is like using a general-purpose codec on a file of all zeros.

### Predictive Power

This gives drift a pre-flight diagnostic: **measure corpus-level entropy before running the pipeline.** If document entropy is high, drift will have minimal impact — don't bother. If entropy is low, large gains are likely.

Concretely:
- Compute per-line entropy across the corpus (character-level or token-level)
- Measure cross-document repetition rate (lines appearing in >N% of documents)
- Estimate within-document redundancy (repeated patterns, boilerplate fraction)

These three signals predict drift's impact without running the full pipeline. High redundancy at any level → drift will help. Low redundancy everywhere → the documents are already information-dense and drift is a no-op.

### The Scope Boundary

This framing draws a clear boundary around drift's applicability:

**Drift works when the data has low entropy relative to the task.** This includes:
- Log files (massive repetition, low information density)
- Template-generated documents (shared boilerplate across instances)
- Structured extraction (LLM output is largely deterministic)
- Classification with finite categories (bounded output entropy)

**Drift does not help when entropy is high.** This includes:
- Free-form natural language documents (every token is informative)
- Creative or open-ended generation (high output entropy by design)
- RAG over clean document collections (nothing redundant to remove)

This is not a limitation to work around — it's a fundamental information-theoretic bound. You cannot compress past the entropy floor, and drift cannot reduce past the information content. The same rules that govern lossless compression govern task-aware preprocessing.
