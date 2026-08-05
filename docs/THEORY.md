# Theoretical Notes

## Empirical Basis

These notes connect external multi-agent literature to findings measured in this repo. Quantitative claims about drift, functional equivalence, pre-digestion, and classifier distillation are grounded in [RESULTS.md](RESULTS.md): pipeline functional equivalence ([Experiments 3–4](RESULTS.md)), temperature and model-version effects ([Experiments 5–6](RESULTS.md)), infrastructure comparison ([Experiment 7](RESULTS.md)), classifier POC ([Experiment 12](RESULTS.md)), and Ansible pre-digestion plus RAG ingestion ([Experiments 13–16](RESULTS.md)). Infrastructure variance framing aligns with [BASIS.md](BASIS.md).

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

3. **Fan-out voting as a cheap defense, with a caveat**: Instead of a linear pipeline (our current approach, the least resilient topology), a fan-out step that runs N instances and takes the majority vote could absorb variance at a fraction of the cost of full deliberation. But voting and deliberation aren't solving the same problem: voting is cheap (~92% on USMLE) and provides no correction when the majority is wrong; deliberation is expensive (5-10x the token cost for +3pp overall) but recovers 53% of wrong-majority cases. Voting is usually sufficient — but only because wrong-majority cases are rare (~7% of questions here), not because it matches deliberation's correction power.

4. **Heterogeneous models for real diversity — aspirational, not yet demonstrated**: Using different model families at different pipeline steps (our mixed pipeline: Claude Sonnet/Vertex -> Qwen2.5/Ollama) may introduce genuinely different reasoning, not just stochastic sampling — at the cost of different variance characteristics (length CV of 0.334 vs 0.158), including the one functional bug we observed (`pipeline-mixed/run_005`, see [Experiment 3](RESULTS.md)). We have not observed heterogeneity actually breaking through a shared error floor; so far it has produced more variance and the only defect found across 30 variants.

5. **Structured output as a variance absorber**: JSON schema constraints at step 3 reduced edit distance from 0.73 (homogeneous free-form, Trial A) to 0.34 (structured JSON sandwich, Trial C) — a cross-trial comparison, and the only case where downstream variance decreased rather than compounded ([Experiment 2](RESULTS.md)). Constraining output space is cheaper than fan-out deliberation and directly reduces degrees of freedom for infrastructure noise.

## Distillation Viability: When Can You Eliminate the LLM?

A recurring question in enterprise AI deployment: for a given task, do you actually need an LLM at every invocation, or can you extract the model's decision-making into deterministic code and run it without inference cost?

The chain-of-thought traces from reasoning models look like a window into the model's logic, but they aren't. They're generated text — a post-hoc narrative the model produces because it's been trained to write reasoning-like prose before answering. The actual decision-making happens in the forward pass through billions of weights and attention heads. The chain-of-thought correlates with better outputs (more token-steps to work through a problem) but it's a reconstruction, not a recording.

That said, for narrow, well-defined tasks, you can observe the model's outputs across many runs and extract the stable patterns. If the model consistently applies the same rules — parameterize queries, exclude sensitive fields, batch operations — those rules are codifiable regardless of whether the model "knows" it's applying them.

### Drift as a Distillation Viability Metric

Our functional equivalence data across three tasks validates this signal:

| Task | Textual Exact Match | Functional Equivalence | Drift Source |
|------|-------------------|----------------------|--------------|
| code_review (30 runs: Vertex×3 / Vertex→Ollama / structured-JSON, [Experiments 3–4](RESULTS.md)) | 33%[^exact] | 96.7% | 1 variant with duplicate dict key bug |
| data_extraction (30 runs, DeepInfra) | 83.3% | 100% | Description formatting, not values |
| iterative_refactor (5 runs, DeepInfra) | 60% | 100% | Exception handler style, not security fixes |
| iterative_refactor (3 runs, Ollama) | 100% | 100% | No drift (single GPU, deterministic) |

[^exact]: *Textual exact match* is the fraction of outputs byte-identical to the first run in the set (`exact_match_rate` in `drift measure`). For code_review, 30 implementer outputs are pooled across three pipeline configurations; 10/30 (33%) match the reference output textually despite 29/30 passing all functional tests.

The pattern is consistent: **textual drift wildly overstates functional drift.** Models vary in surface expression while applying the same underlying rules. The one genuine failure (code_review variant 5, duplicate dictionary key) produced a real bug — a 3.3% defect rate from drift alone.

This suggests a measurable threshold:

| Functional Equivalence Rate | Interpretation | Recommendation |
|----------------------------|----------------|----------------|
| >99% | Decisions are deterministic | Distill to code. LLM adds cost without value. |
| 95-99% | Decisions are stable with rare edge cases | Distill the common path, use LLM as fallback for edge cases. |
| 80-95% | Decisions vary meaningfully | LLM is doing real reasoning. Keep it, but add review gates. |
| <80% | Decisions are inconsistent | LLM is unreliable for this task. Improve the prompt, model, or task decomposition before deploying. |

The pattern: **use the LLM to discover the rules, codify them as deterministic code, use the LLM only for cases that fall outside the rules.** This gives you reproducibility and cost savings on the common path with the model as a fallback for genuinely ambiguous cases.

### Caveats

The threshold table assumes extraction- or transformation-heavy tasks with a meaningful functional test suite. Several factors shift the bands without changing the underlying pattern:

- **Temperature**: On `data_extraction`, functional equivalence holds at 100% through temp=0.1, then degrades linearly to 40% at temp=1.0 ([Experiment 6](RESULTS.md)). Thresholds here assume temp≈0.
- **Model version**: Upgrading Qwen2.5→Qwen3 dropped functional equivalence from 100% to 0% on `iterative_refactor` — a capability regression, not drift ([Experiment 5](RESULTS.md)). Re-measure after any model change.
- **Test-suite design**: Functional equivalence is only as good as the tests. Claude Sonnet scored 0% on `iterative_refactor` because it chose bcrypt over plaintext — incompatible with the test DB, not necessarily incorrect ([Experiment 7](RESULTS.md)).
- **Task type**: The gap between textual and functional metrics is largest where outputs have verifiable structure. Open-ended generation (the output *is* the deliverable) will show a smaller gap; distillation viability must be measured per task.

### A Concrete Example: Infrastructure Automation

The AnsibleClaw project (2026) converts Ansible modules into "skills" for AI agents. The skill generation turns out to be a deterministic transformation — Jinja2 templates populated from `ansible-doc --json` output. No LLM inference required. This works because the task (parsing structured documentation and generating wrapper code) is well-defined enough that the "rules" are fully extractable.

The interesting question is the next layer: should an agent use an LLM to decide *which* Ansible module to run with *which* parameters? Drift measurements on that decision process would tell you. If the agent consistently selects the same module and parameters for a given request (high functional equivalence), the decision logic can be distilled into a deterministic dispatcher. If it varies, the LLM is doing real reasoning and the variance itself is a risk — especially when the agent has write access to infrastructure.

### The Cost Implication

The distillation question reframes drift from a quality metric to a cost optimization tool:

- **High functional equivalence** → the LLM is a constant-function generator with expensive overhead. Replace it with code.
- **Low functional equivalence** → the LLM is doing non-trivial work. Keep it, measure the variance, and gate the output.

This gives `drift measure` a prescriptive output beyond "here's how much your outputs vary." It answers: **should you even be using an LLM for this task?**

## Drift as Task-Aware Compression

> **Note:** The practical application of this theory is implemented in the [logdigest](https://github.com/atyronesmith/logdigest) sibling repo — see logdigest's `docs/PREDIGESTION.md` for results and usage.

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

## Beyond the Entropy Floor: Intelligence as Compression

Drift's classifier distillation proved something specific: logistic regression captures the first level of prediction (keyword → category) with high accuracy at high confidence, but at n≈400 the coverage ceiling (~25%, root_cause_category at threshold 0.90) looked like exactly where single-level models run out of explanatory power. **Revised by later data**: `STATUS.md`'s Gate 1 results (n=500, threshold 0.95) show the ~25% figure was sample starvation, not a hard limit — plain bag-of-words logistic regression reaches 80.6% coverage on `severity` with more labeled data. At that scale BoW also beats hidden-state probes on hold-out accuracy for all three fields (CV 0.97/0.74/0.82 vs 0.93/0.65/0.70); probes win on coverage where vocabulary fails (`root_cause`: 1.4% BoW vs 34.1% hidden, 24×) and on latency (~80ms vs 21.5s full decode). The hard ceiling holds only where surface features genuinely don't carry the signal — deeper representations help there, not everywhere.

Drift's confidence-gated approach (classifier handles easy cases, LLM handles the rest) is a degenerate two-level hierarchy: one trivial model and one full-scale model with nothing in between. The most promising research direction is filling that middle — decomposing a large model into a hierarchy of smaller models, each capturing some of the deeper prediction levels without needing the full parameter count. MoE architectures do this implicitly (different experts for different inputs), speculative decoding does it explicitly (small model drafts, large model verifies), but neither truly decomposes the prediction hierarchy.

Drift's contribution is honest but bounded: it found the entropy floor and proved you can extract the deterministic layer. Everything above that floor is where the actual value lives — and that's not something regex and logistic regression will reach.

### Is it intelligence or just prediction?

LLMs are, at their core, autocomplete engines. Their value lies in encoding multiple levels of prediction — syntactic, semantic, pragmatic, domain-specific — compressed into billions of weights. The question is whether sufficiently accurate prediction across enough levels of abstraction is *indistinguishable* from understanding, and if so, whether the distinction matters.

Hutter's prize literally equates intelligence with compression: better compression = better prediction = better understanding. By that measure, an LLM is just the world's best compressor that accidentally learned to reason. Or "reason."

The emergence debate maps to phase transitions in compression. You add more data to a model and for a long time you get incrementally better predictions. Then at some point the model captures enough structure that it starts exploiting *relationships between patterns* — and suddenly it looks qualitatively different. But there's no magic threshold. It's the same gradient descent doing the same thing. We anthropomorphize the output once it gets good enough to fool us.

The uncomfortable version: maybe understanding is *always* just very obfuscated, highly accurate guessing — in humans too. We have the luxury of introspection that makes us feel like something deeper is happening. The LLM doesn't have that narrative layer, so we call it "just prediction."

This is why "emergent features" is the right framing. Nobody designed reasoning into the training objective. The model was trained to predict tokens. But a predictor's internal model can become complex enough that it captures structure the designers didn't explicitly encode — and at that point, the outputs start looking like intelligence regardless of whether anything resembling understanding is happening underneath.

## References

- Shaikh, Y. et al. "Collaborative Intelligence in AI: Evaluating the Performance of a Council of AIs on the USMLE." *PLOS Digital Health* 4(10): e0000787 (2025). https://doi.org/10.1371/journal.pdig.0000787 — see also [knowledge base note](knowledge/collaborative-intelligence-council-of-ais.md).
- Kim, Y. et al. "Towards a Science of Scaling Agent Systems." Google DeepMind (2025) — distilled in [17x Error Trap / bag of agents](knowledge/17x-error-trap-bag-of-agents.md) ("spend on workers, not orchestrator"; topology over quantity).
- Internal: [RESULTS.md](RESULTS.md), [BASIS.md](BASIS.md), [STATUS.md](STATUS.md) (Gate 1 classifier revision).
