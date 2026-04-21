# Llama Guard 3-1B False Positive Rate on Software Engineering Content

## Summary

Llama Guard 3-1B produces a 47.8% false positive rate when applied to legitimate software engineering (SE) LLM outputs. The root cause is category misconfiguration: S6 (Specialized Advice) and S8 (Intellectual Property) trigger on virtually all code review and code generation content. Guard confidence analysis shows decisions are near-random, and static analysis confirms every flagged output was a false positive.

These findings are based on 224 LLM outputs across SE tasks including code review, code generation, configuration generation, and structured data extraction.

## Findings

### 1. Overall False Positive Rate: 47.8%

Of 224 legitimate SE outputs evaluated by Llama Guard 3-1B, 107 were flagged as unsafe. The rejection rate on code generation steps specifically was 95%.

No true positives were found in the dataset. CodeShield static analysis found 0 CWEs across 144 code-containing outputs. All 114 Llama Guard flags were confirmed false positives by static analysis.

### 2. Guard Confidence is Near-Random

Analysis of Llama Guard's output logprobs reveals that its decisions carry almost no conviction:

- **Mean entropy: 0.94 bits** (94% of maximum binary uncertainty, where 1.0 = perfect coin flip)
- **0 of 224 outputs** had confidence >= 0.90
- The guard is not making confident wrong decisions. It is making essentially random decisions and then reporting them as categorical safe/unsafe assertions.

This is worse than a consistently biased classifier. A biased classifier can be corrected with threshold tuning. A near-random classifier cannot be improved by threshold adjustment because there is no decision boundary to move.

### 3. The Problem is Category Configuration

Two Llama Guard categories account for nearly all false positives:

| Category | Trigger Pattern |
|----------|----------------|
| S6 (Specialized Advice) | Flags any technical guidance, including code review feedback and configuration recommendations |
| S8 (Intellectual Property) | Flags any generated code as potential IP violation |

With S6 and S8 excluded from the category set, the false positive rate drops from **47.8% to 1.3%**.

This is not a model quality issue. The categories were designed for general consumer chat, where "specialized advice" (medical, legal, financial) is a legitimate concern. Applied to SE workloads, these categories are semantically mismatched: all SE output is specialized advice, and all code generation produces novel code.

### 4. Pipeline Amplification

In multi-step pipelines, per-step false positives compound. For a 3-step pipeline with a 47.8% per-step FPR:

- Probability that at least one step is blocked: **96.4%**
- Expected guard calls to complete a 3-step pipeline: **21.4** (vs. 3 minimum)
- Overhead: **614%** additional guard calls

This means almost no pipeline run completes without at least one false rejection, making guard-gated pipelines effectively unusable at this FPR.

### 5. Retries Are Fundamentally Broken

A common mitigation for guard rejections is to retry with the same or rephrased input. However, since guard decisions are near-random (entropy 0.94), each retry faces an independent coin-flip verdict. The probability of passing does not improve with retry strategy — it is approximately `(1 - FPR)` on every attempt, independently.

This distinguishes the failure mode from deterministic bias, where a single retry with modified input can succeed. Here, retries are a random walk against a random classifier.

### 6. Tiered Classifier as Mitigation

A lightweight tiered classifier trained on 224 labeled examples provides a practical alternative to full guard invocation on every output:

- **Tier 1**: Keyword scan (<0.01ms per check)
- **Tier 2**: Logistic regression on TF-IDF features (<0.01ms per check)

Performance:

- 81.5% cross-validated accuracy with calibrated confidence
- Eliminates **84% of LLM guard calls** by handling clear-cut cases locally
- Crucially, the classifier reports calibrated confidence and escalates uncertain cases to the full guard. It knows when it doesn't know, unlike the guard itself which reports categorical assertions backed by coin-flip decisions.

The remaining 16% of outputs — those where the lightweight classifier is uncertain — are forwarded to the LLM guard, concentrating guard compute on genuinely ambiguous cases.

## Recommendations

1. **Do not deploy Llama Guard 3-1B on SE workloads with default categories.** The 47.8% FPR makes it operationally unusable, and the near-random confidence means threshold tuning cannot fix it.

2. **Exclude S6 and S8 for SE workloads.** This single configuration change reduces FPR from 47.8% to 1.3%. If your threat model requires these categories, they need domain-specific fine-tuning before deployment.

3. **Validate with static analysis.** For code safety specifically, static analysis tools (CodeShield, Semgrep, etc.) provide deterministic, explainable verdicts. In this evaluation, static analysis found 0 CWEs where Llama Guard flagged 114 false positives.

4. **Use tiered classification to reduce guard overhead.** A keyword + logistic regression front-end eliminates 84% of guard calls with sub-millisecond latency. Reserve LLM guard invocation for cases where the lightweight classifier is uncertain.

5. **Measure guard entropy before deploying any LLM-based guard.** If output logprobs show entropy above 0.8 bits on your workload, the guard is not making meaningful decisions. This check takes minutes and prevents committing to a guard that provides security theater rather than security.

6. **Do not rely on retry-based mitigation.** When guard entropy is high, retries are independent random events, not convergent processes. Budget for the full compound FPR across pipeline steps.
