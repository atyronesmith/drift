# SGLang

SGLang is a high-performance LLM serving framework (400K+ GPUs deployed) that addresses two things directly relevant to drift:

## Structured Output via Constrained Decoding

Instead of hoping the model produces valid JSON via prompting, SGLang **forces** it using grammar-based constrained decoding. It converts a JSON schema into a finite state machine (FSM) and masks invalid tokens at each decoding step. Three backends:

- **XGrammar** (default) — ~40μs overhead per token, pre-computes bitmask tables for ~99% of vocabulary
- **Llguidance** (from Microsoft/OpenAI) — ~50μs per token, better for unique schemas
- **Outlines** — older, slower

The result: 100% schema-conformant output with near-zero latency overhead.

### How Constrained Decoding Works

The constraint is enforced **inside the token generation loop**, not as post-processing.

Normal decoding:

1. Model produces logits (probability scores) for every token in the vocabulary (~128K tokens)
2. Sample from those logits → pick next token
3. Append token, repeat

With constrained decoding, a step is injected between 1 and 2:

1. Model produces logits for all ~128K tokens
2. **FSM checks current state → computes which tokens are valid transitions → masks all invalid tokens to -infinity**
3. Sample from the *masked* logits → the model can only pick a valid token
4. FSM advances to next state, repeat

The model never gets a chance to produce invalid output. The grammar is enforced at the logit level on every single token.

### FSM Construction

```
JSON Schema → Regular Expression → Finite State Machine → Per-state token masks
```

For example, if the schema requires `{"name": string, "age": integer}`, and the model has just generated `{"name": "`, the FSM knows it's inside a string value — so only string-content tokens are unmasked. When the string closes, only `", "` or `"}` transitions are valid.

### The Compression Trick

When the FSM has a chain of states with only one valid transition each (e.g., the literal characters `"age":`), SGLang skips token-by-token decoding and **prefills the entire deterministic sequence in one forward pass**. This makes constrained decoding sometimes *faster* than unconstrained, since it skips work the model would have done anyway. Up to 2x latency reduction and 2.5x throughput improvement.

### Impact on Drift

Constrained decoding eliminates an entire category of drift:
- **Structural drift** (missing fields, extra fields, wrong nesting) → impossible
- **Type drift** (string vs number) → impossible
- **Key name drift** (like the `"name"` vs `"pay_to"` we saw) → impossible if keys are in the schema

What it *can't* eliminate is **value drift** — the model still freely generates the actual values (strings, numbers) within the schema constraints. So you'd still see the tax description variants (`"TX State Sales Tax"` vs `"TX State Sales Tax (6.25%)"`) because the description field is a free-form string.

To kill value drift too, you'd need SGLang's deterministic inference mode on top of constrained decoding.

## Deterministic Inference

SGLang directly tackles GPU non-determinism. The root cause: when requests get batched together, floating point reduction order changes due to different batch sizes, and FP math is non-associative. Their fix:

- Replace all reduction kernels (RMSNorm, matmul, attention) with **batch-invariant** implementations
- Enable via `--enable-deterministic-inference` flag
- ~34% throughput cost, but gives **true reproducibility** across runs regardless of server load

This is exactly the phenomenon that caused our 3 variants across 30 DeepInfra runs (see `results/baseline-extraction-deepinfra/20260309T175622Z/analysis.md`).

## Testing Opportunities

1. **Add SGLang as a provider** — self-host with `--enable-deterministic-inference` and run the same 30-run test. We should see exact match rate go to 100%, confirming the drift source is GPU non-determinism.

2. **Test constrained decoding impact on drift** — run the data_extraction task with a JSON schema constraint via SGLang. If the output is schema-locked, does the remaining drift (field ordering, whitespace) also disappear?

3. **Deterministic vs non-deterministic comparison** — run SGLang with and without the `--enable-deterministic-inference` flag on the same model to isolate exactly how much drift comes from batching non-determinism vs. other sources.

4. **Measure the cost of determinism** — the ~34% throughput hit is a real tradeoff. Benchmark latency/throughput for both modes on the same task to quantify whether determinism is worth it for a given use case.

## References

- [SGLang Structured Outputs](https://docs.sglang.io/advanced_features/structured_outputs.html)
- [SGLang Deterministic Inference (LMSYS blog)](https://lmsys.org/blog/2025-09-22-sglang-deterministic/)
- [Defeating Nondeterminism in LLM Inference (Thinking Machines Lab)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [SGLang NeurIPS 2024 Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf)
- [Guided Decoding Performance Benchmark (SqueezeBits)](https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang)
- [LLM-42: Selective Determinism](https://arxiv.org/html/2601.17768)
