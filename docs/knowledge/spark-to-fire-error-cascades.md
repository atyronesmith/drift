# From Spark to Fire: Modeling and Mitigating Error Cascades in LLM-Based Multi-Agent Collaboration

**Paper**: arXiv:2603.04474v1, March 2026
**Authors**: Xie, Zhu, Zhang, Zhu, Ye, Qi, Chen, Zhou (City University of Macau / Minzu University of China)
**URL**: https://arxiv.org/html/2603.04474v1

---

## What This Paper Does

Models how small errors in multi-agent LLM systems propagate and solidify into system-wide false consensus. Tests six mainstream frameworks (LangChain, MetaGPT, LangGraph, CrewAI, AutoGen, CAMEL) and proposes a governance layer to stop it.

## Core Model

Represents multi-agent collaboration as a directed dependency graph. Each agent either adopts or rejects a falsehood at each round. The key equation:

```
ℛ ≈ βρ(A) / δ
```

Where β = adoption rate, ρ(A) = spectral radius of the adjacency matrix (topology), δ = decay/self-correction rate. When βρ(A) > δ, the system is **supercritical** — errors amplify instead of dying out.

This is the formal version of what we're measuring empirically with drift curves.

## Three Vulnerability Classes

1. **Cascade Amplification**: Errors compound through iteration. Five of six frameworks hit 100% infection despite having explicit reviewer/QA roles. Mesh topologies contaminate in ≤3 rounds.

2. **Topological Fragility**: Hub nodes are catastrophic failure points. LangGraph showed 10.3× disparity — 100% infection when hub is compromised vs. 9.7% from a leaf node. Architecture determines blast radius.

3. **Consensus Inertia**: Once established, false consensus resists correction. Intervention at t=2 costs 1.0 polluted rounds; at t=6 it costs 3.9. Late corrections fight accumulated history.

## The Defense: Genealogy-Based Governance

A message-layer plugin (doesn't modify the agent framework) that:

1. **Decomposes** messages into atomic claims
2. **Labels** each claim: Green (confirmed), Red (contradicts known facts), Yellow (uncertain)
3. **Routes** by policy: strict mode verifies everything, balanced mode focuses on hub nodes, speed mode tags-and-passes
4. **Blocks** messages containing Red atoms, returns rewrite directives

Results: defense success rate went from 0.32 (self-reflection alone) to 0.89+ (speed mode) to 0.94 (strict mode). Cost: 50-100% more tokens and 50-110% more latency.

## Relevance to Drift

**Direct relevance — this is the adversarial version of our problem.** We measure accidental variance propagation through pipelines; they measure intentional error propagation. But the underlying dynamics are identical:

- Their cascade amplification = our drift curve growing with pipeline depth
- Their topological fragility = our finding that hub/combiner nodes in fan-out architectures amplify variance
- Their consensus inertia = the observation that once a pipeline commits to a path, downstream steps reinforce rather than correct it

**Key insight for our work**: Their governance layer's atomic claim decomposition + tri-state labeling is a more sophisticated version of the "structured output sandwich" pattern. Instead of constraining the output format (our JSON schema approach), they constrain at the claim level — decompose free-form text into verifiable atomic statements. This is a stronger variance absorber because it works on prose output, not just structured output.

**Their spectral risk criterion** (βρ(A)/δ) could give us a theoretical framework for predicting which pipeline topologies will amplify drift vs. absorb it, based on the graph structure alone.

## Limitations They Acknowledge

- Binary adoption model misses partial internalization or semantic drift (exactly what we see — our outputs are semantically equivalent but textually different)
- Stationarity assumption — adoption/decay rates treated as constant, but infrastructure load makes them time-varying (our core thesis)
- Only tested on code-centric and logic tasks
