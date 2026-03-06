# Why Your Multi-Agent System is Failing: Escaping the 17x Error Trap of the "Bag of Agents"

**Article**: Towards Data Science, January 30, 2026
**Author**: Sean Moran
**URL**: https://towardsdatascience.com/why-your-multi-agent-system-is-failing-escaping-the-17x-error-trap-of-the-bag-of-agents/
**Based on**: Kim et al. "Towards a Science of Scaling Agent Systems" (Google DeepMind, December 2025)

---

## What This Article Does

Distills DeepMind's large multi-factorial scaling study into practical rules for building multi-agent systems. Proposes a taxonomy of 10 agent archetypes organized into functional planes. Core argument: topology of coordination matters more than agent quantity.

## The 17x Error Trap

Kim et al. report **up to 17.2× error amplification** in poorly coordinated "bag of agents" networks. Centralized coordination contains this to ~4.4× by acting as a circuit breaker.

The "bag of agents" anti-pattern has three characteristics:
- **Flat topology**: every agent talks to every other agent, no hierarchy
- **Noisy chatter**: agents echo and validate each other's mistakes
- **Open-loop execution**: no verification before passing data downstream

## Three Golden Rules (from DeepMind data)

1. **The 17x Rule**: Unstructured networks amplify errors exponentially. A centralized orchestrator suppresses this by acting as a single verification point.

2. **The Tool-Coordination Trade-off**: More tools require more grounding. A context plane (retriever) ensures agents don't guess how to use tools.

3. **The 45% Saturation Point**: Agent coordination yields highest returns when single-agent baseline is below 45%. If the base model already hits 80%, more agents may introduce more noise than value.

## 10 Agent Archetypes

Organized into functional planes:

**Control Plane**:
- **Orchestrator** — project manager, delegates, doesn't do work
- **Monitor** — health/safety officer, watches for loops/budget overruns/drift

**Planning Plane**:
- **Planner** — decomposes goals into task graphs

**Context Plane**:
- **Retriever** — fetches relevant docs/context on demand
- **Memory Keeper** — summarizes and stores important outcomes

**Execution Plane**:
- **Executor** — does the actual work (code, API calls, text generation)
- **Synthesiser** — cleans up executor output into structured results

**Assurance Plane**:
- **Evaluator** — checks objective correctness (does code compile? does JSON validate?)
- **Critic** — checks subjective risks (security vulnerabilities, edge cases)

**Mediation Plane**:
- **Mediator** — tie-breaker when evaluator and critic disagree

## Key Experimental Findings

**Five topologies tested**: Single-Agent (SAS), Independent MAS, Decentralized MAS, Centralized MAS, Hybrid MAS.

**Results are task-dependent**:
- **Finance-Agent** (highly decomposable): MAS delivers biggest gains — Centralized +80.8%, Decentralized +74.5% over SAS
- **BrowseComp-Plus** (dynamic web navigation): modest gains, topology-sensitive — Decentralized +9.2%, Centralized essentially flat +0.2%, Independent collapses −35%
- **PlanCraft** (strictly sequential planning): every MAS variant degrades −39% to −70% — coordination overhead provides no parallel advantage
- **Workbench**: marginal movement −11% to +6%

**Saturation at ~4 agents**: Performance rises initially then plateaus. Additional agents contribute little after this point.

**Mixed-model teams**:
- Weak orchestrator + strong workers can beat all-strong teams (Anthropic: 0.42 vs 0.32, +31%)
- But this doesn't generalize — for OpenAI/Gemini, heterogeneous centralized setups degraded
- Decentralized debate is surprisingly robust to mixed capabilities — voting averages out weaker agents
- **Spend money on workers, not the orchestrator** (but validate on your model family)

## The Open-Loop → Closed-Loop Transformation

The Assurance → Planner feedback loop is the biggest performance differentiator. Without it:
- Errors escape downstream (open-loop)
- Agents echo and amplify mistakes

With it:
- Errors are caught and recycled into improvements (closed-loop)
- The system self-corrects before passing to next stage

## Cost Model

```
Total MAS cost ≈ Work cost + Coordination cost
Work cost ∝ n × k  (agents × iterations per agent)
Coordination cost ∝ rounds × fan-out × context growth
```

Decentralized and hybrid topologies get expensive fast: debate inflates both message volume and context length (pay twice — agents write more AND read more). Coordination approaches n² effect.

## Architecture Scaling Law

DeepMind trained a predictive model on coordination traces (efficiency, overhead, error amplification, redundancy). It forecasts best topology with R² ≈ 0.513, selecting correct strategy for ~87% of held-out configs. Practical shift: run small probe configurations first, then commit budget to predicted winners.

## Relevance to Drift

**Directly relevant to our pipeline experiments:**

- Our homogeneous pipeline (0.47 → 0.57 → 0.73 drift) is a minimal demonstration of the error amplification they describe — though ours is accidental variance, not errors per se
- Our structured pipeline (0.27 → 0.52 → 0.34) demonstrates their Assurance Plane principle — constraining output format acts as a circuit breaker
- Their finding that **centralized coordination contains amplification to 4.4×** vs 17.2× unstructured aligns with our finding that structured JSON output constrains variance
- The **45% saturation point** suggests our experiments should test tasks at different difficulty levels — easy tasks may not benefit from pipeline structure at all
- Their **mixed-model finding** (spend on workers not orchestrator) is exactly what our mixed pipeline tests — expensive Claude at step 1, cheaper Ollama at steps 2-3

**Design implications for our next experiments:**
- Add an Evaluator step with structured pass/fail output (Assurance Plane) to test closed-loop error containment
- Test fan-out topologies (parallel agents + synthesizer) to measure the n² coordination cost
- The saturation at ~4 agents suggests our 3-step pipeline is in the useful range — going beyond 4-5 steps may just add noise

## References Cited

- Kim et al. (2025) "Towards a Science of Scaling Agent Systems" (Google DeepMind)
- Cursor Team (2026) "Scaling long-running autonomous coding"
- Wang et al. (2023) "A Survey on Large Language Model based Autonomous Agents"
- Wu et al. (2023) "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation"
- Yao et al. (2022) "ReAct: Synergizing Reasoning and Acting in Language Models"
- Shinn et al. (2023) "Reflexion: Language Agents with Verbal Reinforcement Learning"
- Brooks (1975) "The Mythical Man-Month"
