# On the Resilience of LLM-Based Multi-Agent Collaboration with Faulty Agents

**Paper**: Proceedings of the 42nd International Conference on Machine Learning (ICML), PMLR 267, 2025
**Authors**: Huang, J.-t., Zhou, J., Jin, T., Zhou, X., Chen, Z., Wang, W., Yuan, Y., Sap, M.
**Affiliations**: CUHK, Tsinghua, CMU, Peking, Renmin, CUHK Shenzhen
**Code**: https://github.com/CUHK-ARISE/MAS-Resilience

---

## What This Paper Does

Systematically studies how different multi-agent system topologies handle faulty agents — those that produce erroneous or misleading outputs. Tests three structural archetypes (linear, flat, hierarchical) across six MAS frameworks and four downstream tasks. Proposes two error-introduction methods and two defense mechanisms.

## Three Topologies Tested

1. **Linear** (A->B->C): One-way chain. Agents pass output forward only. Systems: MetaGPT, Self-collab.
2. **Flat** (A<->B<->C): Full mutual communication. All agents talk to all others. Systems: Camel, SPP.
3. **Hierarchical** (A->(B<->C)): Combines one-way and mutual. Hub coordinates workers. Systems: MAD, AgentVerse.

## Error Introduction Methods

1. **AUTOTRANSFORM**: Rewrites an agent's profile to introduce stealthy errors while preserving original functionality. The agent itself generates errors naturally.
2. **AUTOINJECT**: Intercepts messages between agents and directly injects specific errors. Provides precise control over error rate (P_m) and density (P_e).

## Key Findings

### Topology Resilience Rankings

**Hierarchical > Flat > Linear** — consistent across both GPT-3.5 and GPT-4o:
- Hierarchical: 5.5% average performance drop
- Flat: 10.5% drop
- Linear: 23.7% drop

The hierarchical advantage comes from a higher-level agent (judge/evaluator) that aggregates multiple perspectives, creating a natural error-correction checkpoint.

### Task Vulnerability Rankings

**Code generation (-22.6%) > Math (-9.9%) > Text evaluation (-5.4%) > Translation (-4.7%)**

Objective, formal tasks (code, math) are far more sensitive to errors than subjective tasks (translation, evaluation). Even minor errors in code generation cascade into compilation/logic failures.

### Surprising Finding: Errors Can Help

In some systems (MAD, Camel, AgentVerse), deliberately injected errors *improved* performance by up to 12.1%. Two mechanisms:
1. **Double Checking**: Obvious errors prompt other agents to verify and correct, catching pre-existing errors too.
2. **Divergent Thinking**: In debate systems, errors break agents out of repetitive loops, introducing productive disagreement.

### Error Characteristics

- **Frequency matters more than density**: More faulty messages (P_m) causes greater drops than more errors per message (P_e)
- **Semantic errors > Syntactic errors**: Systems handle syntactic errors well (via code training), but semantic errors (logically sound but wrong) evade detection
- **LLMs prioritize natural language over code**: A comment saying "the bug has been corrected" causes agents to accept buggy code

### Hub Node Fragility

High-level planners are catastrophic failure points. In AgentVerse with two faulty agents: Solver faulty drops from 28->20; add Planner faulty drops to 12. The planner's error propagates irrecoverably through all subsequent reasoning.

## Defense Mechanisms

1. **Challenger**: Adds challenge capability to each agent's profile (inspired by AUTOTRANSFORM). Agents can question received messages.
2. **Inspector**: An additional agent that intercepts all messages, checks for errors, and corrects them (inspired by AUTOINJECT).

Combined (Challenger + Inspector), they recover up to **96.4%** of performance lost to faulty agents. Most effective on the two weakest topologies (linear and flat).

### Additional rounds don't help
Number of communication rounds showed no correlation with resilience. More rounds ≠ better recovery.

## Relevance to Drift

**Core overlap with our research:**

- Their **linear topology** is exactly our pipeline architecture — sequential one-way chains. Their finding that linear is the *least* resilient (23.7% drop) is the worst case for our setup.
- The **hierarchical advantage** maps to the DeepMind finding (17x paper) that centralized coordination contains error amplification. Both papers converge on: hub-and-spoke beats flat or chain.
- Their **semantic vs syntactic error** distinction maps directly to our observation that textually different but semantically equivalent outputs propagate differently. Semantic drift (our main phenomenon) is harder to detect than syntactic variance.
- The **"errors can help"** finding is counterintuitive but aligns with the Council of AIs paper — diversity in responses can improve outcomes through productive disagreement.
- **Hub node fragility** = our finding that step 1 variance dominates the pipeline. The first step is the "planner" — its variance propagates to all downstream steps.

**Implications for our experiments:**
- Our 3-step pipeline is a linear topology, the most vulnerable structure. Adding a verification/correction step (Inspector pattern) could significantly reduce drift.
- The finding that frequency of errors matters more than density suggests our infrastructure variance (which affects every call, not just some) would be more damaging than occasional large deviations.
- Testing hierarchical topologies (fan-out with synthesizer) should show better variance containment than our current linear chain.
