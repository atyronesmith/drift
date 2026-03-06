# Knowledge Base — Papers & References

Papers analyzed for the Drift project, organized by topic.

## Bibliography

1. **Xie et al.** "From Spark to Fire: Modeling and Mitigating Error Cascades in LLM-Based Multi-Agent Collaboration." arXiv:2603.04474v1, March 2026. [Synopsis](spark-to-fire-error-cascades.md) | [Paper](https://arxiv.org/html/2603.04474v1)
   — Formal propagation dynamics model for error cascades in multi-agent systems. Identifies three vulnerability classes (cascade amplification, topological fragility, consensus inertia). Proposes genealogy-based governance layer achieving 0.89+ defense rate.

2. **Moran, S.** "Why Your Multi-Agent System is Failing: Escaping the 17x Error Trap of the 'Bag of Agents'." Towards Data Science, January 2026. [Synopsis](17x-error-trap-bag-of-agents.md) | [Article](https://towardsdatascience.com/why-your-multi-agent-system-is-failing-escaping-the-17x-error-trap-of-the-bag-of-agents/)
   — Distills DeepMind's scaling laws for multi-agent systems. 17.2× error amplification in unstructured networks vs 4.4× with centralized control. 10 agent archetypes organized into functional planes. Key rules: 45% saturation threshold, spend on workers not orchestrator, topology > quantity.

   - **Kim, Y. et al.** "Towards a Science of Scaling Agent Systems." Google DeepMind, December 2025. *(Underlying research paper)*

3. **Anonymous** "Metacognitive Self-Correction for Multi-Agent System via Prototype-Guided Next-Execution Reconstruction." Under review, ICLR 2026. [Synopsis](metacognitive-self-correction-masc.md)
   — MASC framework for real-time, unsupervised step-level error detection and self-correction. Uses Next-Execution Reconstruction + Prototype-Guided Enhancement to detect anomalous steps without labeled data. Up to 8.47% AUC-ROC gains; +1.29% average accuracy when plugged into existing MAS frameworks.

4. **Huang, J.-t. et al.** "On the Resilience of LLM-Based Multi-Agent Collaboration with Faulty Agents." ICML 2025, PMLR 267. [Synopsis](resilience-faulty-agents.md) | [Code](https://github.com/CUHK-ARISE/MAS-Resilience)
   — Tests three MAS topologies (linear, flat, hierarchical) against faulty agents. Hierarchical most resilient (5.5% drop) vs flat (10.5%) vs linear (23.7%). Proposes Challenger + Inspector defenses recovering 96.4% of lost performance. Surprising finding: deliberately injected errors can improve some systems.

5. **Wei, Y. et al.** "Beyond Input Guardrails: Reconstructing Cross-Agent Semantic Flows for Execution-Aware Attack Detection." arXiv:2603.04469v1, March 2026. [Synopsis](beyond-input-guardrails-mascope.md)
   — MAScope framework shifts MAS defense from static input filtering to execution-aware analysis via Cross-Agent Semantic Flows. Detects OWASP Top 10 agentic attacks with F1 85.3% (node-level) and 66.7% (path-level) by reconstructing behavioral trajectories from fragmented cross-agent events.

6. **Shaikh, Y. et al.** "Collaborative Intelligence in AI: Evaluating the Performance of a Council of AIs on the USMLE." PLOS Digital Health 4(10), October 2025. [Synopsis](collaborative-intelligence-council-of-ais.md) | [Code](https://github.com/councilofai/project-saru)
   — "Council of AIs" with five GPT-4 instances deliberating on USMLE questions achieves 94.7% accuracy (vs 78.2% individual). Deliberation corrects errors 53% of the time (5.0x odds ratio). Semantic entropy converges to zero through rounds. Key constraint: requires at least one correct initial response.
