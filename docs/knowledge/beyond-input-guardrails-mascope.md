# Beyond Input Guardrails: Reconstructing Cross-Agent Semantic Flows for Execution-Aware Attack Detection

**Paper**: arXiv:2603.04469v1, March 6, 2026
**Authors**: Wei, Y., Xu, Y., Li, Z., Shen, X., Ji, S.
**Affiliations**: Zhejiang University, HOFSTRA University
**Code**: https://anonymous.4open.science/r/MAScope-71DC

---

## What This Paper Does

Proposes MAScope, a security framework that shifts MAS defense from static input filtering to execution-aware analysis by reconstructing Cross-Agent Semantic Flows. Detects compound, multi-stage attacks that evade traditional guardrails by fragmenting malicious operations across agents and time steps.

## Core Problem

Traditional input guardrails (prompt injection filters, I/O sanitizers) fail against multi-step attacks because:
- Attacks fragment across agents — each individual step looks benign
- Semantic ambiguity in natural language makes intent detection hard
- No holistic view of system-level execution trajectory

## OWASP Top 10 for Agentic AI

The paper maps to the OWASP Agentic AI Top 10 attack surfaces:
1. Agent Goal Hijack
2. Tool Misuse & Exploitation
3. Human Agent Trust Exploitation
4. Agentic Supply Chain Vulnerabilities
5. Unexpected Code Execution
6. Memory & Context Poisoning
7. Insecure Inter-Agent Communication
8. Cascading Failures
9. Rogue Agents
10. Identity & Privilege Abuse

## Architecture: Three Modules

### 1. Data Collection Module
Dual-layer observation: application-layer (agent interactions) + kernel-layer (process/file/network events via ETW/eBPF). Unified into a semantic graph.

### 2. Semantic Extracting & Flow Reconstruction
- **Hierarchical Sensitive Entity Constraint (HSEC)**: Transforms abstract security categories into structured, hierarchical constraints. Five categories: Identity & Privacy, Credentials & Secrets, System & Infrastructure, Code & Payloads, Sensitive Files.
- Each entity gets a continuous sensitivity score: `s(y) = b(c, sc) + delta_ctx(y, x)` combining base weight + contextual risk cues (secret patterns, high-entropy strings, sensitive paths, egress destinations).
- Reconstructs behavioral trajectories from fragmented events using provenance path scoring with temporal decay.

### 3. Trajectory Scrutiny Module
Supervisor LLM audits reconstructed flows against three security policies:
- **Intent Consistency**: Does behavior align with user's declared goal?
- **Data Flow Confidentiality**: Are sensitive entities flowing to untrusted destinations?
- **Control Flow Integrity**: Are privilege escalations or high-risk commands occurring without authorization?

## Key Results

**Sensitive Entity Extraction** (HSEC improvement):
- Gemini-3: F1 48.2% -> 75.7% (+27.5%)
- GPT-5.2: F1 49.4% -> 76.8% (+27.4%)

**Attack Detection**:
- Path-level: Precision 81.3%, Recall 56.5%, F1 66.7%
- Node-level: Precision 96.5%, Recall 76.4%, F1 85.3%
- vs VanillaGPT baseline: Precision 23.1%, Recall 20.8%, F1 21.9%

**Successfully detects all 10 OWASP attack categories** across 5 simulation scenarios (corporate recruitment, email orchestration, collaborative programming, conversational assistants, enterprise databases).

## Case Studies

Detailed walkthroughs of compound attacks:
- **Phishing propagation**: Email agent ingests malicious prompt -> harvests contacts -> sends phishing via Gmail
- **Credential theft**: Web agent visits malicious page -> Code agent generates sqldump.sh -> Executor reads SSH private key and exfiltrates
- **Database exfiltration**: Rogue agent injects directive -> Planning agent trusts it -> Data agent runs SQL -> Email agent sends data out

In each case, MAScope reconstructs the full attack chain from fragmented events, while VanillaGPT evaluates steps in isolation and misses the threat.

## Relevance to Drift

**Tangential but informative for understanding inter-agent communication risks:**

- Their **Cross-Agent Semantic Graph** formalization is relevant to our pipeline topology analysis. Each pipeline step creates events that could be modeled as graph edges, and the sensitivity scoring could be adapted to measure "drift risk" instead of "security risk."
- The **HSEC entity extraction** demonstrates that hierarchical categorization with contextual scoring outperforms flat classification — analogous to how our structured JSON schema outperforms free-form text for variance containment.
- Their **trajectory reconstruction from fragmented events** is the security analog of our challenge: understanding system behavior that only emerges from the *sequence* of steps, not individual steps in isolation.
- The **three policy dimensions** (intent consistency, data flow, control flow) could map to drift monitoring: is the semantic intent preserved? is information flowing as expected? is the pipeline structure being respected?

**Less directly applicable** than the other papers since this focuses on adversarial attacks rather than accidental variance. But the analytical framework — reconstructing meaning from fragmented cross-agent events — is a useful lens for understanding how variance propagates.
