# the vendor Developer Hub: Constrained Agents and the Intermediary Pattern

## What Is the vendor Developer Hub?

the vendor Developer Hub (the platform) is the vendor's enterprise distribution of [Backstage](https://backstage.io/), the CNCF-hosted internal developer portal originally created by Spotify. It provides a single pane of glass for developers — a software catalog tracking every service, API, and component in the organization, along with templates, documentation (TechDocs), and a plugin ecosystem.

Starting with the platform 1.8, the platform has evolved from a passive portal into an **active agent intermediary**. Through its Lightspeed AI assistant and Model Context Protocol (MCP) integration, the platform now sits between LLM-powered agents and the systems those agents act upon.

This architectural position makes the platform the natural place to apply the same constraint philosophy that SGLang applies at the token level — but at the **action level**.

## Architecture: The Intermediary Stack

the platform's AI architecture is a layered intermediary between the user, the LLM, and the enterprise systems:

```
┌─────────────────────────────────────────────────────┐
│  Developer (human)                                  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  the platform Frontend                                      │
│  (Lightspeed chat plugin)                           │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  Lightspeed Core Service (LCS)                      │
│  ┌────────────────────────────────────────────────┐ │
│  │ Chat history, user feedback, operational config│ │
│  └────────────────────────────────────────────────┘ │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  Llama Stack                                        │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────┐ │
│  │ Inference │  │ Guardrails│  │ Agent/MCP/RAG    │ │
│  │ (vLLM)   │  │ (Llama    │  │ orchestration    │ │
│  │          │  │  Guard)   │  │                  │ │
│  └──────────┘  └───────────┘  └────────┬─────────┘ │
└─────────────────────────────────────────┼───────────┘
                                          │
┌─────────────────────────────────────────▼───────────┐
│  MCP Server (the platform plugins)                          │
│  ┌────────────┐  ┌────────────┐  ┌───────────────┐ │
│  │ Software   │  │ TechDocs   │  │ Future: write │ │
│  │ Catalog    │  │ content    │  │ operations,   │ │
│  │ (read-only)│  │ (read-only)│  │ RBAC-gated    │ │
│  └────────────┘  └────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────┘
```

### Layer 1: Lightspeed Core Service (LCS)

The LCS replaced the earlier Road-Core Service (RCS) in the platform 1.8. It acts as the primary intermediary, managing:

- **Chat history** — maintains conversation state across sessions
- **User feedback** — captures human evaluation of agent responses
- **Operational configuration** — controls which models, tools, and guardrails are active

The LCS is deployed as a sidecar container alongside the Llama Stack service. It extends Llama Stack with the platform-specific features while delegating inference, safety, and tool orchestration to the stack.

### Layer 2: Llama Stack

[Llama Stack](https://llama-stack.readthedocs.io/) is an open-source framework (now described by the vendor as "an emerging industry standard for agentic AI systems") that provides unified APIs for:

- **Inference** — via vLLM or other backends (Bring Your Own Model)
- **Guardrails / Safety Shields** — Llama Guard for content classification, Prompt Guard for injection detection
- **Agent orchestration** — including MCP tool calling and RAG
- **Model evaluation** — benchmarking and quality assessment

the platform configures Llama Stack via a `run.yaml` that specifies inference models, safety models (default: `llama-guard3:8b`), and shield configurations.

### Layer 3: MCP Server

The MCP (Model Context Protocol) server is the interface through which agents interact with the platform's data. the platform 1.8 ships three MCP tools:

- **Software Catalog tool** — lists Backstage entities (Components, Systems, Resources, APIs, Users, Groups) with metadata including name, description, type, owner, tags, and dependencies
- **fetch-techdocs** — lists all entities with TechDocs and their metadata
- **retrieve-techdocs-content** — retrieves the actual documentation content for a specific catalog entity
- **analyze-techdocs-coverage** — calculates documentation coverage percentages, identifying gaps

All current MCP tools are **read-only by design**. Write operations require additional security considerations and are not yet available.

## The Constrained Execution Pattern

### The Parallel to Constrained Decoding

In our SGLang research (`docs/sglang.md`), we documented how constrained decoding works:

```
Token level:   Model logits → FSM mask → only valid tokens selectable
```

the platform applies the same principle at the action level:

```
Action level:  Agent intent → MCP tool boundary → only permitted actions executable
```

Just as an FSM prevents the model from generating a syntactically invalid JSON token, the MCP server boundary prevents an agent from executing an action that isn't exposed as a tool. The agent cannot "hallucinate" an action — it can only call tools that exist and are registered.

This is constraint through **surface area reduction**:

| Constraint Layer | What It Limits | Drift/Risk It Eliminates |
|-----------------|----------------|-------------------------|
| **Constrained decoding** (SGLang) | Which tokens the model can select | Structural drift: invalid JSON, wrong types, missing fields |
| **Guardrails** (Llama Guard) | Which content passes input/output filters | Safety drift: harmful, biased, or policy-violating content |
| **MCP tool boundary** (the platform) | Which actions the agent can take | Action drift: unauthorized operations, scope creep, unintended side effects |
| **RBAC** (the platform/Backstage) | Which entities/resources are accessible | Authorization drift: agents acting beyond their permitted scope |
| **Read-only mode** (MCP default) | Whether agents can mutate state | Mutation drift: unintended writes, deletions, or modifications |

### Read-Only First: The Right Default

the platform's decision to launch MCP tools in read-only mode is significant. It mirrors how constrained decoding works — start by eliminating the most dangerous category of invalid output (structural violations / write operations), then selectively open up expressiveness where needed.

The Ansible Automation Platform MCP server shows how this evolves: administrators configure the server in either **read-only mode** (safe querying and monitoring) or **read-write mode** (agents can execute jobs). The read-write mode inherits user permissions via RBAC, ensuring the agent can only execute actions permitted by both server configuration and the user's defined permissions.

This is the action-level equivalent of a JSON schema constraint — the schema defines what fields can exist (which actions are callable), and RBAC defines what values are valid (which resources are accessible).

### Multi-Step Constraint Chaining

the platform's architecture enables what the MCP documentation calls "multi-agent orchestration" — chaining information from different services for multi-step analysis. Each step in the chain passes through the constraint stack:

1. **Input guardrail** — Llama Guard classifies the user's request; Prompt Guard checks for injection attempts
2. **Agent reasoning** — the LLM decides which MCP tools to call
3. **Tool boundary** — the agent can only call registered, approved tools
4. **RBAC gate** — the tool checks whether the user/agent has permission for the specific entity
5. **Execution** — the tool executes (read-only in current implementation)
6. **Output guardrail** — Llama Guard classifies the response before it reaches the user

Every step is a constraint point. The agent's freedom is bounded at each layer, reducing the surface area for both drift and unsafe behavior.

## How Drift Testing Applies to the platform

Our drift measurements showed that LLM non-determinism produces different outputs for the same inputs (see [RESULTS.md](RESULTS.md)). In the the platform context, this means:

### Where Drift Still Matters

Even with the platform's constraint layers, the LLM's reasoning step (step 2 above) remains unconstrained. The model decides *which* tools to call and *how* to interpret the results. Drift at this layer can produce:

- **Tool selection drift** — the agent calls different tools on different runs for the same query
- **Interpretation drift** — the agent summarizes the same catalog data differently across runs
- **Parameter drift** — the agent passes different filter parameters to the same tool

These are analogous to the value drift we observed in our DeepInfra experiments — the structure was constrained (same tools available), but the model's choices within that structure varied.

### Where the platform's Constraints Eliminate Drift

- **Action drift** — impossible to take an action that isn't a registered tool
- **Authorization drift** — impossible to access resources beyond RBAC scope
- **Mutation drift** — impossible to write when tools are read-only
- **Safety drift** — content filtered by Llama Guard on input and output

### Recommendation: Drift Test the Reasoning Layer

Teams deploying Lightspeed or custom agents on the platform should drift-test the reasoning layer specifically:

1. Ask the same question 30 times through Lightspeed
2. Record which MCP tools the agent selects and in what order
3. Record how the agent interprets and summarizes the results
4. Measure exact match rate on both tool selection and final response

If tool selection is inconsistent, the agent's reasoning needs improvement (better system prompts, constrained output for tool selection, or deterministic inference). If tool selection is consistent but interpretation varies, the output formatting can be addressed with structured output constraints.

## The Bigger Picture: Layered Constraint Architecture

The combination of the platform's intermediary pattern with inference-level constraints (like SGLang's constrained decoding and deterministic inference) creates a defense-in-depth approach to agent reliability:

```
Layer 1: Deterministic inference    → eliminates GPU non-determinism
Layer 2: Constrained decoding       → eliminates structural output drift
Layer 3: Guardrails (Llama Guard)   → eliminates unsafe content
Layer 4: MCP tool boundary          → eliminates unauthorized actions
Layer 5: RBAC                       → eliminates scope violations
Layer 6: Read-only default          → eliminates unintended mutations
Layer 7: Human oversight            → catches everything else
```

No single layer is sufficient. Constrained decoding doesn't prevent an agent from calling the wrong tool. RBAC doesn't prevent an agent from producing inconsistent output. Deterministic inference doesn't prevent an agent from generating unsafe content. Together, they create a system where the agent is constrained at every level — from the individual tokens it generates to the enterprise systems it touches.

This is what it means to build agentic systems that, as the user put it, "don't kill us."

## Counter-Example: The Anti-Pattern of Removing Constraints

The AnsibleClaw project (2026) illustrates the opposite approach. It converts Ansible's 3,000+ modules into "skills" that AI agents invoke directly, explicitly removing the playbook — the reviewable, auditable YAML artifact — from the execution path. The architecture is: Agent → Skill wrapper → Ansible module → Infrastructure, with no review gate between the agent's decision and execution.

The project frames this as a feature: "AI agents don't need playbooks; agents need skills." But the playbook was never just a formatting convention. It's the governance layer — the artifact that gets committed to git, linted, reviewed in a pipeline, and audited after execution. Removing it means the agent chooses which modules to run, with which parameters, against which hosts, and the only constraint is the agent's own judgment (plus Ansible's idempotency, which only means running something twice doesn't break things — not that the thing it does is what you wanted).

This is the architectural opposite of the platform's constrained execution pattern. Where the platform starts read-only and gates write operations through RBAC, AnsibleClaw starts with full execution capability and relies on the LLM to make safe choices. Where the platform uses MCP as a chokepoint for policy enforcement, AnsibleClaw uses a custom skill abstraction that bypasses standard protocol boundaries.

An MCP server wrapping Ansible modules would provide the same agent accessibility with the governance layer intact:

- **Approval gates** — agent proposes the module invocation, human or CI confirms before execution
- **Allowlists** — restrict which modules and target hosts the agent can access
- **Dry-run default** — execute with Ansible's `--check` mode, require explicit approval for `--apply`
- **Audit logging** — every invocation recorded with full parameters and results
- **Scope restrictions** — per-session or per-role limits on what the agent can do

### The Token Cost Objection and Meta-Tool Discovery

A common argument against MCP for large tool surfaces like Ansible (3,000+ modules) is token cost — every tool definition consumes context window tokens. Exposing 3,000 tools with parameter schemas would eat the entire context before the agent starts reasoning.

The solution is a tiered discovery pattern using meta-tools:

| Tool | Purpose |
|------|---------|
| `ansible_search` | Takes a query, returns matching module names and brief descriptions |
| `ansible_module_info` | Takes a module name, returns the full parameter schema |
| `ansible_execute` | Takes a module name and parameters, runs it (with approval gate) |

Three tool definitions in the context window instead of 3,000. The agent discovers what it needs dynamically. The tool schemas are generated deterministically from `ansible-doc --json` output — no LLM inference required for the discovery layer.

This pattern applies to any large tool surface, not just Ansible. The key insight: discovery is a routing problem, not a generation problem, and it doesn't require the entire catalog in context.

## Current Limitations and Future Direction

- **MCP tools are read-only** — write operations are acknowledged as needed but require further security considerations. Expect gated write capabilities in future the platform releases.
- **Llama Guard is content-focused** — it classifies content safety, not factual accuracy. An agent can produce a factually wrong but "safe" answer. This is where drift testing fills the gap.
- **No structured output enforcement at the platform layer** — the LCS doesn't currently enforce JSON schema constraints on the LLM's responses. Adding SGLang-style constrained decoding at the Llama Stack inference layer would close this gap.
- **Constrained decoding can be an attack surface** — recent research (arXiv:2503.24191) shows that malicious grammar constraints can bypass safety mechanisms, achieving 100% jailbreak rates on GPT-4o. This means the guardrail layer (Llama Guard) must run *after* constrained decoding, not be replaced by it.
- **MCP-as-a-Service (MCPaaS)** — the vendor's longer-term vision includes a managed layer for hosting, observing, and auditing MCP servers centrally, with an AI Hub unifying models, agents, guardrails, and MCP servers as composable AI assets.

## References

- [the vendor Developer Hub product page](https://developers.the vendor.com/products/the platform)
- [the platform background and concepts](https://developers.the vendor.com/articles/2026/01/12/red-hat-developer-hub-background-and-concepts)
- [the platform 1.8 announcement](https://www.the vendor.com/en/blog/red-hat-developer-hub-18-delivers-context-aware-ai-faster-self-service-and-scalable-governance)
- [What's new in the platform 1.8](https://developers.the vendor.com/blog/2025/11/12/whats-new-red-hat-developer-hub-18)
- [MCP in the platform: Chat with your catalog](https://developers.the vendor.com/articles/2025/11/10/mcp-red-hat-developer-hub-chat-your-catalog)
- [the platform 1.8 MCP documentation](https://docs.the vendor.com/en/documentation/red_hat_developer_hub/1.8/html/interacting_with_model_context_protocol_tools_for_red_hat_developer_hub/assembly-model-context-protocol-tools)
- [the platform 1.8 Lightspeed documentation](https://docs.the vendor.com/en/documentation/red_hat_developer_hub/1.8/html-single/interacting_with_red_hat_developer_lightspeed_for_red_hat_developer_hub/index)
- [Build an AI agent to automate TechDocs](https://developers.the vendor.com/articles/2025/05/30/build-ai-agent-automate-techdocs-red-hat-developer-hub)
- [Building effective AI agents with MCP](https://developers.the vendor.com/articles/2026/01/08/building-effective-ai-agents-mcp)
- [Ansible Automation Platform MCP server](https://www.the vendor.com/en/blog/it-automation-agentic-ai-introducing-mcp-server-red-hat-ansible-automation-platform)
- [Llama Stack safety guardrails](https://llama-stack.readthedocs.io/en/latest/building_applications/safety.html)
- [Implement AI safeguards with Llama Stack (Python)](https://developers.the vendor.com/articles/2025/08/26/implement-ai-safeguards-python-and-llama-stack)
- [Constrained Decoding as Attack Surface (arXiv:2503.24191)](https://arxiv.org/html/2503.24191v1)
- [LlamaFirewall (arXiv:2505.03574)](https://arxiv.org/pdf/2505.03574)
- [the vendor agentic AI executive guide](https://www.the vendor.com/en/resources/executive-guide-ai-agentic-ebook)
