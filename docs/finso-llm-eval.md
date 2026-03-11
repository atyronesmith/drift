# Evaluation and Benchmarking Suite for Financial Large Language Models and Agents

**Lin et al. (SecureFinAI Lab, Columbia University + Linux Foundation + PyTorch Foundation + the vendor), NeurIPS 2025**

Paper: arXiv:2602.19073v1 (Feb 2026)

## Synopsis

The paper presents a lifecycle framework for evaluating LLMs in financial services, organized into three phases:

### 1. Exploration (2023)

Can LLMs do finance at all? They built the Open FinLLM Leaderboard on HuggingFace, benchmarking models across 42 financial datasets in 7 categories: information extraction, sentiment analysis, QA, text generation, risk management, forecasting, and decision-making. General-purpose LLMs struggle with financial numerical reasoning — FinGPT achieves 85% accuracy vs. Perplexity's 55%, and Google AI Overview gets 43% of finance summaries wrong.

### 2. Readiness (2024)

Financial agents in practice. Four systems showcased:

- **FinSearch** — RAG-based agentic search over financial websites (air-gapped for data security)
- **Tutor Agent** — Financial education via MCP protocol, accessing textbooks/exams
- **FinSight** — Multi-agent system with a Coordinator orchestrating Sentiment, Event, and Volatility agents. Uses *metacognitive reasoning* — the coordinator self-assesses confidence and triggers self-correcting loops or human escalation when uncertain
- **ECC Analyzer** — Extracts trading signals from earnings call transcripts (chunked summarization + RAG for 60+ minute calls)

### 3. Governance (2025)

- **AgentOps framework** — inner loop (dev/test with trajectory tracing, glass-box evaluation) + outer loop (production observability with LLM-as-a-Judge scoring) + feedback loop connecting the two
- **Model Openness Framework (MOF)** — 3-tier classification (Open Model → Open Tooling → Open Science) to combat "openwashing" where models claim to be open but withhold training data or licensing info
- Data lineage verification (no Material Nonpublic Information in training), reproducibility for audit, and EU AI Act compliance

## Relevance to Drift

The paper explicitly calls out "non-deterministic behaviors — common in agentic workflows" as a governance concern. Their FinSight agent's metacognitive self-correction loops are essentially a runtime mitigation for drift — if the output quality drops below a threshold, it re-runs. The AgentOps inner loop metrics (Tool Selection Accuracy, Context Adherence, Step-wise Latency) are complementary to what drift measures. We measure *variability* of outputs; they measure *correctness* of agent trajectories. Both matter for production fintech.

## References

- [Paper (arXiv)](https://arxiv.org/abs/2602.19073)
- [Open FinLLM Leaderboard (HuggingFace)](https://huggingface.co/spaces/SecureFinAI/Open-Financial-LLM-Leaderboard)
- [Model Openness Framework](https://isitopen.ai/)
