/**
 * Pipeline model definitions for drift demo.
 *
 * Compares baseline (3 LLM calls) vs drift hybrid (1 LLM call)
 * for the AAP Ansible log triage pipeline.
 *
 * Data from drift Experiment 20 — 20-log corpus, qwen2.5:32b at temp=0.
 */

/** Typical productive output size fed to each guard as input context. */
export const PRODUCTIVE_TOKENS = { input: 1000, output: 1500 }

export const CHECKPOINT_TYPES = {
  llm:             { color: '#ef4444', label: 'LLM Guard',       speed: 0.3 },
  llm_agent:       { color: '#fb923c', label: 'LLM Agent Call',  speed: 0.3 },
  classifier:      { color: '#f59e0b', label: 'Classifier',      speed: 0.7 },
  static_analysis: { color: '#8b5cf6', label: 'Static Analysis', speed: 0.6 },
  deterministic:   { color: '#22c55e', label: 'Deterministic',    speed: 0.95 },
  crypto:          { color: '#06b6d4', label: 'Crypto',           speed: 0.9 },
}

export const MODELS = [
  {
    id: 'ansible_agentic',
    name: 'AAP Log Triage (Baseline)',
    description: '3 LLM calls per log — summarize, classify, route',
    source: 'RH AI Quickstart: ansible-log-analysis',
    checkpoints: [
      { phase: 'input',       checks: [
        { name: 'Embed & Cluster', type: 'classifier', latency_ms: 200, tokens_in: 0, tokens_out: 0, desc: 'Sentence-transformers + FAISS clustering' },
        { name: 'LLM Summarize', type: 'llm_agent', latency_ms: 500, tokens_in: 2000, tokens_out: 50, desc: 'Structured summary: [Error Type]: [description]' },
        { name: 'LLM Classify', type: 'llm_agent', latency_ms: 300, tokens_in: 550, tokens_out: 10, desc: 'Assign to 1 of 7 expert roles' },
        { name: 'LLM Router', type: 'llm_agent', latency_ms: 200, tokens_in: 350, tokens_out: 20, desc: 'Decide if more context needed' },
      ]},
      { phase: 'llm',         checks: [] },
      { phase: 'output',      checks: [
        { name: 'RAG Retrieval', type: 'classifier', latency_ms: 150, tokens_in: 0, tokens_out: 0, desc: 'FAISS cheat sheet lookup' },
      ]},
    ],
  },
  {
    id: 'ansible_drift',
    name: 'AAP Log Triage (Drift Hybrid)',
    description: '1 LLM call — rules replace classify + route, pre-digested input',
    source: 'drift Experiment 20 (measured — 20-log corpus)',
    checkpoints: [
      { phase: 'input',       checks: [
        { name: 'Embed & Cluster', type: 'classifier', latency_ms: 200, tokens_in: 0, tokens_out: 0, desc: 'Sentence-transformers + FAISS clustering' },
        { name: 'Pre-digest', type: 'deterministic', latency_ms: 5, tokens_in: 0, tokens_out: 0, desc: 'logdigest.ansible: 85% char savings measured' },
        { name: 'Hybrid Summarize', type: 'llm_agent', latency_ms: 315, tokens_in: 2013, tokens_out: 42, desc: 'LLM summary on pre-digested input (measured)' },
        { name: 'Rule Classify', type: 'deterministic', latency_ms: 1, tokens_in: 0, tokens_out: 0, desc: 'Keyword regex, 7 categories, 100% coverage (measured)' },
        { name: 'Rule Route', type: 'deterministic', latency_ms: 1, tokens_in: 0, tokens_out: 0, desc: 'Signal-presence heuristic (measured)' },
      ]},
      { phase: 'llm',         checks: [] },
      { phase: 'output',      checks: [
        { name: 'RAG Retrieval', type: 'classifier', latency_ms: 150, tokens_in: 0, tokens_out: 0, desc: 'FAISS cheat sheet lookup' },
      ]},
    ],
  },
]

/**
 * Probe annotations — connects baseline LLM calls to their drift replacements.
 * Each entry maps a baseline gate index to a drift gate index with probe results.
 * This is the core of drift: probe LLM output entropy, synthesize rules if deterministic.
 */
export const PROBE_ANNOTATIONS = [
  {
    baselineIdx: 2,  // LLM Classify
    driftIdx: 3,     // Rule Classify
    entropy: '0.00',
    method: 'drift probe',
    result: 'keyword regex',
    detail: '7 categories, 100% coverage',
  },
  {
    baselineIdx: 3,  // LLM Router
    driftIdx: 4,     // Rule Route
    entropy: '0.00',
    method: 'drift probe',
    result: 'signal-presence',
    detail: '90% accuracy vs 0% LLM',
  },
  {
    baselineIdx: 1,  // LLM Summarize
    driftIdx: 2,     // Hybrid Summarize
    entropy: '0.94',
    method: 'drift probe',
    result: 'keep LLM (high entropy)',
    detail: 'pre-digested input: 85% char savings',
  },
]

/** Flatten all checks for a model into a single ordered list with phase labels. */
export function flattenChecks(model) {
  const out = []
  for (const cp of model.checkpoints) {
    for (const c of cp.checks) {
      out.push({ ...c, phase: cp.phase })
    }
  }
  return out
}

/** Sum total latency for a model (one pipeline step). */
export function totalLatency(model) {
  return flattenChecks(model).reduce((s, c) => s + c.latency_ms, 0)
}

/** Sum total tokens for a model (one pipeline step). */
export function totalTokens(model) {
  const checks = flattenChecks(model)
  return {
    input: checks.reduce((s, c) => s + (c.tokens_in || 0), 0),
    output: checks.reduce((s, c) => s + (c.tokens_out || 0), 0),
  }
}

/** Count checks by type. */
export function countByType(model) {
  const counts = {}
  for (const c of flattenChecks(model)) {
    counts[c.type] = (counts[c.type] || 0) + 1
  }
  return counts
}
