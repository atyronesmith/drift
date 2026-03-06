# Metacognitive Self-Correction for Multi-Agent System via Prototype-Guided Next-Execution Reconstruction

**Paper**: Under review at ICLR 2026
**Authors**: Anonymous (double-blind review)
**Source**: Downloaded PDF (2148_Metacognitive_Self_Correc.pdf)

---

## What This Paper Does

Introduces MASC, a metacognitive layer for LLM-based multi-agent systems that performs real-time, unsupervised, step-level error detection and self-correction. Key innovation: reframes error detection as history-conditioned anomaly scoring rather than requiring labeled error data.

## Core Architecture

Three stages executed at each pipeline step:

1. **Contextual Encoding**: Converts raw inputs (task query, agent roles, interaction history) into task-aware vector embeddings using a pre-trained encoder + learnable projections.

2. **Prototype-Guided Reconstruction**: The detection module that combines:
   - **Next-Execution Reconstruction**: Uses a frozen LLM backbone to predict the *next* step's embedding from historical context. Anomalous steps violate learned causal dependencies, producing higher reconstruction error.
   - **Prototype Enhancement**: Maintains a learnable prototype vector representing the centroid of normal step embeddings. Provides stability when historical context is sparse (early steps).

3. **Anomaly-Triggered Self-Correction**: When anomaly score exceeds threshold, a dedicated correction agent revises the flagged output before it propagates downstream. The corrected output replaces the original in the shared history.

## Anomaly Scoring

```
s(t) = alpha * ||x_hat_t - x_t||^2 + beta * (1 - cos(x_hat_t, p))
```

Combines L2 reconstruction error with cosine-based prototype misalignment. Higher scores = greater anomaly likelihood.

## Training

Fully unsupervised — trained only on normal trajectories. Loss combines:
- Reconstruction loss (MSE between predicted and realized embeddings)
- Prototype loss (cosine similarity regularization toward learned prototype)

No step-level error labels needed.

## Key Results

**Error Detection** (Who&When benchmark):
- AUC-ROC of 77.84% (handcrafted) and 75.62% (automated) without ground truth
- Outperforms all baselines including supervised models (BERT classifier, LLM classifier)

**Framework Integration** (6 benchmarks: MMLU, GSM8K, AQuA, MultiArith, SVAMP, HumanEval):
- Average +1.29% performance gain when plugged into existing MAS
- LLM-Debate: 87.53% -> 88.89%
- Consistent gains across Chain, Complete, Random, and Debate topologies

**Ablation**: Both reconstruction and prototype components essential. Removing either degrades detection, especially for early-step errors where context is limited.

## Problem Analysis (Motivating Findings)

1. **Context-dependence**: Error steps look similar to normal steps in isolation. BERT embeddings show inter-cluster distance of 0.25 vs intra-cluster 1.45 — nearly indistinguishable without history.

2. **Early-step difficulty**: Substantial fraction of errors occur in first 20% of steps, where historical context is sparse. Motivates the prototype mechanism.

3. **Lack of metacognitive awareness**: Existing MAS frameworks (chain, fully-connected, random topologies) all suffer >50% performance drops from a single faulty agent. Even critic/reviewer roles don't prevent cascading errors.

## Relevance to Drift

**Directly applicable to our pipeline error containment question:**

- Their anomaly scoring approach is a formalization of what we're trying to detect empirically — when a pipeline step produces output that deviates from the expected causal flow. Our edit distance metrics are a crude proxy for what their reconstruction error measures precisely.
- The **prototype mechanism** solves the cold-start problem we'd face at step 1: with no history, you need a reference point for "normal." This is analogous to our structured JSON schema acting as a prototype — constraining what normal output looks like.
- Their finding that **single faulty agents cause >50% performance drops** across all topologies echoes our observation that upstream variance (from infrastructure) propagates and amplifies through pipeline depth.
- The **correction agent pattern** (revise-and-replace before propagation) is a more sophisticated version of our "structured output sandwich" — instead of constraining output format, they detect and fix the content itself.

**Design implications:**
- Their approach is architecture-agnostic and plug-in — could theoretically be added as a monitoring layer to our pipeline system
- The reconstruction-based detection could be adapted to detect infrastructure-induced variance (our core thesis) by training the prototype on deterministic local runs and flagging cloud-induced deviations
