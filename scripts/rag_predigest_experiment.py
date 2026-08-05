#!/usr/bin/env python3
"""RAG Pre-Digestion Experiment: measure how pre-digestion affects
embedding ingestion efficiency and retrieval quality.

Compares raw vs pre-digested Ansible logs through the full RAG pipeline:
chunk → embed → retrieve.

Usage:
    uv run python scripts/rag_predigest_experiment.py \
        --log-dir datasets/ansible_triage/inputs/ \
        --summary-dir datasets/ansible_summaries/ \
        -o results/rag_predigest_experiment.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ansible_log_predigest import classify_line, predigest

# Configuration
CHUNK_SIZE = 2048  # chars (~512 tokens)
CHUNK_OVERLAP = 200  # chars
EMBED_MODEL = "nomic-embed-text"
EMBED_BATCH_SIZE = 50  # chunks per API call
TOP_K = 5


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Fixed-size chunking with overlap."""
    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start >= len(text):
            break
    return chunks


def noise_ratio(chunk):
    """Fraction of a chunk's lines that are noise."""
    lines = [l for l in chunk.strip().split("\n") if l.strip()]
    if not lines:
        return 1.0
    noise = sum(1 for l in lines if classify_line(l).startswith("noise:"))
    return noise / len(lines)


def embed_batch(texts, host, model=EMBED_MODEL):
    """Embed texts via Ollama /api/embed. Returns list of vectors."""
    payload = json.dumps({"model": model, "input": texts})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(payload)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [
                "curl", "-s", "-X", "POST",
                f"{host}/api/embed",
                "-H", "Content-Type: application/json",
                "-d", f"@{tmp_path}",
            ],
            capture_output=True, text=True, timeout=300,
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if result.returncode != 0:
        raise RuntimeError(f"Embed failed: {result.stderr}")

    resp = json.loads(result.stdout)
    return resp["embeddings"]


def embed_all(chunks, host, batch_size=EMBED_BATCH_SIZE):
    """Embed all chunks in batches. Returns numpy array (n, dim)."""
    all_vecs = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vecs = embed_batch(batch, host)
        all_vecs.extend(vecs)
        if (i + batch_size) % 200 == 0 or i + batch_size >= len(chunks):
            print(f"    embedded {min(i + batch_size, len(chunks))}/{len(chunks)}")
    return np.array(all_vecs, dtype=np.float32)


def cosine_similarity_matrix(queries, db):
    """Compute cosine similarity between query vectors and DB vectors."""
    # Normalize
    q_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    d_norm = db / np.linalg.norm(db, axis=1, keepdims=True)
    return q_norm @ d_norm.T


def retrieve_top_k(query_vec, db_vectors, k=TOP_K):
    """Return indices of top-k most similar vectors."""
    sims = query_vec @ db_vectors.T
    top_k = np.argsort(sims)[-k:][::-1]
    return top_k, sims[top_k]


def main():
    parser = argparse.ArgumentParser(description="RAG pre-digestion experiment")
    parser.add_argument("--log-dir", required=True, help="Directory with raw .txt logs")
    parser.add_argument("--n", type=int, default=0, help="Number of logs (0 = all)")
    parser.add_argument("-o", "--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    all_logs = sorted(log_dir.glob("*.txt"))
    if args.n > 0:
        all_logs = all_logs[: args.n]

    db_logs = all_logs
    print(f"Corpus: {len(all_logs)} logs")
    print(f"Chunk size: {CHUNK_SIZE} chars, overlap: {CHUNK_OVERLAP} chars")
    print(f"Embedding model: {EMBED_MODEL}")
    print(f"{'=' * 70}\n")

    # ---------------------------------------------------------------
    # Step 1: Chunk DB logs (raw and pre-digested)
    # ---------------------------------------------------------------
    print("Step 1: Chunking DB logs...")

    raw_chunks = []       # (chunk_text, log_stem)
    digest_chunks = []    # (chunk_text, log_stem)

    raw_total_chars = 0
    digest_total_chars = 0
    raw_noise_chunks = 0
    total_raw_chunks = 0

    for idx, filepath in enumerate(db_logs):
        raw_text = filepath.read_text(errors="replace")
        digested_text = predigest(filepath)
        stem = filepath.stem

        r_chunks = chunk_text(raw_text)
        d_chunks = chunk_text(digested_text)

        for c in r_chunks:
            raw_chunks.append((c, stem))
            raw_total_chars += len(c)
            total_raw_chunks += 1
            if noise_ratio(c) > 0.9:
                raw_noise_chunks += 1

        for c in d_chunks:
            digest_chunks.append((c, stem))
            digest_total_chars += len(c)

        if (idx + 1) % 100 == 0 or idx + 1 == len(db_logs):
            print(f"  [{idx+1}/{len(db_logs)}] raw: {len(raw_chunks)} chunks, digest: {len(digest_chunks)} chunks")

    chunk_reduction = round((1 - len(digest_chunks) / len(raw_chunks)) * 100, 1)
    token_reduction = round((1 - digest_total_chars / raw_total_chars) * 100, 1)
    noise_chunk_pct = round(raw_noise_chunks / total_raw_chunks * 100, 1)

    print(f"\n  Raw chunks:       {len(raw_chunks):,}")
    print(f"  Digest chunks:    {len(digest_chunks):,}")
    print(f"  Chunk reduction:  {chunk_reduction}%")
    print(f"  Raw tokens:       ~{raw_total_chars // 4:,}")
    print(f"  Digest tokens:    ~{digest_total_chars // 4:,}")
    print(f"  Token reduction:  {token_reduction}%")
    print(f"  Noise chunks (>90% noise): {raw_noise_chunks:,} ({noise_chunk_pct}% of raw)")

    # ---------------------------------------------------------------
    # Step 2: Embed all chunks
    # ---------------------------------------------------------------
    print(f"\nStep 2: Embedding...")

    raw_texts = [c[0] for c in raw_chunks]
    digest_texts = [c[0] for c in digest_chunks]

    print(f"  Embedding {len(raw_texts)} raw chunks...")
    t0 = time.time()
    raw_vectors = embed_all(raw_texts, args.ollama_host)
    raw_embed_time = time.time() - t0
    print(f"  Raw embedding: {raw_embed_time:.1f}s")

    print(f"  Embedding {len(digest_texts)} digest chunks...")
    t0 = time.time()
    digest_vectors = embed_all(digest_texts, args.ollama_host)
    digest_embed_time = time.time() - t0
    print(f"  Digest embedding: {digest_embed_time:.1f}s")

    speedup = raw_embed_time / digest_embed_time if digest_embed_time > 0 else 0
    print(f"  Embedding speedup: {speedup:.1f}x")

    # Normalize for cosine similarity
    raw_norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
    raw_norms[raw_norms == 0] = 1
    raw_vectors_norm = raw_vectors / raw_norms

    digest_norms = np.linalg.norm(digest_vectors, axis=1, keepdims=True)
    digest_norms[digest_norms == 0] = 1
    digest_vectors_norm = digest_vectors / digest_norms

    # ---------------------------------------------------------------
    # Step 3: Retrieval quality — cohesion test
    # ---------------------------------------------------------------
    # Measure: do chunks from the same log cluster together?
    # For N sampled chunks, find top-k neighbors (excluding self).
    # Report fraction of neighbors from the same log.
    # Pre-digestion should improve cohesion: noise removed, signal
    # concentrated, so same-log chunks embed closer together.
    N_SAMPLE = min(200, len(raw_chunks), len(digest_chunks))
    print(f"\nStep 3: Retrieval quality — cohesion test ({N_SAMPLE} sampled chunks)...")
    print(f"  Metric: what fraction of a chunk's top-{TOP_K} neighbors come from the same log?")

    raw_stems = [c[1] for c in raw_chunks]
    digest_stems = [c[1] for c in digest_chunks]

    # Sample indices (evenly spaced)
    raw_sample_idx = [i * len(raw_chunks) // N_SAMPLE for i in range(N_SAMPLE)]
    digest_sample_idx = [i * len(digest_chunks) // N_SAMPLE for i in range(N_SAMPLE)]

    # Raw cohesion
    raw_same_log_hits = 0
    raw_total_neighbors = 0
    raw_noise_in_neighbors = 0

    for idx in raw_sample_idx:
        qvec = raw_vectors_norm[idx]
        sims = qvec @ raw_vectors_norm.T
        sims[idx] = -1  # exclude self
        top_k = np.argsort(sims)[-TOP_K:][::-1]
        query_stem = raw_stems[idx]

        for i in top_k:
            raw_total_neighbors += 1
            if raw_stems[i] == query_stem:
                raw_same_log_hits += 1
            if noise_ratio(raw_texts[i]) > 0.9:
                raw_noise_in_neighbors += 1

    # Digest cohesion
    digest_same_log_hits = 0
    digest_total_neighbors = 0
    digest_noise_in_neighbors = 0

    for idx in digest_sample_idx:
        qvec = digest_vectors_norm[idx]
        sims = qvec @ digest_vectors_norm.T
        sims[idx] = -1  # exclude self
        top_k = np.argsort(sims)[-TOP_K:][::-1]
        query_stem = digest_stems[idx]

        for i in top_k:
            digest_total_neighbors += 1
            if digest_stems[i] == query_stem:
                digest_same_log_hits += 1
            if noise_ratio(digest_texts[i]) > 0.9:
                digest_noise_in_neighbors += 1

    raw_cohesion = round(raw_same_log_hits / raw_total_neighbors, 3) if raw_total_neighbors else 0
    digest_cohesion = round(digest_same_log_hits / digest_total_neighbors, 3) if digest_total_neighbors else 0

    retrieval_results = {
        "method": "cohesion_test",
        "n_sampled_chunks": N_SAMPLE,
        "top_k": TOP_K,
        "raw_cohesion": raw_cohesion,
        "digest_cohesion": digest_cohesion,
        "raw_same_log_hits": raw_same_log_hits,
        "raw_total_neighbors": raw_total_neighbors,
        "digest_same_log_hits": digest_same_log_hits,
        "digest_total_neighbors": digest_total_neighbors,
        "raw_noise_in_neighbors": raw_noise_in_neighbors,
        "digest_noise_in_neighbors": digest_noise_in_neighbors,
    }

    print(f"\n  {'Metric':<30s}  {'Raw':>10s}  {'Digest':>10s}")
    print(f"  {'-'*55}")
    print(f"  {'Same-log cohesion':<30s}  {raw_cohesion:>10.3f}  {digest_cohesion:>10.3f}")
    print(f"  {'Same-log hits / total':<30s}  {raw_same_log_hits:>4d}/{raw_total_neighbors:<5d}  {digest_same_log_hits:>4d}/{digest_total_neighbors:<5d}")
    print(f"  {'Noise in neighbors':<30s}  {raw_noise_in_neighbors:>10d}  {digest_noise_in_neighbors:>10d}")

    # ---------------------------------------------------------------
    # Step 4: Cost projection
    # ---------------------------------------------------------------
    raw_tokens = raw_total_chars // 4
    digest_tokens = digest_total_chars // 4

    # OpenAI ada-002: $0.10/M tokens, text-embedding-3-small: $0.02/M
    embed_pricing = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
        "ada-002": 0.10,
    }

    scales = [10_000, 50_000, 100_000]
    avg_raw_tokens_per_log = raw_tokens / len(all_logs)
    avg_digest_tokens_per_log = digest_tokens / len(all_logs)

    cost_projections = {}
    for model_name, price_per_m in embed_pricing.items():
        model_costs = {}
        for n in scales:
            raw_cost = n * avg_raw_tokens_per_log * price_per_m / 1_000_000
            digest_cost = n * avg_digest_tokens_per_log * price_per_m / 1_000_000
            saved_yr = (raw_cost - digest_cost) * 12
            model_costs[f"{n//1000}k_monthly"] = {
                "raw_monthly": round(raw_cost, 2),
                "digest_monthly": round(digest_cost, 2),
                "saved_yearly": round(saved_yr, 2),
            }
        cost_projections[model_name] = model_costs

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------
    output = {
        "corpus": {
            "n_logs": len(all_logs),
        },
        "ingestion": {
            "raw_chunks": len(raw_chunks),
            "digest_chunks": len(digest_chunks),
            "chunk_reduction_pct": chunk_reduction,
            "raw_tokens": raw_tokens,
            "digest_tokens": digest_tokens,
            "token_reduction_pct": token_reduction,
            "raw_noise_chunks": raw_noise_chunks,
            "raw_noise_chunk_pct": noise_chunk_pct,
            "raw_embed_time_s": round(raw_embed_time, 1),
            "digest_embed_time_s": round(digest_embed_time, 1),
            "embed_speedup": f"{speedup:.1f}x",
        },
        "retrieval": retrieval_results,
        "cost_projections": cost_projections,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))

    print(f"\n{'=' * 70}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Chunk reduction:    {chunk_reduction}% ({len(raw_chunks):,} → {len(digest_chunks):,})")
    print(f"Token reduction:    {token_reduction}% (~{raw_tokens:,} → ~{digest_tokens:,})")
    print(f"Noise chunks:       {raw_noise_chunks:,} of {len(raw_chunks):,} raw chunks ({noise_chunk_pct}%)")
    print(f"Embed speedup:      {speedup:.1f}x ({raw_embed_time:.1f}s → {digest_embed_time:.1f}s)")
    if retrieval_results:
        print(f"Cohesion:           raw={retrieval_results['raw_cohesion']:.3f}  digest={retrieval_results['digest_cohesion']:.3f}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
