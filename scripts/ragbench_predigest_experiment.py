#!/usr/bin/env python3
"""RAGBench pre-digestion experiment: measure how general-purpose
pre-digestion affects retrieval quality on standard benchmarks.

Builds a retrieval index from RAGBench documents, applies pre-digestion,
and compares retrieval metrics (nDCG@5, Recall@5) with and without.

Usage:
    uv run python scripts/ragbench_predigest_experiment.py \
        --dataset techqa \
        -o results/ragbench_techqa.json

    uv run python scripts/ragbench_predigest_experiment.py \
        --dataset emanual \
        -o results/ragbench_emanual.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from hashlib import md5
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------
# General pre-digestion (domain-agnostic)
# ---------------------------------------------------------------

SEPARATOR_PATTERN_CHARS = set("*=-_~#.+")


def is_separator_line(line):
    """Lines that are just visual separators (*****, ------, etc.)."""
    stripped = line.strip()
    if len(stripped) < 4:
        return False
    unique = set(stripped)
    return len(unique) <= 2 and unique.issubset(SEPARATOR_PATTERN_CHARS | {" "})


def build_boilerplate_set(documents, threshold=0.05):
    """Find lines that appear in >threshold fraction of documents.

    These are cross-document boilerplate (headers, footers, disclaimers).
    """
    n_docs = len(documents)
    if n_docs < 10:
        return set()

    line_doc_count = Counter()
    for doc in documents:
        seen = set()
        for line in doc.split("\n"):
            line = line.strip()
            if len(line) > 20 and line not in seen:
                seen.add(line)
                line_doc_count[line] += 1

    min_count = max(int(n_docs * threshold), 3)
    return {line for line, cnt in line_doc_count.items() if cnt >= min_count}


def dedup_documents(documents):
    """Remove exact duplicate documents. Returns (deduped_list, n_removed)."""
    seen = set()
    deduped = []
    removed = 0
    for doc in documents:
        h = md5(doc.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(doc)
        else:
            removed += 1
    return deduped, removed


def predigest_document(text, boilerplate_lines=None):
    """Apply general pre-digestion to a single document.

    Techniques:
    - Strip cross-document boilerplate lines
    - Remove separator lines (*****, ------)
    - Collapse excessive blank lines
    - Normalize whitespace
    """
    if boilerplate_lines is None:
        boilerplate_lines = set()

    lines = text.split("\n")
    cleaned = []
    blank_run = 0

    for line in lines:
        stripped = line.strip()

        # Skip boilerplate
        if stripped in boilerplate_lines:
            continue

        # Skip separator lines
        if is_separator_line(stripped):
            continue

        # Collapse blank lines (max 1)
        if not stripped:
            blank_run += 1
            if blank_run <= 1:
                cleaned.append("")
            continue
        else:
            blank_run = 0

        cleaned.append(line.rstrip())

    return "\n".join(cleaned).strip()


# ---------------------------------------------------------------
# Embedding (reuses Ollama pattern from rag_predigest_experiment)
# ---------------------------------------------------------------

EMBED_MODEL = "nomic-embed-text"
EMBED_BATCH_SIZE = 50
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 200


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


def embed_batch(texts, host, model=EMBED_MODEL):
    """Embed texts via Ollama /api/embed."""
    payload = json.dumps({"model": model, "input": texts})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(payload)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", f"{host}/api/embed",
             "-H", "Content-Type: application/json", "-d", f"@{tmp_path}"],
            capture_output=True, text=True, timeout=300,
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    if result.returncode != 0:
        raise RuntimeError(f"Embed failed: {result.stderr}")
    return json.loads(result.stdout)["embeddings"]


def embed_all(chunks, host, batch_size=EMBED_BATCH_SIZE):
    """Embed all chunks in batches."""
    all_vecs = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vecs = embed_batch(batch, host)
        all_vecs.extend(vecs)
        if (i + batch_size) % 200 == 0 or i + batch_size >= len(chunks):
            print(f"    embedded {min(i + batch_size, len(chunks))}/{len(chunks)}")
    return np.array(all_vecs, dtype=np.float32)


# ---------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------

def ndcg_at_k(retrieved_relevant, k):
    """Compute nDCG@k given a list of booleans (is position i relevant?)."""
    dcg = sum(
        (1.0 if rel else 0.0) / np.log2(i + 2)
        for i, rel in enumerate(retrieved_relevant[:k])
    )
    # Ideal: all relevant first
    n_relevant = sum(retrieved_relevant[:k])
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(n_relevant, k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved_doc_ids, relevant_doc_ids, k):
    """Fraction of relevant docs found in top-k."""
    retrieved_set = set(retrieved_doc_ids[:k])
    if not relevant_doc_ids:
        return 0.0
    return len(retrieved_set & set(relevant_doc_ids)) / len(relevant_doc_ids)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAGBench pre-digestion experiment")
    parser.add_argument("--dataset", required=True,
                        choices=["techqa", "emanual", "cuad", "covidqa", "msmarco"],
                        help="RAGBench sub-dataset")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--ollama-host",
                        default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--boilerplate-threshold", type=float, default=0.05,
                        help="Fraction of docs a line must appear in to be boilerplate")
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading RAGBench/{args.dataset} ({args.split})...")
    ds = load_dataset("galileo-ai/ragbench", args.dataset, split=args.split)
    print(f"  {len(ds)} examples loaded")

    # ---------------------------------------------------------------
    # Step 1: Build document corpus + ground truth
    # ---------------------------------------------------------------
    print("\nStep 1: Building document corpus...")

    # Pool all unique documents, track which are relevant per question
    doc_to_id = {}  # doc text → doc_id
    all_doc_texts = []
    queries = []
    query_relevant_ids = []  # per query, set of relevant doc_ids

    for ex in ds:
        question = ex["question"]
        relevant_ids = set()
        for doc in ex["documents"]:
            if doc not in doc_to_id:
                doc_id = len(all_doc_texts)
                doc_to_id[doc] = doc_id
                all_doc_texts.append(doc)
            relevant_ids.add(doc_to_id[doc])
        queries.append(question)
        query_relevant_ids.append(relevant_ids)

    print(f"  {len(all_doc_texts)} unique documents, {len(queries)} queries")
    total_raw_chars = sum(len(d) for d in all_doc_texts)
    print(f"  Total raw chars: {total_raw_chars:,}")

    # ---------------------------------------------------------------
    # Step 2: Pre-digest
    # ---------------------------------------------------------------
    print("\nStep 2: Pre-digesting...")

    # Dedup
    deduped_docs, n_dupes_removed = dedup_documents(all_doc_texts)
    print(f"  Exact duplicates removed: {n_dupes_removed}")

    # Build boilerplate set from the corpus
    boilerplate = build_boilerplate_set(all_doc_texts, args.boilerplate_threshold)
    print(f"  Boilerplate lines detected (>{args.boilerplate_threshold*100:.0f}% threshold): {len(boilerplate)}")
    if boilerplate:
        for line in sorted(boilerplate, key=lambda l: -len(l))[:5]:
            print(f"    [{len(line):3d} chars] {line[:70]}...")

    # Apply pre-digestion
    digested_doc_texts = [predigest_document(doc, boilerplate) for doc in all_doc_texts]

    total_digest_chars = sum(len(d) for d in digested_doc_texts)
    char_reduction = round((1 - total_digest_chars / total_raw_chars) * 100, 1) if total_raw_chars else 0
    print(f"  Raw chars: {total_raw_chars:,}")
    print(f"  Digest chars: {total_digest_chars:,}")
    print(f"  Char reduction: {char_reduction}%")

    # Build mapping from original doc_id to digested text
    # (doc_to_id maps raw text → id, digested text is at same index)

    # ---------------------------------------------------------------
    # Step 3: Chunk
    # ---------------------------------------------------------------
    print("\nStep 3: Chunking...")

    raw_chunks = []     # (chunk_text, doc_id)
    digest_chunks = []  # (chunk_text, doc_id)

    for doc_id, (raw, digested) in enumerate(zip(all_doc_texts, digested_doc_texts)):
        for c in chunk_text(raw):
            raw_chunks.append((c, doc_id))
        for c in chunk_text(digested):
            digest_chunks.append((c, doc_id))

    raw_chunk_reduction = round((1 - len(digest_chunks) / len(raw_chunks)) * 100, 1) if raw_chunks else 0
    print(f"  Raw chunks: {len(raw_chunks):,}")
    print(f"  Digest chunks: {len(digest_chunks):,}")
    print(f"  Chunk reduction: {raw_chunk_reduction}%")

    # ---------------------------------------------------------------
    # Step 4: Embed
    # ---------------------------------------------------------------
    print("\nStep 4: Embedding...")

    raw_texts_flat = [c[0] for c in raw_chunks]
    digest_texts_flat = [c[0] for c in digest_chunks]

    print(f"  Embedding {len(raw_texts_flat)} raw chunks...")
    t0 = time.time()
    raw_vectors = embed_all(raw_texts_flat, args.ollama_host)
    raw_embed_time = time.time() - t0
    print(f"  Raw: {raw_embed_time:.1f}s")

    print(f"  Embedding {len(digest_texts_flat)} digest chunks...")
    t0 = time.time()
    digest_vectors = embed_all(digest_texts_flat, args.ollama_host)
    digest_embed_time = time.time() - t0
    print(f"  Digest: {digest_embed_time:.1f}s")

    speedup = raw_embed_time / digest_embed_time if digest_embed_time > 0 else 0
    print(f"  Speedup: {speedup:.1f}x")

    # Normalize
    raw_norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
    raw_norms[raw_norms == 0] = 1
    raw_vectors_norm = raw_vectors / raw_norms

    digest_norms = np.linalg.norm(digest_vectors, axis=1, keepdims=True)
    digest_norms[digest_norms == 0] = 1
    digest_vectors_norm = digest_vectors / digest_norms

    # Embed queries
    print(f"  Embedding {len(queries)} queries...")
    query_vectors = embed_all(queries, args.ollama_host)
    q_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1
    query_vectors_norm = query_vectors / q_norms

    # ---------------------------------------------------------------
    # Step 5: Retrieval evaluation
    # ---------------------------------------------------------------
    K = 5
    print(f"\nStep 5: Retrieval evaluation (top-{K})...")

    raw_chunk_doc_ids = [c[1] for c in raw_chunks]
    digest_chunk_doc_ids = [c[1] for c in digest_chunks]

    raw_ndcg_scores = []
    raw_recall_scores = []
    digest_ndcg_scores = []
    digest_recall_scores = []

    for qi, (qvec, relevant_ids) in enumerate(zip(query_vectors_norm, query_relevant_ids)):
        # Raw retrieval
        raw_sims = qvec @ raw_vectors_norm.T
        raw_top_k_idx = np.argsort(raw_sims)[-K * 3:][::-1]  # get more to dedupe by doc
        # Dedupe: one result per document
        seen_docs = set()
        raw_retrieved_docs = []
        for idx in raw_top_k_idx:
            doc_id = raw_chunk_doc_ids[idx]
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                raw_retrieved_docs.append(doc_id)
            if len(raw_retrieved_docs) == K:
                break

        raw_relevant = [did in relevant_ids for did in raw_retrieved_docs]
        raw_ndcg_scores.append(ndcg_at_k(raw_relevant, K))
        raw_recall_scores.append(recall_at_k(raw_retrieved_docs, relevant_ids, K))

        # Digest retrieval
        dig_sims = qvec @ digest_vectors_norm.T
        dig_top_k_idx = np.argsort(dig_sims)[-K * 3:][::-1]
        seen_docs = set()
        dig_retrieved_docs = []
        for idx in dig_top_k_idx:
            doc_id = digest_chunk_doc_ids[idx]
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                dig_retrieved_docs.append(doc_id)
            if len(dig_retrieved_docs) == K:
                break

        dig_relevant = [did in relevant_ids for did in dig_retrieved_docs]
        digest_ndcg_scores.append(ndcg_at_k(dig_relevant, K))
        digest_recall_scores.append(recall_at_k(dig_retrieved_docs, relevant_ids, K))

    raw_ndcg = round(np.mean(raw_ndcg_scores), 4)
    digest_ndcg = round(np.mean(digest_ndcg_scores), 4)
    raw_recall = round(np.mean(raw_recall_scores), 4)
    digest_recall = round(np.mean(digest_recall_scores), 4)

    print(f"\n  {'Metric':<20s}  {'Raw':>10s}  {'Digest':>10s}  {'Delta':>10s}")
    print(f"  {'-'*55}")
    print(f"  {'nDCG@5':<20s}  {raw_ndcg:>10.4f}  {digest_ndcg:>10.4f}  {digest_ndcg - raw_ndcg:>+10.4f}")
    print(f"  {'Recall@5':<20s}  {raw_recall:>10.4f}  {digest_recall:>10.4f}  {digest_recall - raw_recall:>+10.4f}")

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------
    output = {
        "dataset": args.dataset,
        "split": args.split,
        "corpus": {
            "n_examples": len(ds),
            "n_queries": len(queries),
            "n_unique_docs": len(all_doc_texts),
            "total_raw_chars": total_raw_chars,
            "total_digest_chars": total_digest_chars,
            "char_reduction_pct": char_reduction,
        },
        "predigestion": {
            "boilerplate_lines_detected": len(boilerplate),
            "boilerplate_threshold": args.boilerplate_threshold,
            "duplicates_removed": n_dupes_removed,
        },
        "chunking": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "raw_chunks": len(raw_chunks),
            "digest_chunks": len(digest_chunks),
            "chunk_reduction_pct": raw_chunk_reduction,
        },
        "embedding": {
            "model": EMBED_MODEL,
            "raw_time_s": round(raw_embed_time, 1),
            "digest_time_s": round(digest_embed_time, 1),
            "speedup": f"{speedup:.1f}x",
        },
        "retrieval": {
            "k": K,
            "raw_ndcg": raw_ndcg,
            "digest_ndcg": digest_ndcg,
            "ndcg_delta": round(digest_ndcg - raw_ndcg, 4),
            "raw_recall": raw_recall,
            "digest_recall": digest_recall,
            "recall_delta": round(digest_recall - raw_recall, 4),
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {args.dataset}")
    print(f"{'=' * 60}")
    print(f"Char reduction:   {char_reduction}%")
    print(f"Chunk reduction:  {raw_chunk_reduction}% ({len(raw_chunks)} → {len(digest_chunks)})")
    print(f"Embed speedup:    {speedup:.1f}x")
    print(f"nDCG@5:           raw={raw_ndcg:.4f}  digest={digest_ndcg:.4f}  ({digest_ndcg - raw_ndcg:+.4f})")
    print(f"Recall@5:         raw={raw_recall:.4f}  digest={digest_recall:.4f}  ({digest_recall - raw_recall:+.4f})")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
