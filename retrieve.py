"""Retrieval strategies (similarity, bm25, hybrid, rerank) and answer generation.

Every retriever returns a list of (chunk text, page, chunk_index, score), best first.
"""
import re
import time
from functools import lru_cache

import chromadb
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import config


# Open the Chroma collection built by ingest.py.
def get_collection():
    return chromadb.PersistentClient(path=config.CHROMA_DIR).get_collection(config.COLLECTION)


# Every chunk in the index as (text, page, chunk_index, 0.0), in document order.
@lru_cache(maxsize=1)
def all_chunks():
    data = get_collection().get(include=["documents", "metadatas"])
    rows = [(d, m["page"], m["chunk_index"], 0.0) for d, m in zip(data["documents"], data["metadatas"])]
    return sorted(rows, key=lambda r: (r[1], r[2]))


# Build the BM25 index once (call bm25_index.cache_clear() after re-ingesting).
@lru_cache(maxsize=1)
def bm25_index():
    return BM25Okapi([tokenize(c[0]) for c in all_chunks()])


# Lowercase word tokens.
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


# Top-k chunks by cosine similarity of Ollama embeddings (score = 1 - cosine distance).
def similarity(query, k=config.FINAL_K):
    vector = ollama.embed(model=config.EMBED_MODEL, input=[query])["embeddings"][0]
    r = get_collection().query(query_embeddings=[vector], n_results=k,
                               include=["documents", "metadatas", "distances"])
    return [(d, m["page"], m["chunk_index"], 1 - dist)
            for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]


# Top-k chunks by BM25 keyword score.
def bm25(query, k=config.FINAL_K):
    chunks = all_chunks()
    scores = bm25_index().get_scores(tokenize(query))
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])[:k]
    return [chunks[i][:3] + (float(scores[i]),) for i in order]


# Merge similarity and BM25 rankings with reciprocal rank fusion; return the top k.
def hybrid(query, k=config.FINAL_K):
    fused, rows = {}, {}
    for ranking in (similarity(query, config.CANDIDATE_K), bm25(query, config.CANDIDATE_K)):
        for rank, row in enumerate(ranking, 1):
            key = (row[1], row[2])
            rows[key] = row
            fused[key] = fused.get(key, 0.0) + 1.0 / (config.RRF_K + rank)
    best = sorted(fused, key=lambda key: -fused[key])[:k]
    return [rows[key][:3] + (fused[key],) for key in best]


# Load the cross-encoder once.
@lru_cache(maxsize=1)
def cross_encoder():
    return CrossEncoder(config.RERANK_MODEL)


# Score (query, chunk) pairs with the cross-encoder and keep the best k.
def rerank(query, candidates, k=config.FINAL_K):
    scores = cross_encoder().predict([(query, c[0]) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])[:k]
    return [c[:3] + (float(s),) for c, s in ranked]


# Run strategy A/B/C/D from config.STRATEGIES and return the final top-5 chunks.
def retrieve(question, strategy):
    source, use_rerank = config.STRATEGIES[strategy]
    finder = {"similarity": similarity, "bm25": bm25, "hybrid": hybrid}[source]
    if not use_rerank:
        return finder(question, config.FINAL_K)
    return rerank(question, finder(question, config.CANDIDATE_K), config.FINAL_K)


# Ask GEN_MODEL to answer from the chunks; return answer text and token counts.
def generate(question, chunks):
    context = "\n\n".join(f"[page {c[1]}] {c[0]}" for c in chunks)
    reply = ollama.chat(
        model=config.GEN_MODEL, options=config.GEN_OPTIONS,
        messages=[{"role": "system", "content": config.SYSTEM_PROMPT},
                  {"role": "user", "content": config.ANSWER_TEMPLATE.format(context=context, question=question)}])
    return {"answer": reply["message"]["content"].strip(),
            "prompt_tokens": reply.get("prompt_eval_count", 0),
            "completion_tokens": reply.get("eval_count", 0)}


# Retrieve then generate; returns answer, chunks, tokens and timings.
def answer(question, strategy):
    start = time.perf_counter()
    chunks = retrieve(question, strategy)
    retrieval_ms = (time.perf_counter() - start) * 1000
    result = generate(question, chunks)
    result.update(chunks=chunks, retrieval_ms=retrieval_ms, total_s=time.perf_counter() - start)
    return result


# Drop chunks scoring below NOISE_MIN_SCORE (the top chunk is always kept); returns (kept, removed).
def remove_noise(chunks):
    kept = [c for c in chunks if c[3] >= config.NOISE_MIN_SCORE] or chunks[:1]
    return kept, [c for c in chunks if c not in kept]


# Phase 2 pipeline: cosine k=20 -> rerank -> top 5 -> remove noise -> LLM answer.
def pipeline(question, remove=True):
    start = time.perf_counter()
    chunks = retrieve(question, config.PIPELINE_STRATEGY)
    kept, removed = remove_noise(chunks) if remove else (chunks, [])
    retrieval_ms = (time.perf_counter() - start) * 1000
    result = generate(question, kept)
    result.update(kept=kept, removed=removed, retrieval_ms=retrieval_ms, total_s=time.perf_counter() - start)
    return result
