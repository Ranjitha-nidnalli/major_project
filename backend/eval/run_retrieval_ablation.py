"""
P1.2 - retrieval-only ablation scored with proper IR metrics against gold
chunk labels (backend/eval/gold.jsonl), replacing the P0-era chrF/embedding-
similarity proxy.

**Updated:** Adds BM25 as a standalone and fused config, plus bucketed
analysis by query type (exact-term vs semantic) per architect review.

For each config (dense / sparse / hybrid / bm25 / bm25+dense / bm25+hybrid,
each with/without reranker), computes per answerable question:
  - recall@k for k = 1, 3, 5, 10
  - reciprocal rank (for MRR)
  - nDCG@5 (binary relevance)
  - latency

Bucketed analysis splits questions into:
  - exact-term: queries with chemical names, dosages, or specific numbers
  - semantic: general descriptive queries

Reports mean, std, and 95% CI half-width alongside n for every metric.

Requires exclusive access to the local Qdrant store - stop main.py first.
"""
import os
import sys
import json
import math
import time
import statistics
import re
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from qdrant_client import models

from vector_db import db_client, COLLECTION_NAME, embed_model, reranker_model
from bm25_retriever import BM25Retriever, load_chunks_from_qdrant_upsert

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")
GOLD_PATH = os.path.join(os.path.dirname(__file__), "gold.jsonl")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "retrieval_results.jsonl")

# Original dense/sparse/hybrid configs
BASE_CONFIGS = [
    ("dense", False), ("dense", True),
    ("sparse", False), ("sparse", True),
    ("hybrid", False), ("hybrid", True),
]

# BM25 configs: standalone and fused with dense/hybrid
BM25_CONFIGS = [
    ("bm25", False),
    ("bm25+dense", False),
    ("bm25+hybrid", False),
]

ALL_CONFIGS = BASE_CONFIGS + BM25_CONFIGS
RECALL_KS = [1, 3, 5, 10]

# Known chemical/pesticide names for exact-term classification
_CHEMICAL_NAMES = {
    "ಕಾರ್ಬೆಂಡೈಜಿಮ್", "ಕ್ಲೋರಪೈರಿಫಾಸ್", "ಡೈಮಿಥೋಯೇಟ್", "ಫೋರೇಟ್", "ಕಾರ್ಬೋಫ್ಯೂರಾನ್",
    "ಕ್ಲೋರಾಂಟ್ರಾನಿಲಿಪ್ರೋಲ್", "ಫಿಪ್ರೋನಿಲ್", "ಅಜಟೊಬ್ಯಾಕ್ಟರ್",
    "carbendazim", "chlorpyrifos", "dimethoate", "phorate", "carbofuran",
    "chlorantraniliprole", "fipronil", "azotobacter",
}

# Number+unit pattern for dosage queries
_NUM_UNIT_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?\s*(?:ಗ್ರಾಂ|gram|ಕೆಜಿ|kg|ಲೀಟರ್|litre|ಮಿಲಿ|ml|ಎಕರೆ|acre|%)\b")


def classify_query_type(question: str) -> str:
    """
    Classify a query as 'exact-term' or 'semantic'.
    exact-term: contains a known chemical name or a dosage/quantity pattern.
    semantic: everything else.
    """
    q_lower = question.lower()
    # Check for chemical names
    for chem in _CHEMICAL_NAMES:
        if chem.lower() in q_lower:
            return "exact-term"
    # Check for dosage/quantity patterns
    if _NUM_UNIT_RE.search(question):
        return "exact-term"
    return "semantic"


def get_vectors(q_text):
    out = embed_model.encode([q_text], return_dense=True, return_sparse=True)
    dense_vec = out["dense_vecs"][0].tolist()
    lex_weights = out["lexical_weights"][0]
    sp_indices = [int(k) for k in lex_weights.keys()]
    sp_values = [float(v) for v in lex_weights.values()]
    return dense_vec, sp_indices, sp_values


def retrieve(mode, d_vec, s_idx, s_val, limit=15):
    if mode == "dense":
        resp = db_client.query_points(
            collection_name=COLLECTION_NAME, query=d_vec, using="dense", limit=limit
        )
        return resp.points
    if mode == "sparse":
        resp = db_client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.SparseVector(indices=s_idx, values=s_val),
            using="sparse", limit=limit,
        )
        return resp.points
    if mode == "hybrid":
        resp = db_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=d_vec, using="dense", limit=15),
                models.Prefetch(query=models.SparseVector(indices=s_idx, values=s_val), using="sparse", limit=15),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )
        return resp.points
    raise ValueError(mode)


def retrieve_ordered_ids(mode, use_reranker, d_vec, s_idx, s_val, query_text, bm25=None):
    """
    Retrieve ordered chunk IDs for a given config.
    For bm25 modes, uses the BM25 retriever.
    """
    # BM25 standalone
    if mode == "bm25":
        if bm25 is None:
            raise ValueError("BM25 retriever required for bm25 mode")
        results = bm25.search(query_text, top_k=15)
        return [cid for cid, _ in results]

    # BM25 + dense fusion
    if mode == "bm25+dense":
        if bm25 is None:
            raise ValueError("BM25 retriever required for bm25+dense mode")
        dense_hits = retrieve("dense", d_vec, s_idx, s_val, limit=15)
        dense_results = [(str(h.id), float(h.score)) for h in dense_hits]
        bm25_results = bm25.search(query_text, top_k=15)
        fused = bm25.rrf_fuse(bm25_results, dense_results, other_weight=1.0, top_k=15)
        return [cid for cid, _ in fused]

    # BM25 + hybrid fusion
    if mode == "bm25+hybrid":
        if bm25 is None:
            raise ValueError("BM25 retriever required for bm25+hybrid mode")
        hybrid_hits = retrieve("hybrid", d_vec, s_idx, s_val, limit=15)
        hybrid_results = [(str(h.id), float(h.score)) for h in hybrid_hits]
        bm25_results = bm25.search(query_text, top_k=15)
        fused = bm25.rrf_fuse(bm25_results, hybrid_results, other_weight=1.0, top_k=15)
        return [cid for cid, _ in fused]

    # Original modes
    hits = retrieve(mode, d_vec, s_idx, s_val, limit=15)
    if use_reranker and hits:
        pairs = [[query_text, h.payload["text"]] for h in hits]
        scores = reranker_model.predict(pairs, activation_fn=torch.nn.Sigmoid())
        hits = [h for h, _ in sorted(zip(hits, scores), key=lambda p: p[1], reverse=True)]
    return [str(h.id) for h in hits]


def recall_at_k(retrieved_ids, gold_ids, k):
    return 1.0 if set(retrieved_ids[:k]) & set(gold_ids) else 0.0


def reciprocal_rank(retrieved_ids, gold_ids):
    gold_set = set(gold_ids)
    for i, cid in enumerate(retrieved_ids):
        if cid in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_5(retrieved_ids, gold_ids):
    gold_set = set(gold_ids)
    dcg = sum(
        (1.0 if cid in gold_set else 0.0) / math.log2(i + 2)
        for i, cid in enumerate(retrieved_ids[:5])
    )
    ideal_hits = min(len(gold_set), 5)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mean_std_ci(values):
    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return mean, std, ci95, n


def percentile(values, p):
    s = sorted(values)
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


def print_config_summary(config_name, metrics_dict, latencies, n):
    metric_names = [f"recall@{k}" for k in RECALL_KS] + ["mrr", "ndcg@5"]
    print(f"\n{config_name}:")
    for m in metric_names:
        mean, std, ci95, _ = mean_std_ci(metrics_dict[m])
        print(f"  {m:<12s} mean={mean:.3f}  std={std:.3f}  95% CI=+/-{ci95:.3f}  (n={n})")
    lat_mean = statistics.mean(latencies)
    lat_p95 = percentile(latencies, 0.95)
    print(f"  {'latency':<12s} mean={lat_mean:.3f}s  p95={lat_p95:.3f}s  (n={len(latencies)})")


def main():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = {q["id"]: q for q in json.load(f)}
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        gold = {rec["id"]: rec for rec in (json.loads(line) for line in f)}

    answerable_ids = [qid for qid, g in gold.items() if not g["unanswerable"]]
    unanswerable_ids = [qid for qid, g in gold.items() if g["unanswerable"]]
    print(f"{len(answerable_ids)} answerable questions, {len(unanswerable_ids)} unanswerable (excluded from IR metrics): {unanswerable_ids}")

    # Build BM25 index once
    print("\n🔧 Building BM25 index...")
    bm25 = BM25Retriever(load_chunks_from_qdrant_upsert())
    print(f"   BM25 index ready: {len(bm25.chunks)} chunks")

    raw_results = []
    per_config_metrics = defaultdict(lambda: defaultdict(list))
    per_config_latency = defaultdict(list)

    # Bucketed metrics
    per_config_per_bucket_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    per_config_per_bucket_latency = defaultdict(lambda: defaultdict(list))

    for qid in answerable_ids:
        q = questions[qid]
        gold_ids = gold[qid]["gold_chunk_ids"]
        d_vec, s_idx, s_val = get_vectors(q["question"])
        query_type = classify_query_type(q["question"])

        for mode, use_reranker in ALL_CONFIGS:
            config_name = f"{mode}{'+rerank' if use_reranker else ''}"
            t0 = time.time()
            retrieved_ids = retrieve_ordered_ids(mode, use_reranker, d_vec, s_idx, s_val, q["question"], bm25=bm25)
            latency = time.time() - t0

            recalls = {k: recall_at_k(retrieved_ids, gold_ids, k) for k in RECALL_KS}
            rr = reciprocal_rank(retrieved_ids, gold_ids)
            ndcg5 = ndcg_at_5(retrieved_ids, gold_ids)

            for k in RECALL_KS:
                per_config_metrics[config_name][f"recall@{k}"].append(recalls[k])
            per_config_metrics[config_name]["mrr"].append(rr)
            per_config_metrics[config_name]["ndcg@5"].append(ndcg5)
            per_config_latency[config_name].append(latency)

            # Bucketed
            for k in RECALL_KS:
                per_config_per_bucket_metrics[config_name][query_type][f"recall@{k}"].append(recalls[k])
            per_config_per_bucket_metrics[config_name][query_type]["mrr"].append(rr)
            per_config_per_bucket_metrics[config_name][query_type]["ndcg@5"].append(ndcg5)
            per_config_per_bucket_latency[config_name][query_type].append(latency)

            record = {
                "config": config_name, "id": qid, "category": q["category"],
                "query_type": query_type,
                "retrieved_chunk_ids": retrieved_ids, "gold_chunk_ids": gold_ids,
                **{f"recall@{k}": recalls[k] for k in RECALL_KS},
                "reciprocal_rank": rr, "ndcg@5": ndcg5, "latency_seconds": latency,
            }
            raw_results.append(record)
            print(
                f"[{config_name}] {qid} ({query_type}): recall@5={recalls[5]:.0f} recall@10={recalls[10]:.0f} "
                f"RR={rr:.3f} nDCG@5={ndcg5:.3f} latency={latency:.2f}s",
                flush=True,
            )

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for r in raw_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print("=== Retrieval Ablation Summary (IR metrics, gold-labeled) ===")
    print("=" * 70)
    for mode, use_reranker in ALL_CONFIGS:
        config_name = f"{mode}{'+rerank' if use_reranker else ''}"
        n = len(per_config_metrics[config_name]["mrr"])
        print_config_summary(config_name, per_config_metrics[config_name], per_config_latency[config_name], n)

    print("\n" + "=" * 70)
    print("=== Bucketed Analysis: Exact-Term vs Semantic Queries ===")
    print("=" * 70)
    for mode, use_reranker in ALL_CONFIGS:
        config_name = f"{mode}{'+rerank' if use_reranker else ''}"
        for bucket in ["exact-term", "semantic"]:
            bucket_metrics = per_config_per_bucket_metrics[config_name][bucket]
            bucket_latencies = per_config_per_bucket_latency[config_name][bucket]
            if not bucket_metrics.get("mrr"):
                continue
            n = len(bucket_metrics["mrr"])
            print(f"\n{config_name} | {bucket} (n={n}):")
            metric_names = [f"recall@{k}" for k in RECALL_KS] + ["mrr", "ndcg@5"]
            for m in metric_names:
                mean, std, ci95, _ = mean_std_ci(bucket_metrics[m])
                print(f"  {m:<12s} mean={mean:.3f}  std={std:.3f}  95% CI=+/-{ci95:.3f}")
            lat_mean = statistics.mean(bucket_latencies)
            lat_p95 = percentile(bucket_latencies, 0.95)
            print(f"  {'latency':<12s} mean={lat_mean:.3f}s  p95={lat_p95:.3f}s")

    print(f"\nExcluded from IR metrics (unanswerable, gold=[]): {unanswerable_ids}")
    print("These test refusal behaviour, not retrieval quality - see PROJECT_PLAN.md P3.")


if __name__ == "__main__":
    main()
