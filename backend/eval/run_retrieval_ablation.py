"""
P1.2 - retrieval-only ablation scored with proper IR metrics against gold
chunk labels (backend/eval/gold.jsonl), replacing the P0-era chrF/embedding-
similarity proxy (which conflated retrieval quality with generation-style
text metrics and had no gold labels to check against).

For each config (dense / sparse / hybrid, each with/without the cross-encoder
reranker), computes per answerable question:
  - recall@k for k = 1, 3, 5, 10  (did any gold chunk appear in the top-k?)
  - reciprocal rank (for MRR)
  - nDCG@5 (binary relevance)
  - latency

The one unanswerable question (price-1, gold_chunk_ids=[]) is excluded from
these retrieval-quality metrics (recall/MRR/nDCG are undefined with no gold)
and reported separately - it's a refusal-behaviour question, not a retrieval
one (see PROJECT_PLAN.md P3).

Reports mean, std, and a 95% CI half-width (normal approximation) alongside
n for every metric - a bare mean over 15 questions is not a result on its own.

Requires exclusive access to the local Qdrant store - stop main.py first.
"""
import os
import sys
import json
import math
import time
import statistics
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from qdrant_client import models

from vector_db import db_client, COLLECTION_NAME, embed_model, reranker_model

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")
GOLD_PATH = os.path.join(os.path.dirname(__file__), "gold.jsonl")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "retrieval_results.jsonl")

CONFIGS = [
    ("dense", False), ("dense", True),
    ("sparse", False), ("sparse", True),
    ("hybrid", False), ("hybrid", True),
]
RECALL_KS = [1, 3, 5, 10]


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


def retrieve_ordered_ids(mode, use_reranker, d_vec, s_idx, s_val, query_text):
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


def main():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = {q["id"]: q for q in json.load(f)}
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        gold = {rec["id"]: rec for rec in (json.loads(line) for line in f)}

    answerable_ids = [qid for qid, g in gold.items() if not g["unanswerable"]]
    unanswerable_ids = [qid for qid, g in gold.items() if g["unanswerable"]]
    print(f"{len(answerable_ids)} answerable questions, {len(unanswerable_ids)} unanswerable (excluded from IR metrics): {unanswerable_ids}")

    raw_results = []  # per (config, question) records for JSONL dump
    per_config_metrics = defaultdict(lambda: defaultdict(list))  # config -> metric -> [values]
    per_config_latency = defaultdict(list)

    for qid in answerable_ids:
        q = questions[qid]
        gold_ids = gold[qid]["gold_chunk_ids"]
        d_vec, s_idx, s_val = get_vectors(q["question"])

        for mode, use_reranker in CONFIGS:
            config_name = f"{mode}{'+rerank' if use_reranker else ''}"
            t0 = time.time()
            retrieved_ids = retrieve_ordered_ids(mode, use_reranker, d_vec, s_idx, s_val, q["question"])
            latency = time.time() - t0

            recalls = {k: recall_at_k(retrieved_ids, gold_ids, k) for k in RECALL_KS}
            rr = reciprocal_rank(retrieved_ids, gold_ids)
            ndcg5 = ndcg_at_5(retrieved_ids, gold_ids)

            for k in RECALL_KS:
                per_config_metrics[config_name][f"recall@{k}"].append(recalls[k])
            per_config_metrics[config_name]["mrr"].append(rr)
            per_config_metrics[config_name]["ndcg@5"].append(ndcg5)
            per_config_latency[config_name].append(latency)

            record = {
                "config": config_name, "id": qid, "category": q["category"],
                "retrieved_chunk_ids": retrieved_ids, "gold_chunk_ids": gold_ids,
                **{f"recall@{k}": recalls[k] for k in RECALL_KS},
                "reciprocal_rank": rr, "ndcg@5": ndcg5, "latency_seconds": latency,
            }
            raw_results.append(record)
            print(
                f"[{config_name}] {qid}: recall@5={recalls[5]:.0f} recall@10={recalls[10]:.0f} "
                f"RR={rr:.3f} nDCG@5={ndcg5:.3f} latency={latency:.2f}s",
                flush=True,
            )

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for r in raw_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== Retrieval Ablation Summary (IR metrics, gold-labeled) ===")
    metric_names = [f"recall@{k}" for k in RECALL_KS] + ["mrr", "ndcg@5"]
    for mode, use_reranker in CONFIGS:
        config_name = f"{mode}{'+rerank' if use_reranker else ''}"
        print(f"\n{config_name}:")
        for m in metric_names:
            mean, std, ci95, n = mean_std_ci(per_config_metrics[config_name][m])
            print(f"  {m:<12s} mean={mean:.3f}  std={std:.3f}  95% CI=+/-{ci95:.3f}  (n={n})")
        lats = per_config_latency[config_name]
        lat_mean = statistics.mean(lats)
        lat_p95 = percentile(lats, 0.95)
        print(f"  {'latency':<12s} mean={lat_mean:.3f}s  p95={lat_p95:.3f}s  (n={len(lats)})")

    print(f"\nExcluded from IR metrics (unanswerable, gold=[]): {unanswerable_ids}")
    print("These test refusal behaviour, not retrieval quality - see PROJECT_PLAN.md P3.")


if __name__ == "__main__":
    main()
