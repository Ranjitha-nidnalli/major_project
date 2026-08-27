
"""
reranker_diagnostic.py

Per-query comparison of dense vs dense+rerank rankings.
For each query, prints side-by-side top-10 rankings showing:
- Which chunks moved up/down
- Where gold chunks ended up
- Whether reranking helped or hurt

Usage:
    cd backend
    python eval/reranker_diagnostic.py

Output: Detailed per-query comparison table
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from qdrant_client import models
from vector_db import db_client, COLLECTION_NAME, embed_model, reranker_model
from indic_preprocess import normalize_kannada

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")
GOLD_PATH = os.path.join(os.path.dirname(__file__), "gold.jsonl")


def get_vectors(q_text):
    q_text = normalize_kannada(q_text)
    out = embed_model.encode([q_text], return_dense=True, return_sparse=True)
    dense_vec = out["dense_vecs"][0].tolist()
    lex_weights = out["lexical_weights"][0]
    sp_indices = [int(k) for k in lex_weights.keys()]
    sp_values = [float(v) for v in lex_weights.values()]
    return dense_vec, sp_indices, sp_values


def retrieve_dense(q_text, limit=15):
    d_vec, s_idx, s_val = get_vectors(q_text)
    resp = db_client.query_points(
        collection_name=COLLECTION_NAME, query=d_vec, using="dense", limit=limit
    )
    return [(str(h.id), h.payload.get("text", "")[:80], float(h.score)) for h in resp.points]


def retrieve_dense_reranked(q_text, limit=15):
    d_vec, s_idx, s_val = get_vectors(q_text)
    resp = db_client.query_points(
        collection_name=COLLECTION_NAME, query=d_vec, using="dense", limit=limit
    )
    hits = resp.points
    if not hits:
        return []

    pairs = [[q_text, h.payload["text"]] for h in hits]
    scores = reranker_model.predict(pairs, activation_fn=torch.nn.Sigmoid())
    reranked = sorted(zip(hits, scores), key=lambda p: p[1], reverse=True)

    return [(str(h.id), h.payload.get("text", "")[:80], float(s)) for h, s in reranked]


def print_comparison(qid, question, gold_ids, dense_results, reranked_results):
    print(f"\n{'='*70}")
    print(f"QUERY: {qid}")
    print(f"TEXT:  {question}")
    print(f"GOLD:  {gold_ids}")
    print(f"{'='*70}")

    # Build lookup for reranked positions
    rerank_pos = {cid: i+1 for i, (cid, _, _) in enumerate(reranked_results)}

    print(f"\n{'Rank':<6} {'Dense ID':<12} {'Rerank Pos':<12} {'Text (truncated)':<50}")
    print("-" * 70)

    for i, (cid, text, score) in enumerate(dense_results[:10]):
        rank_before = i + 1
        rank_after = rerank_pos.get(cid, "—")
        marker = "🟢 GOLD" if cid in gold_ids else ""
        move = ""
        if str(rank_after) != "—":
            delta = int(rank_after) - rank_before
            if delta < 0:
                move = f"↑{abs(delta)}"
            elif delta > 0:
                move = f"↓{delta}"
            else:
                move = "="

        print(f"{rank_before:<6} {cid[:8]:<12} {str(rank_after):<12} {text[:40]:<50} {marker} {move}")

    # Check gold chunk positions
    print(f"\nGOLD CHUNK POSITIONS:")
    for gid in gold_ids:
        dense_pos = next((i+1 for i, (cid, _, _) in enumerate(dense_results) if cid == gid), "NOT FOUND")
        rerank_pos_val = rerank_pos.get(gid, "NOT FOUND")

        if dense_pos != "NOT FOUND" and rerank_pos_val != "NOT FOUND":
            delta = rerank_pos_val - dense_pos
            if delta < 0:
                verdict = f"IMPROVED by {abs(delta)} positions"
            elif delta > 0:
                verdict = f"WORSENED by {delta} positions ⚠️"
            else:
                verdict = "UNCHANGED"
        elif dense_pos != "NOT FOUND" and rerank_pos_val == "NOT FOUND":
            verdict = "DROPPED OUT of top-15! 🔴"
        else:
            verdict = "Not in dense top-15 either"

        print(f"  {gid[:8]}...  dense=#{dense_pos}  rerank=#{rerank_pos_val}  → {verdict}")


def main():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = {q["id"]: q for q in json.load(f)}
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        gold = {rec["id"]: rec for rec in (json.loads(line) for line in f)}

    answerable_ids = [qid for qid, g in gold.items() if not g["unanswerable"]]

    print("=" * 70)
    print("RERANKER DIAGNOSTIC: Per-Query Dense vs Dense+Rerank Comparison")
    print("=" * 70)
    print(f"\nAnalyzing {len(answerable_ids)} answerable questions...")
    print("\n🟢 = gold chunk")
    print("↑N = moved up N positions after reranking")
    print("↓N = moved down N positions after reranking")
    print("= = stayed same position")

    for qid in answerable_ids:
        q = questions[qid]
        gold_ids = gold[qid]["gold_chunk_ids"]

        dense_results = retrieve_dense(q["question"])
        reranked_results = retrieve_dense_reranked(q["question"])

        print_comparison(qid, q["question"], gold_ids, dense_results, reranked_results)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("\nFor each query above, check:")
    print("  1. Did gold chunks move UP (good) or DOWN (bad)?")
    print("  2. Did any gold chunk DROP OUT of top-15? (very bad)")
    print("  3. Did irrelevant chunks get promoted above gold? (explains MRR drop)")
    print("\nCommon reranker failure modes:")
    print("  - Reranker overweights query terms that appear in wrong chunks")
    print("  - Reranker scores long chunks higher (more term matches)")
    print("  - Reranker confuses similar but wrong entities (pest A vs pest B)")


if __name__ == "__main__":
    main()
