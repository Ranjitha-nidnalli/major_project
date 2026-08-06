"""
P2.1 - freezes retrieval as a controlled variable for the generation-model
comparison. Runs retrieval once per eval question, in the current shipping
config (hybrid RRF, top-5, reranker gated by ENABLE_RERANKER same as
rag_service.py - default off per the P1.3 decision), and caches the result to
contexts.json: {question_id: {query, retrieved_chunk_ids, context_text}}.

run_eval.py's P2 generation comparison reads this instead of calling
retrieval itself, so all 3 models see byte-identical context - differences
in the results are then attributable purely to generation, not retrieval
variance or router/query-expansion calls (which themselves depend on
whichever GENERATION_MODEL happens to be configured, which would recontaminate
the "frozen" context if included here).

Requires exclusive access to the local Qdrant store - stop main.py first.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from qdrant_client import models

from vector_db import db_client, COLLECTION_NAME, embed_model, reranker_model

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")
CONTEXTS_PATH = os.path.join(os.path.dirname(__file__), "contexts.json")

ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() == "true"
RERANK_THRESHOLD = 0.5  # must match rag_service.py


def get_vectors(q_text):
    out = embed_model.encode([q_text], return_dense=True, return_sparse=True)
    dense_vec = out["dense_vecs"][0].tolist()
    lex_weights = out["lexical_weights"][0]
    sp_indices = [int(k) for k in lex_weights.keys()]
    sp_values = [float(v) for v in lex_weights.values()]
    return dense_vec, sp_indices, sp_values


def retrieve_hybrid(query_text, d_vec, s_idx, s_val):
    response = db_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=d_vec, using="dense", limit=15),
            models.Prefetch(query=models.SparseVector(indices=s_idx, values=s_val), using="sparse", limit=15),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=5,
    )
    hits = response.points

    if ENABLE_RERANKER and hits:
        pairs = [[query_text, h.payload["text"]] for h in hits]
        scores = reranker_model.predict(pairs, activation_fn=torch.nn.Sigmoid())
        ranked = sorted(zip(hits, scores), key=lambda p: p[1], reverse=True)
        hits = [h for h, s in ranked if s >= RERANK_THRESHOLD]

    return hits


def main():
    print(f"ENABLE_RERANKER={ENABLE_RERANKER} (must match the config being evaluated)")
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    contexts = {}
    for q in questions:
        d_vec, s_idx, s_val = get_vectors(q["question"])
        hits = retrieve_hybrid(q["question"], d_vec, s_idx, s_val)
        docs = [h.payload["text"] for h in hits]
        context_text = "\n\n".join(f"<doc>{doc}</doc>" for doc in docs)

        contexts[q["id"]] = {
            "query": q["question"],
            "retrieved_chunk_ids": [str(h.id) for h in hits],
            "context_text": context_text,
        }
        print(f"{q['id']}: {len(hits)} chunks frozen")

    with open(CONTEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump(contexts, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(contexts)} frozen contexts to {CONTEXTS_PATH}")


if __name__ == "__main__":
    main()
