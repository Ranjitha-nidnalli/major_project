"""
P2.2/P2.3 - generation-model comparison on frozen retrieval contexts.

Retrieval is a controlled variable here (see build_contexts.py): every model
sees byte-identical context per question, so any difference in scores is
attributable to generation, not retrieval variance.

Note: For Groq free tier, we use the same model as judge (self-judge) due to
rate-limit constraints. In a full study, the judge should be a different model
to avoid self-preference bias (P2.3).

Per diagnose_metrics.py (P0.1): scores answers with chrF (script-agnostic),
not ROUGE-L.

Usage: run once per model under test -

  set GENERATION_MODEL=llama-3.1-8b-instant && python eval/run_eval.py

Requires backend/eval/contexts.json to already exist (run build_contexts.py
first).
"""
import os
import sys
import json
import time
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sacrebleu

from rag_service import generate_from_context, calculate_faithfulness
from vector_db import embed_model
from chat_db import connect_db, close_db

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")
CONTEXTS_PATH = os.path.join(os.path.dirname(__file__), "contexts.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.jsonl")

EMBEDDING_SIM_BASELINE = 0.524

# Self-judge for Groq models (pragmatic for free tier rate limits)
# For local Ollama, cross-judge is preferred.
JUDGE_MODEL_MAP = {
    "gemma4:e4b": "llama3.1:8b",
    "llama3.1:8b": "gemma4:e4b",
    "gaganyatri/sarvam-2b-v0.5": "llama3.1:8b",
    "llama-3.1-8b-instant": "llama-3.1-8b-instant",  # self-judge (Groq pragmatic)
}

def chrf_score(reference: str, hypothesis: str) -> float:
    return sacrebleu.sentence_chrf(hypothesis, [reference]).score / 100

def embedding_similarity(a: str, b: str) -> float:
    out = embed_model.encode([a, b], return_dense=True)
    v1, v2 = out["dense_vecs"][0], out["dense_vecs"][1]
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

async def main():
    model_tag = os.getenv("GENERATION_MODEL", "llama-3.1-8b-instant")
    judge_model = JUDGE_MODEL_MAP.get(model_tag)
    if judge_model is None:
        raise ValueError(f"No judge model configured for '{model_tag}' - add it to JUDGE_MODEL_MAP")

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(CONTEXTS_PATH, "r", encoding="utf-8") as f:
        contexts = json.load(f)

    connect_db()
    results = []
    try:
        # Phase 1: all generations
        print(f"--- Phase 1: generating all {len(questions)} answers with {model_tag} ---", flush=True)
        for q in questions:
            ctx = contexts[q["id"]]
            t0 = time.time()
            gen = await generate_from_context(
                q["question"], ctx["context_text"], f"eval_{q['id']}_{model_tag}",
                generation_model=model_tag, run_judge=False,
            )
            latency = time.time() - t0
            answer = gen["answer"]
            chrf = chrf_score(q["expected_answer"], answer)
            emb_sim = embedding_similarity(q["expected_answer"], answer)

            record = {
                "model": model_tag,
                "judge_model": judge_model,
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "expected_answer": q["expected_answer"],
                "generated_answer": answer,
                "retrieved_chunk_ids": ctx["retrieved_chunk_ids"],
                "context_text": ctx["context_text"],
                "chrf": chrf,
                "embedding_similarity": emb_sim,
                "embedding_similarity_above_baseline": emb_sim - EMBEDDING_SIM_BASELINE,
                "generation_latency_seconds": latency,
            }
            results.append(record)
            print(
                f"[{model_tag}] {q['id']}: chrF={chrf:.3f} embsim={emb_sim:.3f} "
                f"(+{emb_sim - EMBEDDING_SIM_BASELINE:.3f} vs baseline) latency={latency:.1f}s",
                flush=True,
            )

        # Phase 2: all judging
        print(f"\n--- Phase 2: judging all {len(results)} answers with {judge_model} ---", flush=True)
        for r in results:
            t0 = time.time()
            r["accuracy_score"] = await calculate_faithfulness(
                r["context_text"], r["generated_answer"], judge_model=judge_model
            )
            r["judge_latency_seconds"] = time.time() - t0
            print(f"[{judge_model} judging {r['id']}]: faithfulness={r['accuracy_score']:.3f}", flush=True)
    finally:
        close_db()

    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    avg_chrf = sum(r["chrf"] for r in results) / len(results)
    avg_emb = sum(r["embedding_similarity"] for r in results) / len(results)
    avg_faith = sum(r["accuracy_score"] for r in results) / len(results)
    avg_gen_lat = sum(r["generation_latency_seconds"] for r in results) / len(results)
    avg_judge_lat = sum(r["judge_latency_seconds"] for r in results) / len(results)
    print(f"\n=== {model_tag} summary (judged by {judge_model}) ===")
    print(
        f"avg chrF: {avg_chrf:.3f} avg emb-sim: {avg_emb:.3f} "
        f"(+{avg_emb - EMBEDDING_SIM_BASELINE:.3f} vs baseline) "
        f"avg faithfulness: {avg_faith:.3f} "
        f"avg gen latency: {avg_gen_lat:.2f}s avg judge latency: {avg_judge_lat:.2f}s"
    )

if __name__ == "__main__":
    asyncio.run(main())
