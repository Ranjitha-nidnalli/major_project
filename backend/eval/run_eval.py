"""
P2.2/P2.3 - generation-model comparison on frozen retrieval contexts.

Retrieval is a controlled variable here (see build_contexts.py): every model
sees byte-identical context per question, so any difference in scores is
attributable to generation, not retrieval variance.

Note: For Groq free tier, self-judging (same model/backend as generation) is
allowed as a documented, pragmatic default. For a genuine cross-provider
judge (mitigating self-preference bias, P2.3), set JUDGE_BACKEND=openrouter
alongside a JUDGE_MODEL that's a valid OpenRouter model ID, with
OPENROUTER_API_KEY set in .env -- this became possible 2026-09-23 (TODO #45)
via call_llm's per-call backend override; previously LLM_BACKEND was
process-wide, so Groq-generates/OpenRouter-judges wasn't achievable in one run.

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
import hashlib
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sacrebleu

from rag_service import generate_from_context, calculate_faithfulness, SYSTEM_INSTRUCTION
from vector_db import embed_model
from chat_db import connect_db, close_db

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")
CONTEXTS_PATH = os.path.join(os.path.dirname(__file__), "contexts.json")
# Each run writes its own file under runs/, never appends to a shared one
# (CLAUDE.md: "Eval runs write to their own directory, never append to a
# shared file. Record dataset hash, model ID, prompt version."). Appending
# to one results.jsonl across runs -- the old behavior -- silently mixed
# rows from different corpus/prompt states with no way to tell them apart,
# and forced export_for_human_eval.py into a fragile "keep the latest
# duplicate" workaround.
RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")


def _short_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]

EMBEDDING_SIM_BASELINE = 0.524

# Judge model is set explicitly via the JUDGE_MODEL env var, not looked up
# from a hardcoded map of specific GENERATION_MODEL values. The old
# JUDGE_MODEL_MAP raised ValueError for any model not already listed in it,
# which crashed the bake-off the first time someone tried a new model
# (verified defect, Task 6). Self-judging (JUDGE_MODEL == GENERATION_MODEL)
# is allowed -- it's a documented, pragmatic tradeoff for free-tier rate
# limits (see module docstring) -- but it must be an explicit choice made
# by whoever runs the script, not a silent default.

def chrf_score(reference: str, hypothesis: str) -> float:
    return sacrebleu.sentence_chrf(hypothesis, [reference]).score / 100

def embedding_similarity(a: str, b: str) -> float:
    out = embed_model.encode([a, b], return_dense=True)
    v1, v2 = out["dense_vecs"][0], out["dense_vecs"][1]
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

async def main():
    model_tag = os.getenv("GENERATION_MODEL", "llama-3.1-8b-instant")
    judge_model = os.getenv("JUDGE_MODEL")
    if not judge_model:
        raise ValueError(
            "JUDGE_MODEL env var is not set. Set it to the model that should "
            "judge GENERATION_MODEL's answers, e.g.:\n"
            "  set GENERATION_MODEL=llama-3.1-8b-instant\n"
            "  set JUDGE_MODEL=llama-3.1-8b-instant   (self-judge; see module docstring)\n"
            "or a different model for cross-judging."
        )
    # Optional (TODO #45): judge via a different provider than generation,
    # e.g. JUDGE_BACKEND=openrouter while generation stays on Groq. None
    # (unset) uses LLM_BACKEND for both, unchanged from prior behavior.
    judge_backend = os.getenv("JUDGE_BACKEND")

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(CONTEXTS_PATH, "rb") as f:
        contexts_bytes = f.read()
    contexts = json.loads(contexts_bytes)

    dataset_hash = _short_hash(contexts_bytes)
    prompt_version = _short_hash(SYSTEM_INSTRUCTION.encode("utf-8"))
    run_timestamp = time.strftime("%Y%m%dT%H%M%S")

    os.makedirs(RUNS_DIR, exist_ok=True)
    # model_tag frequently contains "/" (provider-prefixed IDs like
    # "openai/gpt-oss-20b" are the norm on Groq/OpenRouter) -- a bare "/"
    # in a filename is a path separator, not a valid filename character.
    # Verified 2026-09-21: this crashed the run at the very last line,
    # after every generation+judge API call had already been paid for.
    safe_model_tag = model_tag.replace("/", "-")
    results_path = os.path.join(
        RUNS_DIR, f"{safe_model_tag}__{dataset_hash}__{run_timestamp}.jsonl"
    )

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
                "judge_backend": judge_backend or os.getenv("LLM_BACKEND", "groq"),
                "dataset_hash": dataset_hash,
                "prompt_version": prompt_version,
                "run_timestamp": run_timestamp,
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
        judge_label = f"{judge_model} via {judge_backend}" if judge_backend else judge_model
        print(f"\n--- Phase 2: judging all {len(results)} answers with {judge_label} ---", flush=True)
        for r in results:
            t0 = time.time()
            r["accuracy_score"] = await calculate_faithfulness(
                r["context_text"], r["generated_answer"],
                judge_model=judge_model, judge_backend=judge_backend,
            )
            r["judge_latency_seconds"] = time.time() - t0
            print(f"[{judge_label} judging {r['id']}]: faithfulness={r['accuracy_score']:.3f}", flush=True)
    finally:
        close_db()

    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(results)} records to {results_path}")

    avg_chrf = sum(r["chrf"] for r in results) / len(results)
    avg_emb = sum(r["embedding_similarity"] for r in results) / len(results)
    avg_faith = sum(r["accuracy_score"] for r in results) / len(results)
    avg_gen_lat = sum(r["generation_latency_seconds"] for r in results) / len(results)
    avg_judge_lat = sum(r["judge_latency_seconds"] for r in results) / len(results)
    print(f"\n=== {model_tag} summary (judged by {judge_label}) ===")
    print(
        f"avg chrF: {avg_chrf:.3f} avg emb-sim: {avg_emb:.3f} "
        f"(+{avg_emb - EMBEDDING_SIM_BASELINE:.3f} vs baseline) "
        f"avg faithfulness: {avg_faith:.3f} "
        f"avg gen latency: {avg_gen_lat:.2f}s avg judge latency: {avg_judge_lat:.2f}s"
    )

if __name__ == "__main__":
    asyncio.run(main())
