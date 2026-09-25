"""
End-to-end eval through the LIVE pipeline (rag_service.get_sugarcane_answer).

Unlike run_eval.py -- which replays frozen contexts.json straight into
generate_from_context() and therefore never exercises the router, BM25+dense
fusion (TODO #10), the abstention gate / entity-match hook (TODO #43), or the
faithfulness + numeric refusal gate -- this script sends every question in
questions.json through the same function the API serves. Use it to check
what a user would actually receive; use run_eval.py for controlled
generation-model comparisons on identical context.

Costs ~3 LLM calls per question (router, generation, faithfulness judge;
fewer when the gate refuses early). Judge is whatever production uses
(self-judge with GENERATION_MODEL unless JUDGE_* is wired into the live path).

Writes one file per run under eval/runs/live/ (never appends to a shared
file, per CLAUDE.md), recording model ID, prompt version, and hashes of the
question set and corpus.

chrF / embedding similarity are recorded for continuity with run_eval.py
only. They are NOT quality or safety evidence (CLAUDE.md): chrF cannot
detect numeric errors.

Usage:
    cd backend && python eval/run_live_eval.py
Requires Qdrant (QDRANT_URL), MongoDB (MONGO_URI), and an LLM API key.
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

import rag_service
from rag_service import get_sugarcane_answer, SYSTEM_INSTRUCTION, HARD_REFUSAL_MESSAGE, NOT_IN_CONTEXT_PHRASE
from vector_db import embed_model
from chat_db import connect_db, close_db

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_PATH = os.path.join(EVAL_DIR, "questions.json")
CORPUS_PATH = os.path.join(os.path.dirname(os.path.dirname(EVAL_DIR)), "sugarcanemerged3.json")
RUNS_DIR = os.path.join(EVAL_DIR, "runs", "live")


def _short_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def chrf_score(reference: str, hypothesis: str) -> float:
    return sacrebleu.sentence_chrf(hypothesis, [reference]).score / 100


def embedding_similarity(a: str, b: str) -> float:
    out = embed_model.encode([a, b], return_dense=True)
    v1, v2 = out["dense_vecs"][0], out["dense_vecs"][1]
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def is_refusal(answer: str) -> bool:
    return HARD_REFUSAL_MESSAGE in answer or NOT_IN_CONTEXT_PHRASE in answer


# Record every gate decision without changing its behavior: wrap the name
# rag_service looks up at call time.
_gate_log = []
_real_decide = rag_service.gating_decide


def _recording_decide(**kwargs):
    decision = _real_decide(**kwargs)
    _gate_log.append({
        "category": kwargs.get("category"),
        "relevance": kwargs.get("relevance"),
        "should_answer": decision.should_answer,
        "reason": decision.reason,
        "threshold_used": decision.threshold_used,
    })
    return decision


rag_service.gating_decide = _recording_decide

# Record the router's raw label and the top chunk's corpus category, since
# gate.category above is the category AFTER resolve_entity_category (#47).
_route_log = []
_real_resolve = rag_service.resolve_entity_category


def _recording_resolve(router_category, top_chunk_category, protected_categories):
    _route_log.append({"router_category": router_category, "top_chunk_category": top_chunk_category})
    return _real_resolve(router_category, top_chunk_category, protected_categories)


rag_service.resolve_entity_category = _recording_resolve

# Record the model's answer and judge score BEFORE the faithfulness/numeric
# gate can replace it with the refusal message, so numeric checks on a live
# run can be replayed later (TODO #49 note).
_gen_log = []
_real_generate = rag_service.generate_from_context


async def _recording_generate(*args, **kwargs):
    result = await _real_generate(*args, **kwargs)
    _gen_log.append({"raw_answer": result.get("answer"), "raw_accuracy_score": result.get("accuracy_score")})
    return result


rag_service.generate_from_context = _recording_generate


async def main():
    with open(QUESTIONS_PATH, "rb") as f:
        questions_bytes = f.read()
    questions = json.loads(questions_bytes)
    with open(CORPUS_PATH, "rb") as f:
        corpus_hash = _short_hash(f.read())

    model = rag_service.GENERATION_MODEL
    meta = {
        "pipeline": "live:get_sugarcane_answer",
        "model": model,
        "llm_backend": os.getenv("LLM_BACKEND", "groq"),
        "prompt_version": _short_hash(SYSTEM_INSTRUCTION.encode("utf-8")),
        "questions_hash": _short_hash(questions_bytes),
        "corpus_hash": corpus_hash,
        "lenient_pest_disease_threshold": rag_service.ENABLE_LENIENT_PEST_DISEASE_THRESHOLD,
        "reranker_enabled": os.getenv("ENABLE_RERANKER", "false"),
        "run_timestamp": time.strftime("%Y%m%dT%H%M%S"),
    }
    print(json.dumps(meta, indent=2))

    os.makedirs(RUNS_DIR, exist_ok=True)
    out_path = os.path.join(
        RUNS_DIR, f"{model.replace('/', '-')}__{meta['questions_hash']}__{meta['run_timestamp']}.jsonl"
    )

    connect_db()
    results = []
    try:
        for q in questions:
            _gate_log.clear()
            _route_log.clear()
            _gen_log.clear()
            t0 = time.time()
            resp = await get_sugarcane_answer(
                q["question"], f"live_eval_{q['id']}_{meta['run_timestamp']}",
                return_context=True, interactive=True,
            )
            latency = time.time() - t0
            answer = resp["answer"]
            gate = _gate_log[-1] if _gate_log else None
            expected_refusal = NOT_IN_CONTEXT_PHRASE in q["expected_answer"]
            refused = is_refusal(answer)

            record = {
                **meta,
                "id": q["id"],
                "gold_category": q["category"],
                "question": q["question"],
                "expected_answer": q["expected_answer"],
                "expected_refusal": expected_refusal,
                "generated_answer": answer,
                "refused": refused,
                "refusal_correct": refused == expected_refusal,
                "route": _route_log[-1] if _route_log else None,
                "gate": gate,
                # None when the gate refused before generation ran.
                "raw_generation": _gen_log[-1] if _gen_log else None,
                "accuracy_score": resp.get("accuracy_score"),
                "numeric_faithfulness": resp.get("numeric_faithfulness"),
                "source_chunks": resp.get("source_chunks"),
                "chrf": chrf_score(q["expected_answer"], answer),
                "embedding_similarity": embedding_similarity(q["expected_answer"], answer),
                "latency_seconds": latency,
            }
            results.append(record)
            g = gate or {}
            print(
                f"[{q['id']}] router={(record['route'] or {}).get('router_category')} "
                f"category={g.get('category')} gate={g.get('reason')} "
                f"refused={refused} expected_refusal={expected_refusal} "
                f"faith={record['accuracy_score']} latency={latency:.1f}s",
                flush=True,
            )
    finally:
        close_db()
        # Write whatever completed, so a mid-run failure doesn't discard paid calls.
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(results)} records to {out_path}")

    n = len(results)
    answerable = [r for r in results if not r["expected_refusal"]]
    unanswerable = [r for r in results if r["expected_refusal"]]
    print(f"\n=== live pipeline summary ({model}, n={n}) ===")
    print(f"refusal correct: {sum(r['refusal_correct'] for r in results)}/{n}")
    print(f"  unanswerable refused: {sum(r['refused'] for r in unanswerable)}/{len(unanswerable)}")
    print(f"  answerable answered:  {sum(not r['refused'] for r in answerable)}/{len(answerable)}")


if __name__ == "__main__":
    asyncio.run(main())
