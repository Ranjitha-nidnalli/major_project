"""
No-retrieval baseline (THE PLAN, TODO #17): the same generation model,
asked the same questions with NO retrieved context and NO gate, judge, or
numeric check. It shows what a farmer would get from the bare model, which
is what the RAG pipeline and its refusal layers are compared against.

The prompt is deliberately neutral (a helpful Kannada farm assistant). It
does NOT use rag_service.SYSTEM_INSTRUCTION, which tells the model to
answer only from the provided context and so would make a context-free
baseline refuse by construction. The prompt text is stored in every record.

Token budget: the pipeline's interactive generation budget first, with one
retry at a larger budget if the reasoning model returns nothing (the same
failure as TODO #48/#50). The attempt used is recorded, so an empty reply
is never mistaken for a refusal.

Usage:
    cd backend && python eval/run_no_retrieval_baseline.py [--questions PATH]
Writes eval/runs/baseline/<model>__<questions_hash>__<timestamp>.jsonl.
Makes one or two paid LLM calls per question.
"""
import os
import sys
import json
import time
import hashlib
import asyncio
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from llm_client import call_llm
from eval.question_sets import load_questions

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_PATH = os.path.join(EVAL_DIR, "questions.json")
RUNS_DIR = os.path.join(EVAL_DIR, "runs", "baseline")

# Same string as rag_service.NOT_IN_CONTEXT_PHRASE; duplicated so this
# script does not load the embedding models just to read a constant.
# test_question_sets.py checks the two stay equal.
NOT_IN_CONTEXT_PHRASE = "ಕ್ಷಮಿಸಿ, ಈ ಮಾಹಿತಿ ನಮ್ಮ ಡೇಟಾಬೇಸ್ನಲ್ಲಿ ಲಭ್ಯವಿಲ್ಲ."

BASELINE_PROMPT = (
    "You are an agricultural assistant for sugarcane farmers in Karnataka. "
    "Answer the farmer's question in simple Kannada. "
    "If you do not know the answer, say so."
)
MAX_TOKENS_PER_ATTEMPT = (
    int(os.getenv("INTERACTIVE_MAX_PREDICT", 250)),
    1024,
)


def _short_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


async def main(questions_path):
    questions, questions_bytes = load_questions(questions_path, NOT_IN_CONTEXT_PHRASE)
    model = os.getenv("GENERATION_MODEL")
    if not model:
        sys.exit("GENERATION_MODEL is not set in backend/.env")
    meta = {
        "pipeline": "baseline:no_retrieval",
        "model": model,
        "llm_backend": os.getenv("LLM_BACKEND", "groq"),
        "prompt": BASELINE_PROMPT,
        "prompt_version": _short_hash(BASELINE_PROMPT.encode("utf-8")),
        "questions_file": os.path.basename(questions_path),
        "questions_hash": _short_hash(questions_bytes),
        "run_timestamp": time.strftime("%Y%m%dT%H%M%S"),
    }
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    os.makedirs(RUNS_DIR, exist_ok=True)
    out_path = os.path.join(
        RUNS_DIR, f"{model.replace('/', '-')}__{meta['questions_hash']}__{meta['run_timestamp']}.jsonl"
    )

    results = []
    try:
        for q in questions:
            answer, attempt_used, t0 = None, None, time.time()
            for attempt, max_tokens in enumerate(MAX_TOKENS_PER_ATTEMPT, start=1):
                answer = await call_llm(
                    messages=[{"role": "system", "content": BASELINE_PROMPT},
                              {"role": "user", "content": q["question"]}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                if answer and answer.strip():
                    attempt_used = attempt
                    break
            results.append({
                **meta,
                "id": q["id"],
                "gold_category": q["category"],
                "question": q["question"],
                "expected_answer": q["expected_answer"],
                "expected_refusal": q["expected_refusal"],
                "generated_answer": answer or "",
                # None: both attempts came back empty (a failure, not a refusal).
                "attempt_used": attempt_used,
                "latency_seconds": time.time() - t0,
            })
            print(f"[{q['id']}] attempt={attempt_used} chars={len(answer or '')}", flush=True)
    finally:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(results)} records to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=QUESTIONS_PATH)
    asyncio.run(main(parser.parse_args().questions))
