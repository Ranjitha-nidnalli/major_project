"""
P3.4 — Refusal-accuracy metric for unanswerable questions.

Tests whether the system correctly refuses when the corpus has no
answering chunk. Loads gold.jsonl, filters unanswerable questions,
runs each through the live RAG pipeline, and checks if the response
is a refusal (contains the refusal marker) rather than a hallucinated
answer.

Usage:
    cd backend && python eval/refusal_test.py

Requires: main.py stopped (Qdrant exclusive access), Ollama running.
"""
import os
import sys
import json
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_service import get_sugarcane_answer

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")
GOLD_PATH = os.path.join(os.path.dirname(__file__), "gold.jsonl")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "refusal_results.jsonl")

# The refusal message contains this Kannada marker — we check for it
REFUSAL_MARKERS = [
    "ಕ್ಷಮಿಸಿ",
    "ಲಭ್ಯವಿಲ್ಲ",
    "not in our database",
    "ದಯವಿಟ್ಟು ಬೇರೆ ರೀತಿಯಲ್ಲಿ ಕೇಳಿ",
]


def is_refusal(answer: str) -> bool:
    """Heuristic: does the answer look like a refusal?"""
    if not answer:
        return True
    lower = answer.lower()
    return any(m.lower() in lower for m in REFUSAL_MARKERS)


async def main():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = {q["id"]: q for q in json.load(f)}
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        gold = {rec["id"]: rec for rec in (json.loads(line) for line in f)}

    unanswerable = [qid for qid, g in gold.items() if g["unanswerable"]]
    if not unanswerable:
        print("No unanswerable questions found in gold.jsonl — add some (e.g. price-1).")
        return

    print(f"Testing {len(unanswerable)} unanswerable question(s): {unanswerable}\n")

    results = []
    correct_refusals = 0

    for qid in unanswerable:
        q = questions[qid]
        print(f"[{qid}] {q['question'][:60]}...", flush=True)

        result = await get_sugarcane_answer(q["question"], session_id=f"refusal_test_{qid}")
        answer = result.get("answer", "")
        refused = is_refusal(answer)

        if refused:
            correct_refusals += 1
            status = "✅ REFUSED"
        else:
            status = "❌ HALLUCINATED"

        print(f"  {status} — {answer[:100]}...")

        results.append({
            "id": qid,
            "question": q["question"],
            "answer": answer,
            "refused": refused,
            "search_score": result.get("search_score"),
            "accuracy_score": result.get("accuracy_score"),
        })

    total = len(unanswerable)
    rate = correct_refusals / total if total else 0.0

    print("\n" + "=" * 50)
    print("REFUSAL ACCURACY REPORT")
    print("=" * 50)
    print(f"Correct refusals : {correct_refusals}/{total}")
    print(f"Refusal rate     : {rate:.2%}")
    print("=" * 50)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote detailed results to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
