
"""
auto_relabel_gold.py

Automatically rebuilds gold.jsonl after chunking changes.

For each eval question, finds the chunk(s) that best match the expected answer
using a combination of:
1. Embedding similarity (BGE-M3 dense vectors)
2. Keyword overlap (key terms from expected answer must appear in chunk)

Usage:
    cd backend
    # After rebuilding database with new chunking
    python eval/auto_relabel_gold.py

Output: eval/gold.jsonl (overwrites old one)
"""
import os
import sys
import json
import re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_db import embed_model, db_client, COLLECTION_NAME
from qdrant_client import models

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")
GOLD_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "gold.jsonl")

# Thresholds
EMBEDDING_SIM_THRESHOLD = 0.65      # Minimum cosine sim to be considered a match
KEYWORD_MATCH_MIN = 2               # Minimum unique keywords that must appear
TOP_K_CANDIDATES = 10               # How many chunks to retrieve per question


def extract_keywords(text: str) -> set:
    """
    Extract meaningful keywords from text.
    Keeps Kannada words, English words, and numbers.
    Removes common stopwords and short tokens.
    """
    # Normalize
    text = text.lower()
    # Split on non-word chars (including Kannada)
    tokens = re.findall(r'[\u0C80-\u0CFF]+|[a-z]+|[0-9]+(?:\.[0-9]+)?', text)

    # Kannada + English stopwords
    stopwords = {
        # Kannada
        'ಮತ್ತು', 'ಅಥವಾ', 'ಆಗಿ', 'ಇದು', 'ಅದು', 'ಇಲ್ಲಿ', 'ಅಲ್ಲಿ', 'ಎಂದು', 'ಎಂಬ',
        'ಯಾವ', 'ಎಷ್ಟು', 'ಹೇಗೆ', 'ಏನು', 'ಎಲ್ಲಿ', 'ಯಾರು', 'ಯಾವುದು', 'ಎಷ್ಟೊಂದು',
        'ನ', 'ರ', 'ಗಳ', 'ಗಳು', 'ದ', 'ವ', 'ಕ್ಕೆ', 'ನಲ್ಲಿ', 'ಅನ್ನು', 'ಇಂದ', 'ವರೆಗೆ',
        # English
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'there',
    }

    keywords = set()
    for t in tokens:
        if len(t) > 2 and t not in stopwords:
            keywords.add(t)
    return keywords


def keyword_overlap_score(chunk_text: str, answer_keywords: set) -> float:
    """What fraction of answer keywords appear in the chunk?"""
    if not answer_keywords:
        return 0.0
    chunk_text_lower = chunk_text.lower()
    matched = sum(1 for kw in answer_keywords if kw in chunk_text_lower)
    return matched / len(answer_keywords)


def embedding_similarity(text_a: str, text_b: str) -> float:
    """Compute BGE-M3 dense cosine similarity between two texts."""
    out = embed_model.encode([text_a, text_b], return_dense=True, return_sparse=False)
    v1, v2 = out["dense_vecs"][0], out["dense_vecs"][1]
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def find_gold_chunks(question: str, expected_answer: str, category: str) -> list:
    """
    Find the chunk IDs that best contain the expected answer.

    Strategy:
    1. Retrieve top-k chunks by embedding similarity to the question
    2. Score each candidate by: embedding_sim(answer, chunk) + keyword_overlap
    3. Select chunks that pass both thresholds
    """
    # Step 1: Retrieve candidates using the question embedding
    out = embed_model.encode([question], return_dense=True, return_sparse=True)
    dense_vec = out["dense_vecs"][0].tolist()
    lex_weights = out["lexical_weights"][0]
    sp_indices = [int(k) for k in lex_weights.keys()]
    sp_values = [float(v) for v in lex_weights.values()]

    response = db_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_vec, using="dense", limit=TOP_K_CANDIDATES),
            models.Prefetch(query=models.SparseVector(indices=sp_indices, values=sp_values), using="sparse", limit=TOP_K_CANDIDATES),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=TOP_K_CANDIDATES,
    )
    candidates = response.points

    # Step 2: Extract keywords from expected answer
    answer_keywords = extract_keywords(expected_answer)

    # Step 3: Score each candidate
    scored = []
    for hit in candidates:
        chunk_text = hit.payload.get("text", "")
        chunk_id = str(hit.id)

        # Embedding similarity between expected_answer and chunk
        emb_sim = embedding_similarity(expected_answer, chunk_text)

        # Keyword overlap
        kw_score = keyword_overlap_score(chunk_text, answer_keywords)

        # Combined score (embedding is primary, keywords are secondary filter)
        combined = emb_sim + (0.1 * kw_score)  # small weight for keywords

        scored.append({
            "id": chunk_id,
            "text": chunk_text[:150],
            "emb_sim": emb_sim,
            "kw_score": kw_score,
            "combined": combined,
        })

    # Step 4: Select gold chunks
    # Primary criterion: embedding similarity above threshold
    # Secondary criterion: at least N keywords match
    gold_ids = []
    for s in scored:
        if s["emb_sim"] >= EMBEDDING_SIM_THRESHOLD and s["kw_score"] >= 0.3:
            gold_ids.append(s["id"])

    # If nothing passes strict threshold, take top 2 by combined score
    if not gold_ids:
        scored.sort(key=lambda x: x["combined"], reverse=True)
        gold_ids = [s["id"] for s in scored[:2]]

    return gold_ids, scored


def main():
    print("=" * 60)
    print("Auto Gold Label Rebuilder")
    print("=" * 60)

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    gold_records = []

    for q in questions:
        qid = q["id"]
        question_text = q["question"]
        expected = q["expected_answer"]
        category = q["category"]

        print(f"\n🔍 {qid}: {question_text[:50]}...")

        # Unanswerable questions (like price queries) get empty gold
        if category == "price" or "ಬೆಲೆ" in question_text or "price" in question_text.lower():
            gold_ids = []
            print(f"   → Unanswerable (price query)")
        else:
            gold_ids, scored = find_gold_chunks(question_text, expected, category)
            print(f"   → Found {len(gold_ids)} gold chunk(s)")
            for s in scored[:3]:
                match_flag = "✅" if s["id"] in gold_ids else ""
                print(f"      {match_flag} {s['id'][:8]}... emb={s['emb_sim']:.3f} kw={s['kw_score']:.2f} | {s['text'][:60]}...")

        gold_records.append({
            "id": qid,
            "gold_chunk_ids": gold_ids,
            "unanswerable": len(gold_ids) == 0,
        })

    # Write new gold.jsonl
    with open(GOLD_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in gold_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ Wrote {len(gold_records)} records to {GOLD_OUTPUT_PATH}")

    # Summary stats
    answerable = sum(1 for r in gold_records if not r["unanswerable"])
    unanswerable = sum(1 for r in gold_records if r["unanswerable"])
    avg_gold_per_q = sum(len(r["gold_chunk_ids"]) for r in gold_records) / max(answerable, 1)

    print(f"\n📊 Summary:")
    print(f"   Answerable:     {answerable}")
    print(f"   Unanswerable:   {unanswerable}")
    print(f"   Avg gold chunks per answerable Q: {avg_gold_per_q:.1f}")
    print(f"\n⚠️  IMPORTANT: Review the output manually!")
    print(f"   This script uses heuristics (embedding sim + keywords).")
    print(f"   Check a few questions to ensure the matched chunks make sense.")
    print(f"   If a chunk looks wrong, edit {GOLD_OUTPUT_PATH} manually.")


if __name__ == "__main__":
    main()
