"""
threshold_sweep.py

Empirically justify confidence thresholds using gold-labeled data.

Given gold.jsonl (answerable vs unanswerable labels), runs retrieval for each
question and scores candidate thresholds. Reports:
  - True Positive Rate (answerable questions that pass the threshold)
  - True Negative Rate (unanswerable questions that are refused)
  - F1 score for the binary classification (pass vs refuse)

Usage:
    cd backend
    python eval/threshold_sweep.py

Output: a table of candidate thresholds with TPR, TNR, F1.
"""
import os
import sys
import json
import math
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import models
from vector_db import db_client, COLLECTION_NAME, embed_model

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")
GOLD_PATH = os.path.join(os.path.dirname(__file__), "gold.jsonl")


def get_vectors(q_text):
    out = embed_model.encode([q_text], return_dense=True, return_sparse=True)
    dense_vec = out["dense_vecs"][0].tolist()
    lex_weights = out["lexical_weights"][0]
    sp_indices = [int(k) for k in lex_weights.keys()]
    sp_values = [float(v) for v in lex_weights.values()]
    return dense_vec, sp_indices, sp_values


def retrieve_best_score(q_text):
    """Run hybrid retrieval and return the best fused score."""
    d_vec, s_idx, s_val = get_vectors(q_text)
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
    return hits[0].score if hits else 0.0


def evaluate_threshold(scores, labels, threshold):
    """
    labels: list of bool, True = answerable, False = unanswerable.
    scores: list of float, retrieval best score per question.
    threshold: float, questions with score >= threshold are "passed".
    """
    tp = fp = tn = fn = 0
    for score, is_answerable in zip(scores, labels):
        passed = score >= threshold
        if is_answerable and passed:
            tp += 1
        elif is_answerable and not passed:
            fn += 1
        elif not is_answerable and not passed:
            tn += 1
        elif not is_answerable and passed:
            fp += 1

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return tpr, tnr, precision, recall, f1


def main():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = {q["id"]: q for q in json.load(f)}
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        gold = {rec["id"]: rec for rec in (json.loads(line) for line in f)}

    # Collect scores
    scores = []
    labels = []
    print("Running retrieval for all questions to collect scores...")
    for qid, g in gold.items():
        q = questions[qid]
        score = retrieve_best_score(q["question"])
        is_answerable = not g["unanswerable"]
        scores.append(score)
        labels.append(is_answerable)
        print(f"  {qid}: score={score:.4f}, answerable={is_answerable}")

    # Sweep thresholds
    print("\n" + "=" * 70)
    print("Threshold Sweep Results")
    print("=" * 70)
    print(f"{'Threshold':>10s} {'TPR':>8s} {'TNR':>8s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s}")
    print("-" * 70)

    candidates = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    best_f1 = 0.0
    best_thresh = 0.0

    for thresh in candidates:
        tpr, tnr, precision, recall, f1 = evaluate_threshold(scores, labels, thresh)
        print(f"{thresh:>10.2f} {tpr:>8.3f} {tnr:>8.3f} {precision:>10.3f} {recall:>8.3f} {f1:>8.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print("-" * 70)
    print(f"\n🏆 Best F1 = {best_f1:.3f} at threshold = {best_thresh:.2f}")

    # Safety-critical analysis: for pest/disease/fertilizer, recommend stricter threshold
    safety_scores = []
    safety_labels = []
    for qid, g in gold.items():
        q = questions[qid]
        if q["category"] in {"pest", "disease", "fertilizer"}:
            score = retrieve_best_score(q["question"])
            is_answerable = not g["unanswerable"]
            safety_scores.append(score)
            safety_labels.append(is_answerable)

    if safety_scores:
        print("\n" + "=" * 70)
        print("Safety-Critical Subset (pest/disease/fertilizer) Threshold Sweep")
        print("=" * 70)
        print(f"{'Threshold':>10s} {'TPR':>8s} {'TNR':>8s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s}")
        print("-" * 70)
        best_f1_safety = 0.0
        best_thresh_safety = 0.0
        for thresh in candidates:
            tpr, tnr, precision, recall, f1 = evaluate_threshold(safety_scores, safety_labels, thresh)
            print(f"{thresh:>10.2f} {tpr:>8.3f} {tnr:>8.3f} {precision:>10.3f} {recall:>8.3f} {f1:>8.3f}")
            if f1 > best_f1_safety:
                best_f1_safety = f1
                best_thresh_safety = thresh
        print("-" * 70)
        print(f"\n🏆 Safety-critical best F1 = {best_f1_safety:.3f} at threshold = {best_thresh_safety:.2f}")
        print(f"\n📌 Recommendation:")
        print(f"   General HARD_REFUSAL_THRESHOLD: {best_thresh:.2f} (F1={best_f1:.3f})")
        print(f"   Safety-critical threshold:      {best_thresh_safety:.2f} (F1={best_f1_safety:.3f})")


if __name__ == "__main__":
    main()
