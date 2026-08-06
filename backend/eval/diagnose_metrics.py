"""
P0.1 - metric sanity check. Before trusting any ablation numbers, prove the
scoring functions actually discriminate between related and unrelated text.

For each metric, computes:
  - self-score: reference scored against itself (expected ~1.0 if healthy)
  - unrelated-score: reference scored against a different, unrelated reference
    from the eval set (the metric's noise floor)

If self-score and unrelated-score are close together, the metric has no usable
range for this data and any conclusion drawn from small differences within
that range is noise, not signal.

Does not need the Qdrant lock beyond importing embed_model (vector_db.py opens
the Qdrant client eagerly at import time) - stop main.py first regardless.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rouge_score import rouge_scorer
import sacrebleu

from vector_db import embed_model

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")


def cosine(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def main():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    refs = [q["expected_answer"] for q in questions]
    ids = [q["id"] for q in questions]
    n = len(refs)

    # Rotate by 1 to pair each reference with a different, unrelated one -
    # deterministic and guarantees no accidental self-pairing.
    shuffled_refs = refs[1:] + refs[:1]

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    rouge_self = [rouge.score(r, r)["rougeL"].fmeasure for r in refs]
    rouge_unrelated = [rouge.score(refs[i], shuffled_refs[i])["rougeL"].fmeasure for i in range(n)]

    chrf_self = [sacrebleu.sentence_chrf(r, [r]).score / 100 for r in refs]
    chrf_unrelated = [sacrebleu.sentence_chrf(refs[i], [shuffled_refs[i]]).score / 100 for i in range(n)]

    print("Embedding all references + shuffled pairing...")
    all_vecs = embed_model.encode(refs, return_dense=True)["dense_vecs"]
    shuffled_vecs = np.roll(all_vecs, -1, axis=0)
    emb_self = [cosine(all_vecs[i], all_vecs[i]) for i in range(n)]
    emb_unrelated = [cosine(all_vecs[i], shuffled_vecs[i]) for i in range(n)]

    def avg(x):
        return sum(x) / len(x)

    print("\nPer-question detail:")
    print(f"{'id':<14}{'rougeL_self':<13}{'rougeL_unrl':<13}{'chrF_self':<11}{'chrF_unrl':<11}{'embsim_self':<13}{'embsim_unrl':<13}")
    for i in range(n):
        print(
            f"{ids[i]:<14}{rouge_self[i]:<13.3f}{rouge_unrelated[i]:<13.3f}"
            f"{chrf_self[i]:<11.3f}{chrf_unrelated[i]:<11.3f}"
            f"{emb_self[i]:<13.3f}{emb_unrelated[i]:<13.3f}"
        )

    print("\n=== Metric Sanity Summary ===")
    print(f"{'metric':<16}{'self-score':<13}{'unrelated-score':<17}{'usable range':<13}")
    for name, self_scores, unrel_scores in [
        ("ROUGE-L", rouge_self, rouge_unrelated),
        ("chrF", chrf_self, chrf_unrelated),
        ("embedding-sim", emb_self, emb_unrelated),
    ]:
        s, u = avg(self_scores), avg(unrel_scores)
        print(f"{name:<16}{s:<13.3f}{u:<17.3f}{s - u:<13.3f}")

    print(
        "\nInterpretation: a metric is usable for this ablation only if "
        "(self-score - unrelated-score) is comfortably larger than the spread "
        "seen between retrieval configs. If our 6 retrieval configs scored "
        "within ~0.04 of each other yesterday, compare that gap to the ranges above."
    )


if __name__ == "__main__":
    main()
