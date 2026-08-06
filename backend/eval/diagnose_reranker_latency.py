"""
P0.3 - investigate the ~30s/query reranker latency seen in the retrieval
ablation. Checks the three hypothesized causes from PROJECT_PLAN.md:
  1. Is the model reloaded per call? (it's a module-level singleton in
     vector_db.py, loaded once at import - this measures repeat-call cost
     to confirm there's no hidden per-call reinitialization)
  2. Is scoring done one-pair-at-a-time instead of batched?
  3. Is use_fp16 relevant on this CPU-only machine?

Requires exclusive access to the local Qdrant store - stop main.py first.
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from vector_db import reranker_model

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")

with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    QUESTIONS = json.load(f)

# Build 15 realistic (query, chunk-length-ish text) pairs using real eval
# question/answer text repeated/sliced to ~15 candidates, similar to what the
# ablation script actually scores per query.
QUERY = QUESTIONS[0]["question"]
LONG_TEXT = " ".join(q["expected_answer"] for q in QUESTIONS)  # long realistic Kannada text
PAIRS_15 = [[QUERY, LONG_TEXT[: 200 + i * 60]] for i in range(15)]
PAIRS_5 = PAIRS_15[:5]
PAIRS_1 = PAIRS_15[:1]


def timed_predict(pairs, label, **kwargs):
    t0 = time.time()
    scores = reranker_model.predict(pairs, activation_fn=torch.nn.Sigmoid(), **kwargs)
    dt = time.time() - t0
    print(f"{label:<40s} n_pairs={len(pairs):<4d} time={dt:.2f}s  ({dt/len(pairs):.2f}s/pair)")
    return dt


def main():
    print(f"torch.get_num_threads() = {torch.get_num_threads()}")
    print(f"reranker device = {reranker_model.model.device if hasattr(reranker_model, 'model') else 'unknown'}")
    print()

    # Repeat the same 15-pair call 3x to check for warmup vs steady-state cost.
    timed_predict(PAIRS_15, "15 pairs, call #1 (cold)")
    timed_predict(PAIRS_15, "15 pairs, call #2 (warm)")
    timed_predict(PAIRS_15, "15 pairs, call #3 (warm)")
    print()

    # Scale with pair count to see if cost is roughly linear (inherent
    # per-pair inference cost) or dominated by fixed overhead.
    timed_predict(PAIRS_1, "1 pair")
    timed_predict(PAIRS_5, "5 pairs")
    timed_predict(PAIRS_15, "15 pairs")
    print()

    # Explicit batch_size variations - sentence-transformers default is 32
    # (i.e. already a single batch for 15 pairs). Try forcing smaller batches
    # to see if one-at-a-time scoring would be slower (it should be, if
    # batching is actually helping).
    timed_predict(PAIRS_15, "15 pairs, batch_size=1 (forces sequential)", batch_size=1)
    timed_predict(PAIRS_15, "15 pairs, batch_size=32 (default)", batch_size=32)


if __name__ == "__main__":
    main()
