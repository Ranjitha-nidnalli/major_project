"""
One-line-of-purpose integrity check for gold.jsonl (per this session's
gold-labeling request): every gold_chunk_id referenced must actually exist
in the current chunk set. Fails loudly (non-zero exit, explicit list of
dangling IDs) rather than silently producing an ablation that scores 0 on
a question whose gold chunk ID just doesn't resolve anymore.

Usage: cd backend && python eval/check_gold_integrity.py
Requires eval/chunks.jsonl to be a fresh dump (run dump_chunks.py first).
"""
import json
import os
import sys

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "chunks.jsonl")
GOLD_PATH = os.path.join(os.path.dirname(__file__), "gold.jsonl")


def main():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        current_chunk_ids = {json.loads(line)["chunk_id"] for line in f if line.strip()}

    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        gold_records = [json.loads(line) for line in f if line.strip()]

    dangling = []
    for rec in gold_records:
        for cid in rec["gold_chunk_ids"]:
            if cid not in current_chunk_ids:
                dangling.append((rec["id"], cid))

    if dangling:
        print(f"FAIL: {len(dangling)} dangling gold_chunk_id reference(s):")
        for qid, cid in dangling:
            print(f"  {qid} -> {cid} (not in current {len(current_chunk_ids)}-chunk set)")
        sys.exit(1)

    total_gold_refs = sum(len(r["gold_chunk_ids"]) for r in gold_records)
    unanswerable_count = sum(1 for r in gold_records if r["unanswerable"])
    print(
        f"OK: all {total_gold_refs} gold_chunk_id references across "
        f"{len(gold_records)} questions resolve in the current "
        f"{len(current_chunk_ids)}-chunk set. "
        f"{unanswerable_count} marked unanswerable."
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
