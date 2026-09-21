"""
Export generated answers to a CSV for human evaluation.

Usage:
    # After running run_eval.py for all 3 models (each run writes its own
    # file under eval/runs/, per CLAUDE.md's "never append to a shared
    # file" rule):
    cd backend && python eval/export_for_human_eval.py

Produces: eval/human_eval_batch.csv
"""
import os
import sys
import glob
import json
import csv

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "human_eval_batch.csv")


def main():
    run_files = sorted(glob.glob(os.path.join(RUNS_DIR, "*.jsonl")))
    if not run_files:
        print(f"❌ No run files found under {RUNS_DIR}. Run run_eval.py first for all models.")
        sys.exit(1)

    records = []
    for path in run_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    if not records:
        print("No records found.")
        return

    # Runs from different dataset states aren't directly comparable -- warn
    # rather than silently combine them into one human-eval batch.
    dataset_hashes = {r.get("dataset_hash") for r in records}
    if len(dataset_hashes) > 1:
        print(
            f"⚠️  Combining runs with {len(dataset_hashes)} different dataset_hash "
            f"values: {dataset_hashes}. These were generated against different "
            f"corpus/context states -- results may not be directly comparable."
        )

    # Deduplicate by (model, id), keeping the most recent run_timestamp --
    # each run file is already a complete, clean single run (no in-file
    # duplicates), so this only matters if run_eval.py was run twice for
    # the same model.
    seen = {}
    for r in records:
        key = (r["model"], r["id"])
        existing = seen.get(key)
        if existing is None or r.get("run_timestamp", "") >= existing.get("run_timestamp", ""):
            seen[key] = r
    records = list(seen.values())
    print(f"Loaded {len(records)} records from {len(run_files)} run file(s) under {RUNS_DIR}")

    # Sample: pick ~10 diverse questions, all 3 models each = ~30 rows
    # Or just export everything and let the rater pick
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "row_num", "model", "judge_model", "id", "category",
            "question", "expected_answer", "generated_answer",
            "chrf", "embedding_similarity", "faithfulness",
            "human_faithfulness_1_to_5", "human_usefulness_1_to_5",
            "human_notes"
        ])
        for i, r in enumerate(records, 1):
            writer.writerow([
                i,
                r.get("model", ""),
                r.get("judge_model", ""),
                r.get("id", ""),
                r.get("category", ""),
                r.get("question", ""),
                r.get("expected_answer", ""),
                r.get("generated_answer", ""),
                f"{r.get('chrf', 0):.3f}",
                f"{r.get('embedding_similarity', 0):.3f}",
                f"{r.get('accuracy_score', 0):.3f}",
                "",  # human_faithfulness
                "",  # human_usefulness
                "",  # human_notes
            ])

    print(f"✅ Exported {len(records)} rows to {OUTPUT_PATH}")
    print("\n📋 Instructions for your human rater:")
    print("   1. Read the Question and Expected Answer (gold reference).")
    print("   2. Read the Generated Answer.")
    print("   3. Rate FAITHFULNESS (1-5): does the answer stick to facts? 1 = hallucinates, 5 = fully grounded.")
    print("   4. Rate USEFULNESS (1-5): does it actually help the farmer? 1 = useless/wrong, 5 = exactly right.")
    print("   5. Add notes for any interesting failures (bad Kannada, wrong dosage, etc.).")


if __name__ == "__main__":
    main()
