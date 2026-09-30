"""
Turn a live-pipeline run (eval/runs/live/*.jsonl) into the answer-rating
sheet for the agri expert (THE PLAN, days 8-9).

Shows what the farmer actually received plus, when a check refused it,
the model's answer before refusal, so the expert can also judge whether
the refusal was right. No automatic score is shown to the expert: chrF
and the LLM judge are not quality evidence (CLAUDE.md) and would anchor
the rating.

Usage:
    cd backend && python eval/expert/export_rating_sheet.py eval/runs/live/<run>.jsonl
Writes eval/expert/rating_sheet__<run>.csv (UTF-8 with BOM, for Excel).
"""
import os
import sys
import csv
import json

EXPERT_DIR = os.path.dirname(os.path.abspath(__file__))

EXPERT_COLUMNS = [
    "expert_answer_correct (correct / partly / wrong / refused)",
    "expert_should_have_refused (yes / no)",
    "expert_dose_correct (yes / no / no dose given)",
    "expert_could_cause_harm (yes / no)",
    "expert_kannada_clarity_1_to_5",
    "expert_notes",
]


def main(run_path):
    with open(run_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        sys.exit(f"No records in {run_path}")

    name = os.path.splitext(os.path.basename(run_path))[0]
    out_path = os.path.join(EXPERT_DIR, f"rating_sheet__{name}.csv")
    fields = ["row", "id", "question", "expected_answer", "answer_given_to_farmer",
              "refused", "answer_before_refusal"] + EXPERT_COLUMNS
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, r in enumerate(records, start=1):
            raw = r.get("raw_generation") or {}
            writer.writerow({
                "row": i,
                "id": r["id"],
                "question": r["question"],
                "expected_answer": r["expected_answer"],
                "answer_given_to_farmer": r["generated_answer"],
                "refused": "yes" if r["refused"] else "no",
                "answer_before_refusal": (raw.get("raw_answer") or "") if r["refused"] else "",
                **{c: "" for c in EXPERT_COLUMNS},
            })
    print(f"Wrote {len(records)} rows to {out_path} (model {records[0].get('model')}).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
