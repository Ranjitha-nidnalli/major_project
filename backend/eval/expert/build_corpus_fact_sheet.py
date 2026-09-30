"""
Build the corpus fact-verification sheet for the agri expert (THE PLAN,
days 1-2). One row per indexed chunk, with its exact text as the model
sees it, so every chemical, dose, and unit can be checked against a real
source before the day-7 run.

Reads the chunks straight from Qdrant (same IDs the pipeline retrieves),
so a correction can be traced back to a chunk ID. Writes a UTF-8 CSV
with BOM so Excel shows Kannada correctly. Expert columns are left empty
on purpose: this script never fills in or suggests a correction.

Usage:
    cd backend && python eval/expert/build_corpus_fact_sheet.py
Stop main.py / run_live_eval.py first if Qdrant is in file mode (single-
process lock).
"""
import os
import re
import csv

from dotenv import load_dotenv
from qdrant_client import QdrantClient

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_fact_sheet.csv")
COLLECTION_NAME = "sugarcane_knowledge"

# Flags rows the expert should check first. A flag is only a reading aid,
# not a claim that the row is (or isn't) safety-critical.
_DOSE_RE = re.compile(r"\d|Chemical|Dosage|ರಾಸಾಯನಿಕ", re.IGNORECASE)
_TAMIL_NADU_RE = re.compile(r"Tamil Nadu|TNAU|Coimbatore|Erode|Thanjavur|ತಮಿಳುನಾಡು", re.IGNORECASE)

EXPERT_COLUMNS = [
    "expert_facts_correct (yes / no / partly)",
    "expert_correction (exact corrected text; leave empty if correct)",
    "expert_valid_for_karnataka (yes / no / unsure)",
    "expert_keep_in_corpus (keep / remove)",
    "expert_source (book, document or page you checked against)",
    "expert_notes",
]


def _client():
    load_dotenv(os.path.join(BACKEND_DIR, ".env"))
    url = os.getenv("QDRANT_URL")
    if url:
        return QdrantClient(url=url)
    return QdrantClient(path=os.path.join(BACKEND_DIR, "qdrant_sugarcane_db"))


def main():
    points, _ = _client().scroll(COLLECTION_NAME, limit=10_000, with_payload=True)
    rows = []
    for p in points:
        text = p.payload["text"]
        rows.append({
            "chunk_id": str(p.id),
            "category": p.payload.get("category", ""),
            "topic": p.payload.get("topic", ""),
            "has_numbers_or_chemicals": "yes" if _DOSE_RE.search(text) else "no",
            "mentions_tamil_nadu": "yes" if _TAMIL_NADU_RE.search(text) else "no",
            "chunk_text": text,
        })
    # Safety-critical categories and dose-bearing rows first.
    order = {"pest": 0, "disease": 1, "fertilizer": 2, "weed": 3}
    rows.sort(key=lambda r: (r["has_numbers_or_chemicals"] != "yes", order.get(r["category"], 9), r["topic"]))

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + EXPERT_COLUMNS)
        writer.writeheader()
        for i, r in enumerate(rows, start=1):
            writer.writerow({**r, **{c: "" for c in EXPERT_COLUMNS}})

    flagged = sum(r["has_numbers_or_chemicals"] == "yes" for r in rows)
    tn = sum(r["mentions_tamil_nadu"] == "yes" for r in rows)
    print(f"Wrote {len(rows)} chunks to {OUTPUT_PATH} ({flagged} with numbers/chemicals, {tn} mentioning Tamil Nadu).")


if __name__ == "__main__":
    main()
