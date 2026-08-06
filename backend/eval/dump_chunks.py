"""
P1.1 - dumps every chunk in the Qdrant collection to chunks.jsonl for manual
gold-label review: {chunk_id, category, text}.

Chunk ids are deterministic (uuid5 hash of chunk text, see vector_db.py) so
they stay stable across re-seeding as long as the chunk text itself doesn't
change - gold.jsonl labels keyed on these ids survive a rebuild.

Requires exclusive access to the local Qdrant store - stop main.py first.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_db import db_client, COLLECTION_NAME

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "chunks.jsonl")


def main():
    points, _ = db_client.scroll(
        collection_name=COLLECTION_NAME, limit=1000, with_payload=True, with_vectors=False
    )
    print(f"Dumping {len(points)} chunks to {CHUNKS_PATH}")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for p in points:
            record = {
                "chunk_id": str(p.id),
                "category": p.payload.get("category"),
                "text": p.payload.get("text"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
