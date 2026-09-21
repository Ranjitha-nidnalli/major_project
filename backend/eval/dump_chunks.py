"""
P1.1 - dumps every chunk in the Qdrant collection to chunks.jsonl for manual
gold-label review: {chunk_id, category, text}.

Chunk ids are deterministic (uuid5 hash of chunk text, see vector_db.py) so
they stay stable across re-seeding as long as the chunk text itself doesn't
change - gold.jsonl labels keyed on these ids survive a rebuild.

Runs against whatever Qdrant instance QDRANT_URL points at (server mode -- see docker-compose.yml). No exclusive-access requirement in server mode.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_db import db_client, COLLECTION_NAME

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "chunks.jsonl")


def _scroll_all_points():
    """
    Paginate through the full collection. A single scroll() call caps at
    its `limit` and silently returns only the first page -- fine while the
    corpus is ~43 chunks, but TODO #33 (multi-crop expansion) would make a
    fixed limit silently drop chunks past it, right before a human is
    meant to review every one for gold labeling.
    """
    points = []
    offset = None
    while True:
        batch, offset = db_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break
    return points


def main():
    points = _scroll_all_points()
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
