"""
Chunks withheld from the corpus (corpus_exclusions.json), applied by
vector_db.load_and_chunk_data, which feeds both Qdrant and
bm25_retriever. Kept separate from vector_db so it can be tested without
loading the embedding models.
"""
import os
import json
import uuid


EXCLUSIONS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "corpus_exclusions.json"
)


def apply_exclusions(chunks, exclusions_file=EXCLUSIONS_FILE):
    """
    Drop chunks listed in corpus_exclusions.json, from both indexes (this
    loader feeds Qdrant and bm25_retriever alike). Subtractive only: a
    chunk is withheld whole, never edited, so no dose is ever changed or
    added here. IDs are the same uuid5(text) the upsert uses.

    Fails loudly if a listed ID matches no chunk: an exclusion that
    silently stops applying would put a withheld dose back in the corpus.
    """
    if not os.path.exists(exclusions_file):
        return chunks

    with open(exclusions_file, "r", encoding="utf-8") as f:
        excluded = {e["chunk_id"] for e in json.load(f)["excluded"]}

    kept = []
    found = set()
    for chunk in chunks:
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["text"]))
        if chunk_id in excluded:
            found.add(chunk_id)
        else:
            kept.append(chunk)

    missing = excluded - found
    if missing:
        raise ValueError(
            f"{len(missing)} excluded chunk ID(s) match no chunk "
            f"(corpus or chunker changed?): {sorted(missing)}"
        )

    print(f"Withheld {len(found)} chunks listed in {os.path.basename(exclusions_file)}.")
    return kept
