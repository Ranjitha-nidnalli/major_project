import os
import chromadb


def safe_text(value: str) -> str:
    return value.encode("unicode_escape").decode("ascii")


def inspect_path(db_path: str) -> None:
    print("\n" + "=" * 60)
    print(f"Inspecting path: {os.path.abspath(db_path)}")
    print("=" * 60)

    if not os.path.exists(db_path):
        print("Path does not exist.")
        return

    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    print(f"Collection names: {[c.name for c in collections]}")

    if not collections:
        print("No collections found.")
        return

    for coll_ref in collections:
        collection = client.get_collection(coll_ref.name)
        count = collection.count()
        print(f"\nCollection: {coll_ref.name}")
        print(f"Total document count: {count}")

        sample_n = min(3, count)
        if sample_n == 0:
            print("Collection is empty.")
            continue

        sample = collection.get(
            limit=sample_n,
            include=["documents", "metadatas", "embeddings"],
        )

        ids = sample.get("ids", [])
        docs = sample.get("documents", [])
        metas = sample.get("metadatas", [])
        embs = sample.get("embeddings", [])

        for i in range(sample_n):
            emb_dim = len(embs[i]) if i < len(embs) and embs[i] is not None else 0
            emb_head = embs[i][:8] if i < len(embs) and embs[i] is not None else []
            print(f"\n--- Sample #{i + 1} ---")
            print(f"id: {ids[i] if i < len(ids) else 'N/A'}")
            print(f"metadata: {metas[i] if i < len(metas) else {}}")
            doc = docs[i] if i < len(docs) and docs[i] else ""
            print(f"document: {safe_text(doc[:220])}")
            print(f"embedding_dim: {emb_dim}")
            print(f"embedding_head: {emb_head}")


if __name__ == "__main__":
    # Check both likely locations.
    inspect_path("./sugarcane_vector_db")
    inspect_path("./sugarcane_vector_db/sugarcane_vector_db")
