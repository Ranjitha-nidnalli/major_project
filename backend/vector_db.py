
import json
import os
import uuid
from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder

import sys

from semantic_chunker import semantic_chunk_document
from indic_preprocess import normalize_kannada  # NEW: Kannada normalization

# 1. SETUP
qdrant_path = "./qdrant_sugarcane_db"
model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
reranker_model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

print(f"Loading {model_name}...")
embed_model = BGEM3FlagModel(model_name, use_fp16=False)

print(f"Loading {reranker_model_name}...")
reranker_model = CrossEncoder(reranker_model_name)

try:
    db_client = QdrantClient(path=qdrant_path)
except Exception as e:
    err_str = str(e)
    if "AlreadyLocked" in err_str or "already accessed" in err_str or "PermissionError" in str(type(e)):
        print("\n" + "="*70)
        print("!!! QDRANT DB SERVER IS LOCKED! !!!")
        print("Local mode requires exclusive access. Please stop main.py before")
        print("running tests. Run `python unlock_db.py` to kill zombie processes.")
        print("="*70 + "\n")
        sys.exit(1)
    raise e
COLLECTION_NAME = "sugarcane_knowledge"


# 2. THE CLEANING & CHUNKING LOGIC

def flatten_value(value, indent=0):
    """Recursively renders nested dicts/lists as readable indented text."""
    prefix = "  " * indent
    lines = []
    if isinstance(value, dict):
        for k, v in value.items():
            label = str(k).replace('_', ' ').title()
            if isinstance(v, (dict, list)):
                lines.append(f"{prefix}{label}:")
                lines.append(flatten_value(v, indent + 1))
            else:
                lines.append(f"{prefix}{label}: {v}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(flatten_value(item, indent))
                lines.append("")
            else:
                lines.append(f"{prefix}- {item}")
    else:
        lines.append(f"{prefix}{value}")
    return "\n".join(lines)


def _infer_category(section_key: str) -> str:
    """Infer document category from section key name."""
    cat_lower = section_key.lower()
    if "disease" in cat_lower or "rot" in cat_lower or "smut" in cat_lower or "wilt" in cat_lower:
        return "disease"
    elif "pest" in cat_lower or "borer" in cat_lower or "bug" in cat_lower:
        return "pest"
    elif "soil" in cat_lower or "land" in cat_lower or "climate" in cat_lower:
        return "soil"
    elif "fertilizer" in cat_lower or "nutrient" in cat_lower or "manure" in cat_lower:
        return "fertilizer"
    elif "weed" in cat_lower:
        return "weed"
    elif "irrigation" in cat_lower or "water" in cat_lower:
        return "irrigation"
    else:
        return "general"


def load_and_clean_data_semantic(file_path: str, max_chunk_chars: int = 1200) -> list:
    """
    Load JSON and create SEMANTIC chunks with Indic NLP normalization.

    Strategy:
    1. Flatten each JSON section to readable text
    2. NORMALIZE with Indic NLP (handles agglutination artifacts)
    3. Pass normalized text through semantic_chunk_document()
    4. The semantic chunker uses BGE-M3 embeddings to detect topic boundaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    crop_main = data.get("crop_name", "Sugarcane")
    all_chunks = []

    for section_key, content in data.items():
        if section_key in ("crop_name", "metadata"):
            continue

        category = _infer_category(section_key)
        header = f"Crop: {crop_main} | Topic: {section_key.replace('_', ' ').title()}"

        if isinstance(content, (dict, list)):
            body = flatten_value(content)
        else:
            body = str(content)

        # CRITICAL: Normalize Kannada text BEFORE chunking
        # This improves embedding quality for agglutinated forms
        body = normalize_kannada(body)
        header = normalize_kannada(header)

        section_chunks = semantic_chunk_document(
            header=header,
            body=body,
            embed_model=embed_model,
            category=category,
            max_chunk_chars=max_chunk_chars
        )
        all_chunks.extend(section_chunks)

    return all_chunks


# --- Legacy chunkers (kept for backward compatibility) ---

def load_and_clean_data(file_path):
    """Original flat chunking — kept for backward compatibility."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    documents = []
    crop_main = data.get("crop_name", "Sugarcane")
    for section_key, content in data.items():
        if section_key == "crop_name": continue
        category = _infer_category(section_key)
        header = f"Crop: {crop_main} | Topic: {section_key.replace('_', ' ').title()}\n"
        body = flatten_value(content) + "\n"
        documents.append({"text": header + body, "category": category})
    return documents


def create_overlapping_chunks(documents, max_chars=1000, overlap_chars=200):
    """Original sliding-window chunker — kept for backward compatibility."""
    final_chunks = []
    for doc_obj in documents:
        doc = doc_obj["text"]
        cat = doc_obj["category"]
        if len(doc) <= max_chars:
            final_chunks.append({"text": doc, "category": cat})
            continue
        start = 0
        while start < len(doc):
            end = start + max_chars
            chunk = doc[start:end]
            final_chunks.append({"text": chunk, "category": cat})
            if end >= len(doc): break
            start += (max_chars - overlap_chars)
    return final_chunks


# 3. THE BUILDER

def build_database(use_semantic: bool = True, max_chunk_chars: int = 1200):
    """Build the Qdrant database."""
    if db_client.collection_exists(COLLECTION_NAME):
        print("🧹 Clearing old database...")
        db_client.delete_collection(collection_name=COLLECTION_NAME)

    db_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )

    print("📖 Processing JSON...")
    if use_semantic:
        print("   Using SEMANTIC chunking + Indic NLP normalization...")
        chunk_objs = load_and_clean_data_semantic('sugarcanemerged3.json', max_chunk_chars=max_chunk_chars)
    else:
        print("   Using legacy sliding-window chunking...")
        raw_docs = load_and_clean_data('sugarcanemerged3.json')
        chunk_objs = create_overlapping_chunks(raw_docs, max_chars=max_chunk_chars, overlap_chars=200)

    chunks = [c["text"] for c in chunk_objs]
    metadatas = [{"category": c["category"], "text": c["text"]} for c in chunk_objs]

    print(f"🧬 Generating Dense & Sparse Embeddings for {len(chunks)} chunks...")
    output = embed_model.encode(chunks, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    dense_vecs = output['dense_vecs']
    lexical_weights_list = output['lexical_weights']

    points = []
    for i in range(len(chunks)):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunks[i]))
        lex_weights = lexical_weights_list[i]
        sparse_indices = [int(k) for k in lex_weights.keys()]
        sparse_values = [float(v) for v in lex_weights.values()]

        points.append(
            models.PointStruct(
                id=point_id,
                payload=metadatas[i],
                vector={
                    "dense": dense_vecs[i].tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    )
                }
            )
        )

    db_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("🚀 Local Qdrant Hybrid Database built successfully!")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Chunking mode: {'semantic + indic-nlp' if use_semantic else 'sliding-window'}")

    sizes = [len(c) for c in chunks]
    if sizes:
        print(f"   Chunk sizes: min={min(sizes)}, avg={sum(sizes)//len(sizes)}, max={max(sizes)}")


# Wrapper for embedding queries with Indic NLP normalization
def embed_query(text: str):
    """
    Embed a query text with Indic NLP normalization.
    Ensures queries are normalized the same way as corpus chunks.
    """
    normalized = normalize_kannada(text)
    output = embed_model.encode([normalized], return_dense=True, return_sparse=True)
    dense_vec = output['dense_vecs'][0].tolist()
    lex_weights = output['lexical_weights'][0]
    sp_indices = [int(k) for k in lex_weights.keys()]
    sp_values = [float(v) for v in lex_weights.values()]
    return dense_vec, sp_indices, sp_values


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build Qdrant database for Krishi Mitra")
    parser.add_argument("--legacy", action="store_true", help="Use legacy sliding-window chunking")
    parser.add_argument("--max-chars", type=int, default=1200, help="Max chars per chunk")
    args = parser.parse_args()

    build_database(use_semantic=not args.legacy, max_chunk_chars=args.max_chars)
    print("Database is ready and loaded.")
