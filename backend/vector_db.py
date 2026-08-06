import json
import os
import uuid
from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder

import sys

# 1. SETUP
qdrant_path = "./qdrant_sugarcane_db"
model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
# If bge-reranker-v2-m3 is too slow on CPU, zeroentropy/zerank-1-small (Apache 2.0, 1.7B)
# is a lighter drop-in alternative worth benchmarking - not switched to yet.
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
    # Recursively renders nested dicts/lists as readable indented "Label: value"
    # lines instead of a Python repr dump (e.g. "Notes: [{'stage': 'Basal', ...}]"),
    # which otherwise poisons embedding/reranking quality for nested JSON fields.
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


def load_and_clean_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    documents = []
    crop_main = data.get("crop_name", "Sugarcane")
    for section_key, content in data.items():
        if section_key == "crop_name": continue
        
        # Assign category based on key
        cat_lower = section_key.lower()
        if "disease" in cat_lower or "rot" in cat_lower or "smut" in cat_lower or "wilt" in cat_lower: 
            category = "disease"
        elif "pest" in cat_lower or "borer" in cat_lower or "bug" in cat_lower: 
            category = "pest"
        elif "soil" in cat_lower or "land" in cat_lower or "climate" in cat_lower: 
            category = "soil"
        elif "fertilizer" in cat_lower or "nutrient" in cat_lower or "manure" in cat_lower: 
            category = "fertilizer"
        else: 
            category = "general"

        header = f"Crop: {crop_main} | Topic: {section_key.replace('_', ' ').title()}\n"
        body = flatten_value(content) + "\n"
        documents.append({"text": header + body, "category": category})
    return documents

def create_overlapping_chunks(documents, max_chars=1000, overlap_chars=200):
    final_chunks = []
    for doc_obj in documents:
        doc = doc_obj["text"]
        cat = doc_obj["category"]
        if len(doc) <= max_chars:
            final_chunks.append({"text": doc, "category": cat})
            continue
        
        # Simple character-level sliding window
        start = 0
        while start < len(doc):
            end = start + max_chars
            chunk = doc[start:end]
            final_chunks.append({"text": chunk, "category": cat})
            if end >= len(doc): break
            start += (max_chars - overlap_chars)
    return final_chunks

# 3. THE BUILDER
def build_database():
    # Clear old data first to avoid mixing models/logic
    if db_client.collection_exists(COLLECTION_NAME):
        print("🧹 Clearing old database...")
        db_client.delete_collection(collection_name=COLLECTION_NAME)

    db_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams() # Lexical weights from BGE-M3
        }
    )

    print("📖 Processing JSON...")
    raw_docs = load_and_clean_data('sugarcanemerged3.json')
    chunk_objs = create_overlapping_chunks(raw_docs, max_chars=1000, overlap_chars=200)

    chunks = [c["text"] for c in chunk_objs]
    metadatas = [{"category": c["category"], "text": c["text"]} for c in chunk_objs]

    print(f"🧬 Generating Dense & Sparse Embeddings for {len(chunks)} chunks...")
    output = embed_model.encode(chunks, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    dense_vecs = output['dense_vecs']
    lexical_weights_list = output['lexical_weights']
    
    points = []
    for i in range(len(chunks)):
        # Deterministic id (hash of chunk text via uuid5) so eval gold labels
        # survive re-seeding instead of getting new random ids each rebuild.
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunks[i]))

        # Parse lexical weights (token_id string -> weight) to sparse indices/values
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
    
    db_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print("🚀 Local Qdrant Hybrid Database built successfully!")

if __name__ == "__main__":
    build_database()
    print("Database is ready and loaded.")
    