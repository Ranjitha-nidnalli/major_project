import json
import os
import uuid
from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder

import sys

# Import Indic NLP preprocessing
from indic_preprocess import normalize_kannada, normalize_batch

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


def _create_logical_unit_chunk(header: str, body: str, category: str) -> dict:
    """Create a single chunk dict with normalized text."""
    raw_text = header + "\n" + body
    # Apply Indic NLP normalization
    normalized_text = normalize_kannada(raw_text)
    return {"text": normalized_text, "category": category}


def _chunk_array_section(section_key: str, items: list, crop_main: str, max_chars: int = 1200) -> list:
    """
    Chunk an array section (pest_management, disease_management, etc.)
    into one chunk per logical array element.
    If a single element exceeds max_chars, apply sliding window within it.
    """
    chunks = []
    category = _infer_category(section_key)
    for item in items:
        body = flatten_value(item)
        header = f"Crop: {crop_main} | Topic: {section_key.replace('_', ' ').title()}"
        chunk = _create_logical_unit_chunk(header, body, category)
        # If this single logical unit is too long, split it
        if len(chunk["text"]) > max_chars:
            chunks.extend(_sliding_window_chunk(chunk["text"], category, max_chars))
        else:
            chunks.append(chunk)
    return chunks


def _chunk_dict_section(section_key: str, content: dict, crop_main: str, max_chars: int = 1200) -> list:
    """
    Chunk a dict section. For small dicts, one chunk.
    For large dicts with array sub-fields, chunk each array element separately.
    """
    category = _infer_category(section_key)

    # Special handling for nutrient_management — it has arrays inside
    if section_key == "nutrient_management":
        chunks = []
        # Regional profiles
        if "regional_profiles" in content:
            for profile in content["regional_profiles"]:
                body = flatten_value(profile)
                header = f"Crop: {crop_main} | Topic: Nutrient Management — {profile.get('region', 'General')}"
                chunk = _create_logical_unit_chunk(header, body, category)
                if len(chunk["text"]) > max_chars:
                    chunks.extend(_sliding_window_chunk(chunk["text"], category, max_chars))
                else:
                    chunks.append(chunk)
        # General profiles
        if "general_profiles" in content:
            for profile in content["general_profiles"]:
                body = flatten_value(profile)
                header = f"Crop: {crop_main} | Topic: Nutrient Management — General"
                chunk = _create_logical_unit_chunk(header, body, category)
                if len(chunk["text"]) > max_chars:
                    chunks.extend(_sliding_window_chunk(chunk["text"], category, max_chars))
                else:
                    chunks.append(chunk)
        # Application schedules
        for key in ["application_schedule_detailed", "application_schedule_general"]:
            if key in content:
                for stage in content[key]:
                    body = flatten_value(stage)
                    header = f"Crop: {crop_main} | Topic: Nutrient Schedule — {stage.get('stage', 'General')}"
                    chunk = _create_logical_unit_chunk(header, body, category)
                    if len(chunk["text"]) > max_chars:
                        chunks.extend(_sliding_window_chunk(chunk["text"], category, max_chars))
                    else:
                        chunks.append(chunk)
        # Other keys (nutrient_specifics, micronutrients, etc.)
        for key in ["nutrient_specifics", "micronutrients_and_biofertilizers"]:
            if key in content:
                body = flatten_value(content[key])
                header = f"Crop: {crop_main} | Topic: Nutrient Management — {key.replace('_', ' ').title()}"
                chunk = _create_logical_unit_chunk(header, body, category)
                if len(chunk["text"]) > max_chars:
                    chunks.extend(_sliding_window_chunk(chunk["text"], category, max_chars))
                else:
                    chunks.append(chunk)
        return chunks

    # Default: flatten the whole dict and chunk if too long
    header = f"Crop: {crop_main} | Topic: {section_key.replace('_', ' ').title()}"
    body = flatten_value(content)
    chunk = _create_logical_unit_chunk(header, body, category)
    if len(chunk["text"]) > max_chars:
        return _sliding_window_chunk(chunk["text"], category, max_chars)
    return [chunk]


def _sliding_window_chunk(text: str, category: str, max_chars: int = 1200, overlap_chars: int = 200) -> list:
    """
    Fallback sliding-window chunker for text that exceeds max_chars.
    Respects sentence boundaries where possible.
    """
    chunks = []
    if len(text) <= max_chars:
        chunks.append({"text": text, "category": category})
        return chunks

    start = 0
    while start < len(text):
        end = start + max_chars
        # Try to find a sentence boundary near the end
        if end < len(text):
            # Look for Kannada danda (।) or period within last 100 chars
            search_start = max(start + max_chars - 100, start)
            boundary = text.rfind('।', search_start, end)
            if boundary == -1:
                boundary = text.rfind('.', search_start, end)
            if boundary != -1 and boundary > start + max_chars // 2:
                end = boundary + 1

        chunk_text = text[start:end]
        chunks.append({"text": chunk_text.strip(), "category": category})
        if end >= len(text):
            break
        start = end - overlap_chars

    return chunks


def load_and_clean_data_field_aware(file_path: str, max_chars: int = 1200) -> list:
    """
    Load JSON and create field-aware chunks.

    Strategy:
    - Array sections (pest, disease, weed, irrigation): one chunk per array element
    - Dict sections: one chunk per logical sub-unit
    - Very long units: sliding-window fallback within the unit
    - All text normalized with Indic NLP before chunking
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    crop_main = data.get("crop_name", "Sugarcane")
    all_chunks = []

    for section_key, content in data.items():
        if section_key == "crop_name":
            continue
        if section_key == "metadata":
            continue

        if isinstance(content, list):
            # Array section: one chunk per element
            all_chunks.extend(_chunk_array_section(section_key, content, crop_main, max_chars))
        elif isinstance(content, dict):
            # Dict section: chunk by logical sub-units
            all_chunks.extend(_chunk_dict_section(section_key, content, crop_main, max_chars))
        else:
            # Simple value: wrap in a chunk
            category = _infer_category(section_key)
            header = f"Crop: {crop_main} | Topic: {section_key.replace('_', ' ').title()}"
            body = str(content)
            chunk = _create_logical_unit_chunk(header, body, category)
            all_chunks.append(chunk)

    return all_chunks


# --- Legacy sliding-window chunker (kept for backward compatibility) ---

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
        # Normalize before chunking
        doc = normalize_kannada(doc)
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

def build_database(use_field_aware: bool = True, max_chars: int = 1200):
    """
    Build the Qdrant database.

    Args:
        use_field_aware: If True, uses field-aware chunking (one logical unit per chunk).
                          If False, uses legacy sliding-window chunking.
        max_chars: Maximum characters per chunk. Field-aware mode uses this as a
                   ceiling for individual logical units, not a fixed window size.
    """
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
            "sparse": models.SparseVectorParams()  # Lexical weights from BGE-M3
        }
    )

    print("📖 Processing JSON...")
    if use_field_aware:
        print("   Using field-aware chunking (Indic NLP + logical units)...")
        chunk_objs = load_and_clean_data_field_aware('sugarcanemerged3.json', max_chars=max_chars)
    else:
        print("   Using legacy sliding-window chunking...")
        raw_docs = load_and_clean_data('sugarcanemerged3.json')
        chunk_objs = create_overlapping_chunks(raw_docs, max_chars=max_chars, overlap_chars=200)

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
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Chunking mode: {'field-aware' if use_field_aware else 'sliding-window'}")

    # Print chunk size distribution
    sizes = [len(c) for c in chunks]
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    max_size = max(sizes) if sizes else 0
    min_size = min(sizes) if sizes else 0
    print(f"   Chunk sizes: min={min_size}, avg={avg_size:.0f}, max={max_size}")


# Wrapper for embedding queries with Indic NLP normalization
def embed_query(text: str):
    """
    Embed a query text with Indic NLP normalization.
    This ensures queries are normalized the same way as corpus chunks.
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

    build_database(use_field_aware=not args.legacy, max_chars=args.max_chars)
    print("Database is ready and loaded.")
