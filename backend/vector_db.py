import json
import os
import uuid
import sys
from collections import Counter

from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder

from indic_preprocess import normalize_kannada

# ============================================================
# 1. CONFIGURATION
# ============================================================

qdrant_path = "./qdrant_sugarcane_db"

COLLECTION_NAME = "sugarcane_knowledge"
DATA_FILE = "sugarcanemerged3.json"

DENSE_VECTOR_SIZE = 1024

model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
reranker_model_name = os.getenv(
    "RERANKER_MODEL",
    "BAAI/bge-reranker-v2-m3"
)

print(f"Loading {model_name}...")
embed_model = BGEM3FlagModel(
    model_name,
    use_fp16=False
)

print(f"Loading {reranker_model_name}...")
reranker_model = CrossEncoder(
    reranker_model_name
)

try:
    db_client = QdrantClient(
        path=qdrant_path
    )

except Exception as e:
    err_str = str(e)

    if (
        "AlreadyLocked" in err_str
        or "already accessed" in err_str
        or "PermissionError" in str(type(e))
    ):
        print("\n" + "=" * 70)
        print("!!! QDRANT DB SERVER IS LOCKED! !!!")
        print(
            "Local mode requires exclusive access. "
            "Please stop main.py before running this."
        )
        print(
            "Run `python unlock_db.py` "
            "to kill zombie processes."
        )
        print("=" * 70 + "\n")

        sys.exit(1)

    raise


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def flatten_value(value, indent=0):
    """
    Recursively render nested dictionaries and lists
    as readable text.
    """

    prefix = " " * indent
    lines = []

    if isinstance(value, dict):

        for key, item in value.items():

            label = (
                str(key)
                .replace("_", " ")
                .title()
            )

            if isinstance(item, (dict, list)):

                lines.append(
                    f"{prefix}{label}:"
                )

                lines.append(
                    flatten_value(
                        item,
                        indent + 1
                    )
                )

            else:

                lines.append(
                    f"{prefix}{label}: {item}"
                )

    elif isinstance(value, list):

        for item in value:

            if isinstance(item, (dict, list)):

                lines.append(
                    flatten_value(
                        item,
                        indent
                    )
                )

                lines.append("")

            else:

                lines.append(
                    f"{prefix}- {item}"
                )

    else:

        lines.append(
            f"{prefix}{value}"
        )

    return "\n".join(lines)


def _infer_category(section_key: str) -> str:
    """
    Infer a retrieval category from the JSON section name.
    """

    cat_lower = section_key.lower()

    if (
        "disease" in cat_lower
        or "rot" in cat_lower
        or "smut" in cat_lower
        or "wilt" in cat_lower
    ):
        return "disease"

    elif (
        "pest" in cat_lower
        or "borer" in cat_lower
        or "bug" in cat_lower
    ):
        return "pest"

    elif (
        "soil" in cat_lower
        or "land" in cat_lower
        or "climate" in cat_lower
    ):
        return "soil"

    elif (
        "fertilizer" in cat_lower
        or "nutrient" in cat_lower
        or "manure" in cat_lower
    ):
        return "fertilizer"

    elif "weed" in cat_lower:
        return "weed"

    elif (
        "irrigation" in cat_lower
        or "water" in cat_lower
    ):
        return "irrigation"

    return "general"


# ============================================================
# 3. LEGACY CHUNKING FUNCTIONS
#    Kept for compatibility with evaluation/BM25 code
# ============================================================

def load_and_clean_data(file_path):
    """
    Original flat document loader.

    Kept for backward compatibility with evaluation scripts.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    documents = []

    crop_main = data.get(
        "crop_name",
        "Sugarcane"
    )

    for section_key, content in data.items():

        if section_key == "crop_name":
            continue

        category = _infer_category(
            section_key
        )

        header = (
            f"Crop: {crop_main} | "
            f"Topic: "
            f"{section_key.replace('_', ' ').title()}\n"
        )

        body = (
            flatten_value(content)
            + "\n"
        )

        documents.append(
            {
                "text": header + body,
                "category": category
            }
        )

    return documents


def create_overlapping_chunks(
    documents,
    max_chars=1000,
    overlap_chars=200
):
    """
    Original sliding-window chunker.

    Kept for backward compatibility.
    """

    final_chunks = []

    for doc_obj in documents:

        document = doc_obj["text"]
        category = doc_obj["category"]

        if len(document) <= max_chars:

            final_chunks.append(
                {
                    "text": document,
                    "category": category
                }
            )

            continue

        start = 0

        while start < len(document):

            end = start + max_chars

            chunk = document[start:end]

            final_chunks.append(
                {
                    "text": chunk,
                    "category": category
                }
            )

            if end >= len(document):
                break

            start += (
                max_chars
                - overlap_chars
            )

    return final_chunks


# ============================================================
# 4. ADAPTIVE STRUCTURE-AWARE CHUNKING
# ============================================================

def _make_chunk(
    text,
    category,
    topic
):
    """
    Normalize and create a chunk dictionary.
    """

    text = normalize_kannada(
        text.strip()
    )

    return {
        "text": text,
        "category": category,
        "topic": topic
    }


def _split_large_text(
    text,
    max_chunk_chars
):
    """
    Split oversized text conservatively.

    Preference order:
    paragraphs -> lines -> sentence boundaries -> hard split.
    """

    if len(text) <= max_chunk_chars:
        return [text]

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    if len(paragraphs) <= 1:

        paragraphs = [
            p.strip()
            for p in text.split("\n")
            if p.strip()
        ]

    chunks = []
    current = ""

    for paragraph in paragraphs:

        candidate = (
            paragraph
            if not current
            else current + "\n\n" + paragraph
        )

        if len(candidate) <= max_chunk_chars:

            current = candidate

        else:

            if current:
                chunks.append(
                    current.strip()
                )

            if len(paragraph) <= max_chunk_chars:

                current = paragraph

            else:

                start = 0

                while start < len(paragraph):

                    end = (
                        start
                        + max_chunk_chars
                    )

                    chunks.append(
                        paragraph[start:end].strip()
                    )

                    start = end

                current = ""

    if current:
        chunks.append(
            current.strip()
        )

    return chunks


def _adaptive_chunk_section(
    section_key,
    content,
    crop_name,
    max_chunk_chars
):
    """
    Adaptive structure-aware chunking.

    Each top-level dataset section is preserved as a topic.
    Nested structures are flattened into readable blocks,
    while large sections are split without mixing unrelated
    topics.
    """

    category = _infer_category(
        section_key
    )

    topic = (
        section_key
        .replace("_", " ")
        .title()
    )

    header = (
        f"Crop: {crop_name} "
        f"| Topic: {topic}"
    )

    if isinstance(content, dict):

        units = []

        for key, value in content.items():

            label = (
                str(key)
                .replace("_", " ")
                .title()
            )

            if isinstance(value, (dict, list)):

                unit_text = (
                    f"{label}:\n"
                    f"{flatten_value(value)}"
                )

            else:

                unit_text = (
                    f"{label}: {value}"
                )

            units.append(
                unit_text
            )

    elif isinstance(content, list):

        units = []

        for item in content:

            if isinstance(item, (dict, list)):

                unit_text = (
                    flatten_value(item)
                )

            else:

                unit_text = str(item)

            units.append(
                unit_text
            )

    else:

        units = [
            str(content)
        ]

    chunks = []
    current = header

    for unit in units:

        candidate = (
            current
            + "\n"
            + unit
        )

        if len(candidate) <= max_chunk_chars:

            current = candidate

        else:

            if current != header:

                chunks.extend(
                    _split_large_text(
                        current,
                        max_chunk_chars
                    )
                )

            current = (
                header
                + "\n"
                + unit
            )

            if len(current) > max_chunk_chars:

                chunks.extend(
                    _split_large_text(
                        current,
                        max_chunk_chars
                    )
                )

                current = header

    if current != header:

        chunks.extend(
            _split_large_text(
                current,
                max_chunk_chars
            )
        )

    return [
        _make_chunk(
            text=chunk,
            category=category,
            topic=topic
        )
        for chunk in chunks
        if chunk.strip()
    ]


def load_and_chunk_data(
    file_path,
    max_chunk_chars=1200
):
    """
    Load sugarcanemerged3.json using adaptive
    structure-aware chunking.
    """

    print(
        "Loading chunks using "
        "adaptive structure-aware chunking..."
    )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    crop_name = data.get(
        "crop_name",
        "Sugarcane"
    )

    all_chunks = []

    for section_key, content in data.items():

        if section_key in (
            "crop_name",
            "metadata"
        ):
            continue

        section_chunks = (
            _adaptive_chunk_section(
                section_key=section_key,
                content=content,
                crop_name=crop_name,
                max_chunk_chars=max_chunk_chars
            )
        )

        all_chunks.extend(
            section_chunks
        )

    return all_chunks


# ============================================================
# 5. DUPLICATE PROTECTION
# ============================================================

def remove_exact_duplicates(chunk_objs):
    """
    Remove exact duplicate chunk text before embedding/upsert.

    This is an additional safeguard even though UUID5 is
    deterministic. It prevents duplicate corpus content from
    entering the evaluation corpus.
    """

    seen = set()
    unique_chunks = []

    for chunk in chunk_objs:

        normalized_text = (
            chunk["text"]
            .strip()
        )

        if normalized_text in seen:
            continue

        seen.add(
            normalized_text
        )

        unique_chunks.append(
            chunk
        )

    removed = (
        len(chunk_objs)
        - len(unique_chunks)
    )

    if removed > 0:

        print(
            f"⚠️ Removed {removed} "
            f"exact duplicate chunk(s)."
        )

    return unique_chunks


# ============================================================
# 6. DATABASE BUILDER
# ============================================================

def build_database(
    data_file=DATA_FILE,
    max_chunk_chars=1200
):
    """
    Build a completely fresh Qdrant hybrid database.

    IMPORTANT:
    The existing collection is always deleted first.
    This prevents stale chunks from previous chunking
    strategies contaminating the new corpus.
    """

    print("📖 Processing JSON...")

    chunk_objs = load_and_chunk_data(
        data_file,
        max_chunk_chars=max_chunk_chars
    )

    chunk_objs = remove_exact_duplicates(
        chunk_objs
    )

    chunks = [
        chunk["text"]
        for chunk in chunk_objs
    ]

    metadatas = [
        {
            "category": chunk["category"],
            "topic": chunk["topic"],
            "text": chunk["text"]
        }
        for chunk in chunk_objs
    ]

    if not chunks:

        raise ValueError(
            "No chunks were generated. "
            "Check the dataset path and JSON structure."
        )

    # --------------------------------------------------------
    # CRITICAL: ALWAYS REBUILD THE COLLECTION FROM SCRATCH
    # --------------------------------------------------------
print(
        f"🧹 Recreating collection from scratch: "
        f"{COLLECTION_NAME}"
    )

db_client.recreate_collection(
        collection_name=COLLECTION_NAME,

        vectors_config={
            "dense": models.VectorParams(
                size=DENSE_VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        },

        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )
    

    # --------------------------------------------------------
    # GENERATE EMBEDDINGS
    # --------------------------------------------------------

    print(
        f"🧬 Generating Dense & Sparse Embeddings "
        f"for {len(chunks)} chunks..."
    )

    output = embed_model.encode(
        chunks,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False
    )

    dense_vecs = output[
        "dense_vecs"
    ]

    lexical_weights_list = output[
        "lexical_weights"
    ]

    # --------------------------------------------------------
    # CREATE QDRANT POINTS
    # --------------------------------------------------------

    points = []

    for i in range(
        len(chunks)
    ):

        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunks[i]
            )
        )

        lex_weights = (
            lexical_weights_list[i]
        )

        sparse_indices = [
            int(key)
            for key in lex_weights.keys()
        ]

        sparse_values = [
            float(value)
            for value in lex_weights.values()
        ]

        points.append(

            models.PointStruct(

                id=point_id,

                payload=metadatas[i],

                vector={

                    "dense": (
                        dense_vecs[i]
                        .tolist()
                    ),

                    "sparse": (
                        models.SparseVector(
                            indices=sparse_indices,
                            values=sparse_values
                        )
                    )

                }

            )

        )

    # --------------------------------------------------------
    # UPSERT
    # --------------------------------------------------------

    db_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    # --------------------------------------------------------
    # VERIFICATION
    # --------------------------------------------------------

    points_after, _ = db_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        with_payload=True,
        with_vectors=False
    )

    stored_texts = [
        point.payload
        .get("text", "")
        .strip()
        for point in points_after
    ]

    text_counts = Counter(
        stored_texts
    )

    duplicate_groups = sum(
        1
        for count in text_counts.values()
        if count > 1
    )

    extra_duplicates = sum(
        count - 1
        for count in text_counts.values()
        if count > 1
    )

    print(
        "\n🚀 Local Qdrant Hybrid Database "
        "built successfully!"
    )

    print(
        f"📦 Total generated chunks: "
        f"{len(chunks)}"
    )

    print(
        f"📦 Total stored points: "
        f"{len(points_after)}"
    )

    print(
        "🧩 Chunking mode: "
        "adaptive structure-aware"
    )

    sizes = [
        len(chunk)
        for chunk in chunks
    ]

    if sizes:

        average_size = (
            sum(sizes)
            / len(sizes)
        )

        print(
            f"📏 Chunk sizes: "
            f"min={min(sizes)}, "
            f"avg={average_size:.1f}, "
            f"max={max(sizes)}"
        )

    category_counts = Counter(
        chunk["category"]
        for chunk in chunk_objs
    )

    print(
        "\n📊 Chunk distribution:"
    )

    for category in sorted(
        category_counts
    ):

        print(
            f"  {category}: "
            f"{category_counts[category]}"
        )

    print(
        "\n🔍 Duplicate verification:"
    )

    print(
        f"  Unique texts: "
        f"{len(text_counts)}"
    )

    print(
        f"  Exact duplicate groups: "
        f"{duplicate_groups}"
    )

    print(
        f"  Extra duplicate chunks: "
        f"{extra_duplicates}"
    )

    if len(points_after) != len(chunks):

        raise RuntimeError(
            "Qdrant point count does not match "
            "generated chunk count."
        )

    if extra_duplicates != 0:

        raise RuntimeError(
            "Exact duplicate chunks detected "
            "after database build."
        )

    print(
        "\n✅ Database verification passed."
    )


# ============================================================
# 7. QUERY EMBEDDING
# ============================================================

def embed_query(text: str):
    """
    Embed a query using the same Kannada normalization
    applied to corpus content.
    """

    normalized = normalize_kannada(
        text
    )

    output = embed_model.encode(
        [normalized],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False
    )

    dense_vec = (
        output["dense_vecs"][0]
        .tolist()
    )

    lex_weights = (
        output["lexical_weights"][0]
    )

    sparse_indices = [
        int(key)
        for key in lex_weights.keys()
    ]

    sparse_values = [
        float(value)
        for value in lex_weights.values()
    ]

    return (
        dense_vec,
        sparse_indices,
        sparse_values
    )


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Build Qdrant database "
            "for Krishi Mitra"
        )
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help=(
            "Maximum characters "
            "per chunk"
        )
    )

    parser.add_argument(
        "--data-file",
        default=DATA_FILE,
        help=(
            "Path to the sugarcane "
            "JSON dataset"
        )
    )

    args = parser.parse_args()

    build_database(
        data_file=args.data_file,
        max_chunk_chars=args.max_chars
    )

    print(
        "\nDatabase is ready and loaded."
    )
