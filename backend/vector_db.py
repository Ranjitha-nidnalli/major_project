import json
import os
import sys
import uuid
from typing import Any, Dict, List

from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder

from indic_preprocess import normalize_kannada


# ============================================================
# 1. CONFIGURATION
# ============================================================

qdrant_path = "./qdrant_sugarcane_db"

model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
reranker_model_name = os.getenv(
    "RERANKER_MODEL",
    "BAAI/bge-reranker-v2-m3"
)

DATA_FILE = os.getenv(
    "SUGARCANE_DATA_FILE",
    "sugarcanemerged3.json"
)

COLLECTION_NAME = "sugarcane_knowledge"

# A soft upper bound, not a reason to cut through a record.
DEFAULT_MAX_CHUNK_CHARS = 1400


# ============================================================
# 2. LOAD MODELS
# ============================================================

print(f"Loading {model_name}...")
embed_model = BGEM3FlagModel(
    model_name,
    use_fp16=False
)

print(f"Loading {reranker_model_name}...")
reranker_model = CrossEncoder(reranker_model_name)


# ============================================================
# 3. QDRANT CLIENT
# ============================================================

try:
    db_client = QdrantClient(path=qdrant_path)

except Exception as e:
    err_str = str(e)

    if (
        "AlreadyLocked" in err_str
        or "already accessed" in err_str
        or "PermissionError" in str(type(e))
    ):
        print("\n" + "=" * 70)
        print("!!! QDRANT DB SERVER IS LOCKED! !!!")
        print("Local mode requires exclusive access.")
        print("Please stop main.py before running tests.")
        print("Run `python unlock_db.py` to kill zombie processes.")
        print("=" * 70 + "\n")

        sys.exit(1)

    raise


# ============================================================
# 4. HELPERS
# ============================================================

def pretty_key(key: Any) -> str:
    """
    Convert JSON keys into readable labels.

    Example:
        nutrient_management -> Nutrient Management
    """
    return str(key).replace("_", " ").strip().title()


def is_scalar(value: Any) -> bool:
    return not isinstance(value, (dict, list))


def normalize_text(text: str) -> str:
    """
    Normalize Kannada while also removing excessive blank lines.
    """
    text = normalize_kannada(str(text))

    lines = [
        line.rstrip()
        for line in text.splitlines()
    ]

    cleaned = []
    previous_blank = False

    for line in lines:
        blank = not line.strip()

        if blank and previous_blank:
            continue

        cleaned.append(line)
        previous_blank = blank

    return "\n".join(cleaned).strip()


# ============================================================
# 5. CATEGORY INFERENCE
# ============================================================

def _infer_category(section_key: str) -> str:
    """
    Infer broad retrieval category from the top-level dataset section.
    """

    cat_lower = section_key.lower()

    if any(
        word in cat_lower
        for word in [
            "disease",
            "rot",
            "smut",
            "wilt",
            "pathogen"
        ]
    ):
        return "disease"

    if any(
        word in cat_lower
        for word in [
            "pest",
            "borer",
            "bug",
            "insect",
            "aphid",
            "termite"
        ]
    ):
        return "pest"

    if any(
        word in cat_lower
        for word in [
            "soil",
            "land",
            "climate"
        ]
    ):
        return "soil"

    if any(
        word in cat_lower
        for word in [
            "fertilizer",
            "nutrient",
            "manure",
            "micronutrient"
        ]
    ):
        return "fertilizer"

    if "weed" in cat_lower:
        return "weed"

    if any(
        word in cat_lower
        for word in [
            "irrigation",
            "water"
        ]
    ):
        return "irrigation"

    return "general"


# ============================================================
# 6. STRUCTURE-AWARE RENDERING
# ============================================================

def render_value(
    value: Any,
    indent: int = 0
) -> str:
    """
    Render JSON recursively without losing field names.

    This is used only after the chunk boundary has already been
    chosen structurally.
    """

    prefix = " " * indent
    lines = []

    if isinstance(value, dict):

        for key, child in value.items():

            label = pretty_key(key)

            if is_scalar(child):
                lines.append(
                    f"{prefix}{label}: {child}"
                )

            else:
                lines.append(
                    f"{prefix}{label}:"
                )

                child_text = render_value(
                    child,
                    indent + 2
                )

                if child_text:
                    lines.append(child_text)

    elif isinstance(value, list):

        for item in value:

            if is_scalar(item):
                lines.append(
                    f"{prefix}- {item}"
                )

            else:
                item_text = render_value(
                    item,
                    indent + 2
                )

                lines.append(
                    f"{prefix}-"
                )

                if item_text:
                    lines.append(item_text)

    else:
        lines.append(
            f"{prefix}{value}"
        )

    return "\n".join(lines)


# ============================================================
# 7. DETECT STRUCTURAL RECORDS
# ============================================================

def looks_like_entity_record(value: Any) -> bool:
    """
    Decide whether a dictionary represents one independent,
    retrievable unit.

    Examples likely to return True:
        - one pest
        - one disease
        - one variety
        - one regional nutrient profile
        - one chemical recommendation
        - one irrigation stage
        - one application stage

    The function deliberately does not depend on exact field names
    because the dataset contains different schemas across sections.
    """

    if not isinstance(value, dict):
        return False

    if not value:
        return False

    keys = {
        str(k).lower()
        for k in value.keys()
    }

    identity_keys = {
        "name",
        "variety",
        "region",
        "stage",
        "season",
        "chemical",
        "element",
        "title",
        "type"
    }

    if keys & identity_keys:
        return True

    # A compact dictionary consisting mainly of scalar fields
    # is also likely to be a meaningful record.
    scalar_count = sum(
        1
        for v in value.values()
        if is_scalar(v)
    )

    return (
        scalar_count >= 2
        and scalar_count >= len(value) * 0.6
    )


# ============================================================
# 8. STRUCTURE-AWARE CHUNKING
# ============================================================

def make_chunk(
    crop_name: str,
    topic: str,
    category: str,
    body: str,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Construct one final chunk with stable contextual information.
    """

    header = (
        f"Crop: {crop_name} | "
        f"Topic: {topic}"
    )

    text = f"{header}\n{body}".strip()

    text = normalize_text(text)

    payload = {
        "text": text,
        "category": category,
        "topic": topic,
        "crop": crop_name
    }

    if metadata:
        for key, value in metadata.items():
            if value is not None:
                payload[key] = value

    return payload


def split_large_record(
    crop_name: str,
    topic: str,
    category: str,
    record: Dict[str, Any],
    max_chunk_chars: int
) -> List[Dict[str, Any]]:
    """
    Split a genuinely large record only at child boundaries.

    Important:
    We never slice raw text by character position.
    """

    full_body = render_value(record)

    if len(full_body) <= max_chunk_chars:
        return [
            make_chunk(
                crop_name,
                topic,
                category,
                full_body
            )
        ]

    chunks = []

    # Keep scalar fields as shared context.
    scalar_context = {}

    # Complex children are potential chunk boundaries.
    complex_children = []

    for key, value in record.items():

        if is_scalar(value):
            scalar_context[key] = value

        else:
            complex_children.append(
                (key, value)
            )

    # If there is nothing meaningful to split,
    # preserve the whole record rather than cutting arbitrarily.
    if not complex_children:
        return [
            make_chunk(
                crop_name,
                topic,
                category,
                full_body
            )
        ]

    for key, value in complex_children:

        subrecord = dict(scalar_context)
        subrecord[key] = value

        sub_body = render_value(subrecord)

        # If one child is itself too large,
        # split lists into record-sized units.
        if (
            len(sub_body) > max_chunk_chars
            and isinstance(value, list)
        ):
            for item in value:

                item_record = dict(scalar_context)
                item_record[key] = [item]

                item_body = render_value(
                    item_record
                )

                chunks.append(
                    make_chunk(
                        crop_name,
                        topic,
                        category,
                        item_body
                    )
                )

        else:
            chunks.append(
                make_chunk(
                    crop_name,
                    topic,
                    category,
                    sub_body
                )
            )

    return chunks


def chunk_section(
    crop_name: str,
    section_key: str,
    content: Any,
    max_chunk_chars: int
) -> List[Dict[str, Any]]:
    """
    Adaptive structure-aware chunking for one top-level JSON section.

    Strategy:

    1. A list of entity records:
       one chunk per entity.

    2. A dictionary containing lists of entity records:
       one chunk per record.

    3. Compact dictionaries:
       keep together.

    4. Large dictionaries:
       split at child field boundaries.

    5. Never split by arbitrary character windows.
    """

    category = _infer_category(section_key)

    topic = pretty_key(section_key)

    chunks = []

    # --------------------------------------------------------
    # CASE 1: TOP-LEVEL LIST
    # --------------------------------------------------------

    if isinstance(content, list):

        scalar_items = []

        for item in content:

            if isinstance(item, dict):

                if scalar_items:
                    body = render_value(
                        scalar_items
                    )

                    chunks.append(
                        make_chunk(
                            crop_name,
                            topic,
                            category,
                            body
                        )
                    )

                    scalar_items = []

                chunks.extend(
                    split_large_record(
                        crop_name,
                        topic,
                        category,
                        item,
                        max_chunk_chars
                    )
                )

            else:
                scalar_items.append(item)

        if scalar_items:

            body = render_value(
                scalar_items
            )

            chunks.append(
                make_chunk(
                    crop_name,
                    topic,
                    category,
                    body
                )
            )

        return chunks

    # --------------------------------------------------------
    # CASE 2: TOP-LEVEL DICTIONARY
    # --------------------------------------------------------

    if isinstance(content, dict):

        # First preserve top-level scalar fields as section context.
        section_scalars = {
            key: value
            for key, value in content.items()
            if is_scalar(value)
        }

        complex_children = [
            (key, value)
            for key, value in content.items()
            if not is_scalar(value)
        ]

        # Compact section: keep everything together.
        full_body = render_value(content)

        if len(full_body) <= max_chunk_chars:

            return [
                make_chunk(
                    crop_name,
                    topic,
                    category,
                    full_body
                )
            ]

        # Large section with no complex children:
        # preserve it intact instead of arbitrary slicing.
        if not complex_children:

            return [
                make_chunk(
                    crop_name,
                    topic,
                    category,
                    full_body
                )
            ]

        # Split each meaningful child.
        for child_key, child_value in complex_children:

            child_topic = (
                f"{topic} - "
                f"{pretty_key(child_key)}"
            )

            # --------------------------------------------
            # LIST OF RECORDS
            # --------------------------------------------

            if isinstance(child_value, list):

                # If the list contains independent records,
                # one chunk per record.
                dict_items = [
                    item
                    for item in child_value
                    if isinstance(item, dict)
                ]

                if dict_items:

                    for item in child_value:

                        if isinstance(item, dict):

                            record = dict(
                                section_scalars
                            )

                            # Keep the child label visible.
                            record[child_key] = item

                            chunks.extend(
                                split_large_record(
                                    crop_name,
                                    child_topic,
                                    category,
                                    record,
                                    max_chunk_chars
                                )
                            )

                        else:
                            body_record = dict(
                                section_scalars
                            )

                            body_record[child_key] = item

                            chunks.append(
                                make_chunk(
                                    crop_name,
                                    child_topic,
                                    category,
                                    render_value(
                                        body_record
                                    )
                                )
                            )

                else:
                    # List of scalar values stays together.
                    record = dict(
                        section_scalars
                    )

                    record[child_key] = child_value

                    chunks.append(
                        make_chunk(
                            crop_name,
                            child_topic,
                            category,
                            render_value(record)
                        )
                    )

            # --------------------------------------------
            # NESTED DICTIONARY
            # --------------------------------------------

            elif isinstance(child_value, dict):

                # If it already looks like one independent
                # entity/record, keep it together.
                if looks_like_entity_record(
                    child_value
                ):

                    record = dict(
                        section_scalars
                    )

                    record[child_key] = child_value

                    chunks.extend(
                        split_large_record(
                            crop_name,
                            child_topic,
                            category,
                            record,
                            max_chunk_chars
                        )
                    )

                else:

                    # Examine nested children.
                    nested_split = False

                    for nested_key, nested_value in child_value.items():

                        if (
                            isinstance(
                                nested_value,
                                list
                            )
                            and nested_value
                            and all(
                                isinstance(x, dict)
                                for x in nested_value
                            )
                        ):

                            nested_split = True

                            for item in nested_value:

                                record = dict(
                                    section_scalars
                                )

                                record[child_key] = {
                                    nested_key: item
                                }

                                chunks.extend(
                                    split_large_record(
                                        crop_name,
                                        child_topic,
                                        category,
                                        record,
                                        max_chunk_chars
                                    )
                                )

                    if not nested_split:

                        record = dict(
                            section_scalars
                        )

                        record[child_key] = child_value

                        chunks.extend(
                            split_large_record(
                                crop_name,
                                child_topic,
                                category,
                                record,
                                max_chunk_chars
                            )
                        )

        # Add purely scalar section context if it exists.
        # This prevents scalar-only information from disappearing
        # when a large section is split into child chunks.
        if section_scalars:

            scalar_body = render_value(
                section_scalars
            )

            chunks.insert(
                0,
                make_chunk(
                    crop_name,
                    topic,
                    category,
                    scalar_body
                )
            )

        return chunks

    # --------------------------------------------------------
    # CASE 3: SCALAR
    # --------------------------------------------------------

    return [
        make_chunk(
            crop_name,
            topic,
            category,
            str(content)
        )
    ]


# ============================================================
# 9. LOAD DATA USING ADAPTIVE CHUNKING
# ============================================================

def load_and_chunk_data(
    file_path: str,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS
) -> List[Dict[str, Any]]:
    """
    Load the JSON corpus and apply adaptive structure-aware chunking.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    crop_main = data.get(
        "crop_name",
        "Sugarcane"
    )

    crop_main = normalize_text(
        crop_main
    )

    all_chunks = []

    for section_key, content in data.items():

        if section_key in (
            "crop_name",
            "metadata"
        ):
            continue

        section_chunks = chunk_section(
            crop_name=crop_main,
            section_key=section_key,
            content=content,
            max_chunk_chars=max_chunk_chars
        )

        all_chunks.extend(
            section_chunks
        )

    # Remove exact duplicate chunks while preserving order.
    seen = set()
    unique_chunks = []

    for chunk in all_chunks:

        text = chunk["text"]

        if not text or text in seen:
            continue

        seen.add(text)

        unique_chunks.append(chunk)

    return unique_chunks


# ============================================================
# 10. BUILD QDRANT DATABASE
# ============================================================

def build_database(
    data_file: str = DATA_FILE,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS
):
    """
    Build the Qdrant database using adaptive structure-aware chunks.
    """

# Delete the old collection completely before rebuilding
if db_client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing collection: {COLLECTION_NAME}")
    db_client.delete_collection(COLLECTION_NAME)

# Create a fresh collection
db_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(
            size=DENSE_VECTOR_SIZE,
            distance=models.Distance.COSINE,
        )
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams()
    },
)

print(f"Created fresh collection: {COLLECTION_NAME}")

    chunk_objs = load_and_chunk_data(
        data_file,
        max_chunk_chars=max_chunk_chars
    )

    chunks = [
        chunk["text"]
        for chunk in chunk_objs
    ]

    metadatas = [
        dict(chunk)
        for chunk in chunk_objs
    ]

    if not chunks:
        raise ValueError(
            "No chunks were generated. "
            "Check the dataset path and JSON structure."
        )

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

    points = []

    for i, chunk_text in enumerate(chunks):

        # Stable deterministic ID.
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk_text
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

                    "dense":
                        dense_vecs[i].tolist(),

                    "sparse":
                        models.SparseVector(
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

    print(
        "🚀 Local Qdrant Hybrid Database "
        "built successfully!"
    )

    print(
        f"📦 Total chunks: {len(chunks)}"
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
            sum(sizes) / len(sizes)
        )

        print(
            f"📏 Chunk sizes: "
            f"min={min(sizes)}, "
            f"avg={average_size:.1f}, "
            f"max={max(sizes)}"
        )

    # Print a small topic/category distribution.
    distribution = {}

    for chunk in chunk_objs:

        category = chunk["category"]

        distribution[category] = (
            distribution.get(category, 0) + 1
        )

    print("\n📊 Chunk distribution:")

    for category, count in sorted(
        distribution.items()
    ):
        print(
            f"  {category}: {count}"
        )


# ============================================================
# 11. QUERY EMBEDDING
# ============================================================

def embed_query(text: str):
    """
    Embed a query using the same Kannada normalization
    used for corpus chunks.
    """

    normalized = normalize_text(text)

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
# 12. COMMAND LINE
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
        "--data-file",
        default=DATA_FILE,
        help=(
            "Path to the sugarcane JSON dataset"
        )
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHUNK_CHARS,
        help=(
            "Soft maximum chunk size. "
            "Records are never cut arbitrarily."
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
