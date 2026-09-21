"""
Unit tests for pure helper functions in vector_db.py (TODO #30: "chunking
boundaries, flatten_value(), UUID5 determinism across reseeds").

vector_db.py loads BGE-M3 and the cross-encoder reranker (~7GB combined) as
import-time side effects, and opens a Qdrant client (file-mode lock if
QDRANT_URL is unset). Actually `import vector_db` here would drag that cost
into every `pytest` run, defeating the fast, dependency-free invariant suite
the rest of tests/invariants/ deliberately keeps (see test_category_filter.py,
which builds its own in-memory Qdrant collection for the same reason instead
of importing vector_db.py).

Instead: statically extract the exact source of the pure, self-contained
functions under test via `ast`, and exec just that source in an isolated
namespace. This tests the real, current implementation (not a hand-copied
duplicate that could drift out of sync) without paying the import cost.
"""
import ast
import os

VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "vector_db.py")

with open(VECTOR_DB_PATH, "r", encoding="utf-8") as f:
    _SOURCE = f.read()
_TREE = ast.parse(_SOURCE, filename=VECTOR_DB_PATH)


def _load_function(name):
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(_SOURCE, node)
            namespace = {}
            exec(compile(segment, filename=f"<vector_db.{name}>", mode="exec"), namespace)
            return namespace[name]
    raise AssertionError(f"vector_db.py no longer defines a function named {name!r}")


flatten_value = _load_function("flatten_value")
_infer_category = _load_function("_infer_category")
_split_large_text = _load_function("_split_large_text")


# ---------------------------------------------------------------------------
# flatten_value
# ---------------------------------------------------------------------------

def test_flatten_value_scalar():
    assert flatten_value("hello") == "hello"


def test_flatten_value_flat_dict_renders_key_value_pairs():
    out = flatten_value({"dose_rate": "500g"})
    assert "Dose Rate: 500g" in out


def test_flatten_value_nested_dict_recurses():
    out = flatten_value({"pest": {"chemical": "carbendazim"}})
    assert "Pest:" in out
    assert "Chemical: carbendazim" in out


def test_flatten_value_list_of_scalars_renders_bullets():
    out = flatten_value(["step one", "step two"])
    assert "- step one" in out
    assert "- step two" in out


# ---------------------------------------------------------------------------
# _infer_category
# ---------------------------------------------------------------------------

def test_infer_category_matches_known_keywords():
    cases = {
        "Red Rot Disease": "disease",
        "Stem Borer Pest": "pest",
        "Soil and Climate": "soil",
        "Fertilizer Schedule": "fertilizer",
        "Weed Management": "weed",
        "Irrigation Water": "irrigation",
    }
    for section_key, expected in cases.items():
        assert _infer_category(section_key) == expected, section_key


def test_infer_category_unmatched_falls_back_to_general():
    assert _infer_category("Harvesting Schedule") == "general"


def test_infer_category_is_case_insensitive():
    assert _infer_category("PEST CONTROL") == "pest"


# ---------------------------------------------------------------------------
# _split_large_text
# ---------------------------------------------------------------------------

def test_split_large_text_returns_whole_text_when_under_limit():
    text = "short text"
    assert _split_large_text(text, max_chunk_chars=100) == [text]


def test_split_large_text_never_exceeds_max_chunk_chars():
    text = "\n\n".join(f"paragraph {i} " + "x" * 40 for i in range(10))
    chunks = _split_large_text(text, max_chunk_chars=50)
    assert all(len(c) <= 50 for c in chunks), [len(c) for c in chunks]


def test_split_large_text_hard_splits_a_single_oversized_paragraph():
    text = "x" * 130
    chunks = _split_large_text(text, max_chunk_chars=50)
    assert all(len(c) <= 50 for c in chunks)
    assert "".join(chunks) == text


# ---------------------------------------------------------------------------
# Chunk ID determinism (Task 1.1 / ARCHITECTURE.md Section 4): point IDs must
# be a deterministic hash of chunk content, not uuid4(), so gold-label chunk
# IDs in eval sets survive re-seeding the collection.
# ---------------------------------------------------------------------------

def test_point_id_generation_uses_deterministic_uuid5_not_uuid4():
    """
    Static guard against regressing to uuid.uuid4() (random IDs). Finds the
    `point_id = uuid.uuid5(...)` assignment in build_database() and checks
    it's a uuid5 call seeded with a fixed namespace, not uuid4.
    """
    found = False
    for node in ast.walk(_TREE):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "uuid5"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "uuid"
        ):
            found = True
            break
    assert found, (
        "expected a uuid.uuid5(...) call generating point IDs; if this now "
        "fails, check whether it was reverted to uuid.uuid4() (random IDs), "
        "which breaks gold-label survival across re-seeds"
    )

    uuid4_calls = [
        node for node in ast.walk(_TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "uuid4"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "uuid"
    ]
    assert not uuid4_calls, "vector_db.py must not generate point IDs with uuid.uuid4() (random, not reproducible)"


def test_uuid5_is_deterministic_across_calls():
    """The property build_database() relies on: same content -> same ID,
    every time, with no external state."""
    import uuid
    text = "ಕಾರ್ಬೆಂಡೈಜಿಮ್ 50 ಡಬ್ಲ್ಯೂ.ಪಿ (1 ಗ್ರಾಂ/ಲೀಟರ್)"
    id_a = uuid.uuid5(uuid.NAMESPACE_URL, text)
    id_b = uuid.uuid5(uuid.NAMESPACE_URL, text)
    assert id_a == id_b

    different_text = text + " extra"
    assert uuid.uuid5(uuid.NAMESPACE_URL, different_text) != id_a
