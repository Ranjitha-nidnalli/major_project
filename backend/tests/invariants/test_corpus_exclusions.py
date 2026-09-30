"""
corpus_exclusions.apply_exclusions and the committed corpus_exclusions.json.
Matching the listed IDs against the real corpus happens at index time
(apply_exclusions raises on any ID that matches no chunk), since chunking
the corpus here would mean importing vector_db and its models.
"""
import csv
import json
import os
import uuid

import pytest

from corpus_exclusions import apply_exclusions, EXCLUSIONS_FILE

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _chunk(text):
    return {"text": text, "category": "pest", "topic": "t"}


def _id(text):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, text))


def _exclusions(tmp_path, ids):
    p = tmp_path / "ex.json"
    p.write_text(json.dumps({"excluded": [{"chunk_id": i} for i in ids]}), encoding="utf-8")
    return str(p)


def test_withholds_listed_chunks_and_keeps_order(tmp_path):
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    kept = apply_exclusions(chunks, _exclusions(tmp_path, [_id("b")]))
    assert [c["text"] for c in kept] == ["a", "c"]


def test_never_edits_kept_chunks(tmp_path):
    chunks = [_chunk("a"), _chunk("b")]
    kept = apply_exclusions(chunks, _exclusions(tmp_path, [_id("b")]))
    assert kept[0] is chunks[0]


def test_stale_exclusion_fails_loudly(tmp_path):
    with pytest.raises(ValueError):
        apply_exclusions([_chunk("a")], _exclusions(tmp_path, [_id("not in corpus")]))


def test_no_file_means_no_exclusions(tmp_path):
    chunks = [_chunk("a")]
    assert apply_exclusions(chunks, str(tmp_path / "missing.json")) == chunks


def test_committed_list_matches_review_remove_rows():
    """Every row the review marked 'remove' is excluded, and nothing else."""
    with open(EXCLUSIONS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    with open(os.path.join(BACKEND_DIR, data["review_file"].removeprefix("backend/")), encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    keep_col = next(c for c in rows[0] if c.startswith("expert_keep"))
    removed = {r["chunk_id"] for r in rows if r[keep_col].strip().lower() == "remove"}
    assert {e["chunk_id"] for e in data["excluded"]} == removed
    assert data["status"] == "pending_expert_confirmation"
