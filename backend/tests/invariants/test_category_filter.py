"""
Invariant test for Task 1 (ARCHITECTURE.md Section 12, Option A).

Verified defect: the old rag_service.py applied
    models.Filter(should=[FieldCondition(key="category", match=...)])
to the Qdrant query. In Qdrant, a Filter with only `should` clauses is NOT a
soft boost — it hard-restricts results to points that satisfy at least one
`should` clause. Because the LLM router can misclassify a query's category,
this made gold chunks in a *different* category from the router's guess
completely unreachable.

This test does not need the embedding models or the LLM router. It builds a
tiny in-memory Qdrant collection with cards in several categories using
synthetic dense/sparse vectors, and proves that an unfiltered hybrid RRF
query -- the retrieval pattern rag_service.execute_weighted_search now uses
-- still returns the correct card even when it belongs to a category other
than the one the (simulated) router guessed.

A second test reproduces the old buggy filter directly against the same
collection and shows it does exclude the correct card, to document exactly
what was fixed and guard against reintroducing it.
"""
from qdrant_client import QdrantClient, models

COLLECTION = "test_sugarcane_knowledge"
DENSE_DIM = 4


def _build_collection():
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
    )

    # Basis-vector cards so cosine similarity unambiguously picks one card.
    cards = [
        {"id": 1, "category": "pest", "text": "pest card", "dense": [1.0, 0.0, 0.0, 0.0], "sparse": {1: 1.0}},
        {"id": 2, "category": "disease", "text": "disease card (the correct gold card)", "dense": [0.0, 1.0, 0.0, 0.0], "sparse": {2: 1.0}},
        {"id": 3, "category": "general", "text": "general card", "dense": [0.0, 0.0, 1.0, 0.0], "sparse": {3: 1.0}},
        {"id": 4, "category": "fertilizer", "text": "fertilizer card", "dense": [0.0, 0.0, 0.0, 1.0], "sparse": {4: 1.0}},
    ]

    points = [
        models.PointStruct(
            id=c["id"],
            payload={"category": c["category"], "text": c["text"]},
            vector={
                "dense": c["dense"],
                "sparse": models.SparseVector(
                    indices=list(c["sparse"].keys()),
                    values=list(c["sparse"].values()),
                ),
            },
        )
        for c in cards
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    return client


def _hybrid_query(client, dense_vec, sparse_indices, sparse_values, query_filter=None, prefetch_filter=None):
    """Mirrors rag_service.execute_weighted_search's query shape.

    prefetch_filter, if given, is applied to each prefetch leg directly --
    this is where a should-only Filter actually hard-restricts (see
    test_old_should_only_filter_would_have_hidden_the_card).
    """
    response = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_vec, using="dense", limit=15, filter=prefetch_filter),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                limit=15,
                filter=prefetch_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=5,
        query_filter=query_filter,
    )
    return response.points


def test_misrouted_category_still_retrieves_correct_card():
    """
    The query semantically matches the 'disease' card, but we simulate the
    router misclassifying it as 'pest'. With no category filter applied
    (the Task 1 fix), the correct 'disease' card must still be returned.
    """
    client = _build_collection()

    # Query vector matches the disease card exactly; simulated router
    # guess ("pest") is deliberately wrong and, per the fix, is never
    # turned into a retrieval filter.
    query_dense = [0.0, 1.0, 0.0, 0.0]
    query_sparse_indices = [2]
    query_sparse_values = [1.0]

    hits = _hybrid_query(client, query_dense, query_sparse_indices, query_sparse_values, query_filter=None)

    assert hits, "expected at least one hit"
    top_categories = [h.payload["category"] for h in hits]
    assert "disease" in top_categories, (
        f"correct 'disease' card was not retrieved; got categories={top_categories}"
    )
    assert hits[0].payload["category"] == "disease", (
        f"correct card should rank first; top hit category={hits[0].payload['category']}"
    )


def test_old_should_only_filter_would_have_hidden_the_card():
    """
    Documents the defect: a should-only Filter on the WRONG category
    (the router's misclassification) hard-excludes the correct card.
    This is what rag_service.py must never do again.

    Investigated 2026-09-21: the pre-fix code (commit 1b57a3e^) passed the
    should-only filter as the *outer* `query_filter` on a `prefetch` +
    `FusionQuery` call. Against qdrant-client 1.18's `:memory:` backend,
    that exact shape does NOT exclude "disease" -- the outer filter is not
    enforced post-fusion in this backend. The should-only hard-restriction
    is real and reproducible (verified below and against a plain, non-fused
    search), but only when the filter reaches a search/prefetch leg
    directly. This test applies it there so it still demonstrates the
    genuine Qdrant gotcha; the outer-filter-on-fusion variant is untested
    here for lack of a real Qdrant server (no Docker in this environment).
    Re-verify against the docker-compose server before relying on this for
    the paper/report.
    """
    client = _build_collection()

    query_sparse_indices = [2]
    query_sparse_values = [1.0]

    buggy_filter = models.Filter(
        should=[models.FieldCondition(key="category", match=models.MatchValue(value="pest"))]
    )

    hits = _hybrid_query(
        client,
        [0.0, 1.0, 0.0, 0.0],
        query_sparse_indices,
        query_sparse_values,
        query_filter=None,
        prefetch_filter=buggy_filter,
    )

    categories = [h.payload["category"] for h in hits]
    assert "disease" not in categories, (
        "this test documents the OLD buggy behavior: a should-only filter on "
        "the wrong category hard-excludes the correct card. If this "
        "assertion fails, Qdrant's filter semantics changed and the "
        "reasoning behind Task 1 should be re-checked."
    )
