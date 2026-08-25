"""
bm25_retriever.py

Standalone BM25 retriever for the Krishi Mitra ablation harness.
Operates on the same chunked corpus as the Qdrant index (reproduces
vector_db.py chunking logic for identical chunk IDs).

Uses rank_bm25 (pure statistical BM25Okapi) — no neural weights,
just TF-IDF + length normalization computed fresh from the corpus.

Usage:
    from bm25_retriever import BM25Retriever, load_chunks_from_qdrant_upsert
    bm25 = BM25Retriever(load_chunks_from_qdrant_upsert())
    results = bm25.search(query, top_k=10)
    # results: [(chunk_id, bm25_score), ...]

    # RRF fusion with another result list:
    fused = bm25.rrf_fuse(query, dense_results, top_k=5)
"""
import json
import re
import time
import os
import sys
from typing import List, Dict, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("pip install rank_bm25")

# Import chunking logic from vector_db to guarantee identical chunks + IDs
from vector_db import load_and_clean_data, create_overlapping_chunks

# Simple Kannada-aware tokenization for BM25
# We want recall-oriented tokenization: keep Kannada words and ASCII words
_NON_WORD = re.compile(r"[^\w\u0C80-\u0CFF]+")


def tokenize(text: str) -> List[str]:
    """
    Tokenize for BM25: lowercase, split on non-word chars,
    keep Kannada words and ASCII words, drop single chars.
    """
    text = text.lower()
    text = _NON_WORD.sub(" ", text)
    tokens = [t for t in text.split() if len(t) > 1]
    return tokens


class BM25Retriever:
    def __init__(self, chunks: List[Dict]):
        """
        chunks: list of dicts with 'text', 'id', 'category' fields,
                same structure as Qdrant payload.
        """
        self.chunks = chunks
        self.corpus_tokens = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.id_to_idx = {c["id"]: i for i, c in enumerate(chunks)}

    def search(self, query: str, top_k: int = 15) -> List[Tuple[str, float]]:
        """
        Returns: list of (chunk_id, bm25_score) sorted descending.
        Only returns items with score > 0.
        """
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            if scores[idx] > 0:
                results.append((self.chunks[idx]["id"], float(scores[idx])))
        return results

    def rrf_fuse(
        self,
        bm25_results: List[Tuple[str, float]],
        other_results: List[Tuple[str, float]],
        other_weight: float = 1.0,
        k: int = 60,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Reciprocal Rank Fusion between BM25 and another result list.

        Args:
            bm25_results: list of (chunk_id, score) from BM25 search.
            other_results: list of (chunk_id, score) from dense/sparse/hybrid.
            other_weight: weight for the other result list (1.0 = equal).
            k: RRF constant (default 60).
            top_k: number of results to return.
        """
        bm25_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(bm25_results)}
        other_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(other_results)}

        all_ids = set(bm25_ranks.keys()) | set(other_ranks.keys())
        fused = {}
        for cid in all_ids:
            score = 0.0
            if cid in bm25_ranks:
                score += 1.0 / (k + bm25_ranks[cid])
            if cid in other_ranks:
                score += other_weight / (k + other_ranks[cid])
            fused[cid] = score

        sorted_fused = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return sorted_fused[:top_k]


def load_chunks_from_qdrant_upsert(
    json_path: str = "sugarcanemerged3.json",
    max_chars: int = 1000,
    overlap: int = 200,
) -> List[Dict]:
    """
    Reproduce the exact chunking logic from vector_db.py so BM25
    operates on identical chunks with identical UUID5-based IDs.
    """
    raw_docs = load_and_clean_data(json_path)
    chunk_objs = create_overlapping_chunks(raw_docs, max_chars=max_chars, overlap_chars=overlap)
    # vector_db.py generates IDs with uuid5(NAMESPACE_URL, chunk_text)
    import uuid

    return [
        {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, c["text"])),
            "text": c["text"],
            "category": c["category"],
        }
        for c in chunk_objs
    ]


def demo():
    """Quick sanity check — run against a sample query."""
    chunks = load_chunks_from_qdrant_upsert()
    print(f"Loaded {len(chunks)} chunks for BM25 index")
    bm25 = BM25Retriever(chunks)

    queries = [
        "ಕಾರ್ಬೆಂಡೈಜಿಮ್ 50 ಡಬ್ಲ್ಯೂ.ಪಿ",
        "ಕಬ್ಬಿನ ಸೆಟ್ಸ್‌ಗಳಲ್ಲಿ ಅನಾನಸ್ ರೋಗ",
        "ಎಷ್ಟು ಸಾರಜನಕ",
    ]
    for q in queries:
        t0 = time.time()
        results = bm25.search(q, top_k=5)
        latency = time.time() - t0
        print(f"\n🔍 '{q}' ({latency:.3f}s)")
        for cid, score in results[:3]:
            print(f"   {cid[:8]}...  score={score:.3f}")


if __name__ == "__main__":
    demo()
