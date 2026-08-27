"""
bm25_retriever.py

Standalone BM25 retriever for the Krishi Mitra ablation harness.

Uses the exact same adaptive structure-aware chunking logic as
vector_db.py so BM25 operates on the same corpus and generates the
same UUID5-based chunk IDs as the Qdrant index.

Uses rank_bm25 (pure statistical BM25Okapi) — no neural weights,
just TF-IDF + length normalization computed from the chunked corpus.

Usage:
    from bm25_retriever import BM25Retriever, load_chunks_from_qdrant_upsert

    chunks = load_chunks_from_qdrant_upsert()
    bm25 = BM25Retriever(chunks)

    results = bm25.search(query, top_k=10)

    # results:
    # [(chunk_id, bm25_score), ...]

    # RRF fusion with another result list:
    fused = bm25.rrf_fuse(
        bm25_results,
        dense_results,
        top_k=5
    )
"""

import re
import time
import os
import sys
import uuid

from typing import List, Dict, Tuple

import numpy as np


# Allow importing vector_db.py from backend/
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError(
        "rank_bm25 is not installed. Run: pip install rank_bm25"
    )


# Import the CURRENT adaptive structure-aware chunking pipeline.
# This is important: BM25 must operate on the exact same chunks
# as Qdrant for the ablation comparison to be valid.
from vector_db import load_and_chunk_data


# ============================================================
# TOKENIZATION
# ============================================================

# Kannada Unicode range: \u0C80-\u0CFF
# Keep Kannada words, ASCII words and numbers.
_NON_WORD = re.compile(
    r"[^\w\u0C80-\u0CFF]+"
)


def tokenize(text: str) -> List[str]:
    """
    Kannada-aware tokenizer for BM25.

    Steps:
    1. Convert to lowercase.
    2. Replace punctuation and non-word characters with spaces.
    3. Split into tokens.
    4. Remove single-character tokens.

    This is recall-oriented so Kannada and English terms
    can both be matched.
    """

    text = text.lower()

    text = _NON_WORD.sub(
        " ",
        text
    )

    tokens = [
        token
        for token in text.split()
        if len(token) > 1
    ]

    return tokens


# ============================================================
# BM25 RETRIEVER
# ============================================================

class BM25Retriever:

    def __init__(
        self,
        chunks: List[Dict]
    ):
        """
        Build a BM25 index over the supplied chunks.

        Expected chunk format:

        {
            "id": "...",
            "text": "...",
            "category": "..."
        }
        """

        self.chunks = chunks

        # Tokenize every chunk once during initialization.
        self.corpus_tokens = [
            tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            self.corpus_tokens
        )

        # Useful for looking up chunks by ID.
        self.id_to_idx = {
            chunk["id"]: index
            for index, chunk in enumerate(chunks)
        }


    def search(
        self,
        query: str,
        top_k: int = 15
    ) -> List[Tuple[str, float]]:
        """
        Search the BM25 index.

        Args:
            query:
                User query.

            top_k:
                Maximum number of results to return.

        Returns:
            List of:

            [
                (chunk_id, bm25_score),
                ...
            ]

        Results are sorted in descending score order.

        Only results with score > 0 are returned.
        """

        query_tokens = tokenize(
            query
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        top_indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for index in top_indices:

            score = scores[index]

            if score > 0:

                chunk_id = self.chunks[index]["id"]

                results.append(
                    (
                        chunk_id,
                        float(score)
                    )
                )

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
        Reciprocal Rank Fusion between BM25 and another
        retrieval result list.

        Args:
            bm25_results:
                List of (chunk_id, score) from BM25.

            other_results:
                List of (chunk_id, score) from another
                retriever such as dense or hybrid.

            other_weight:
                Weight applied to the other retriever.

                1.0 = equal weighting.

            k:
                RRF constant.

            top_k:
                Number of fused results to return.

        Returns:
            List of:

            [
                (chunk_id, rrf_score),
                ...
            ]
        """

        bm25_ranks = {
            chunk_id: rank + 1
            for rank, (
                chunk_id,
                _
            ) in enumerate(bm25_results)
        }

        other_ranks = {
            chunk_id: rank + 1
            for rank, (
                chunk_id,
                _
            ) in enumerate(other_results)
        }

        all_ids = (
            set(bm25_ranks.keys())
            |
            set(other_ranks.keys())
        )

        fused = {}

        for chunk_id in all_ids:

            score = 0.0

            if chunk_id in bm25_ranks:

                score += (
                    1.0
                    /
                    (
                        k
                        +
                        bm25_ranks[chunk_id]
                    )
                )

            if chunk_id in other_ranks:

                score += (
                    other_weight
                    /
                    (
                        k
                        +
                        other_ranks[chunk_id]
                    )
                )

            fused[chunk_id] = score

        sorted_fused = sorted(
            fused.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return sorted_fused[:top_k]


# ============================================================
# LOAD IDENTICAL CHUNKS
# ============================================================

def load_chunks_from_qdrant_upsert(
    json_path: str = "sugarcanemerged3.json",
) -> List[Dict]:
    """
    Load the corpus using the exact same adaptive
    structure-aware chunking logic as vector_db.py.

    Chunk IDs are generated using:

        uuid.uuid5(
            uuid.NAMESPACE_URL,
            chunk_text
        )

    This must match vector_db.py exactly.

    Returns:
        List of dictionaries:

        {
            "id": "...",
            "text": "...",
            "category": "..."
        }
    """

    print(
        "Loading chunks using adaptive structure-aware chunking..."
    )

    chunk_objs = load_and_chunk_data(
        json_path
    )

    chunks = []

    for chunk in chunk_objs:

        chunk_text = chunk["text"]

        chunk_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk_text
            )
        )

        chunks.append(
            {
                "id": chunk_id,
                "text": chunk_text,
                "category": chunk["category"],
            }
        )

    return chunks


# ============================================================
# DEMO / SANITY CHECK
# ============================================================

def demo():
    """
    Quick sanity check.

    Run:

        python eval/bm25_retriever.py

    The chunk count should match the chunk count reported
    when building Qdrant.
    """

    print(
        "\nLoading BM25 corpus..."
    )

    chunks = load_chunks_from_qdrant_upsert()

    print(
        f"Loaded {len(chunks)} chunks for BM25 index"
    )

    bm25 = BM25Retriever(
        chunks
    )

    queries = [

        "ಕಾರ್ಬೆಂಡೈಜಿಮ್ 50 ಡಬ್ಲ್ಯೂ.ಪಿ",

        "ಕಬ್ಬಿನ ಸೆಟ್ಸ್‌ಗಳಲ್ಲಿ ಅನಾನಸ್ ರೋಗ",

        "ಎಷ್ಟು ಸಾರಜನಕ",

    ]

    for query in queries:

        start_time = time.time()

        results = bm25.search(
            query,
            top_k=5
        )

        latency = (
            time.time()
            -
            start_time
        )

        print(
            f"\n🔍 '{query}' "
            f"({latency:.3f}s)"
        )

        if not results:

            print(
                "   No BM25 matches found."
            )

            continue

        for chunk_id, score in results[:3]:

            print(
                f"   {chunk_id[:8]}... "
                f"score={score:.3f}"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    demo()
