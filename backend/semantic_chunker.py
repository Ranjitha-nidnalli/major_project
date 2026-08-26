
"""
semantic_chunker.py

Semantic chunking for Kannada agricultural text.

Uses the BGE-M3 embedding model to detect natural topic boundaries by:
1. Splitting text into sentences (Kannada danda + period aware)
2. Creating sliding windows of sentences
3. Embedding each window
4. Computing cosine similarity between adjacent windows
5. Finding "valleys" (sharp similarity drops) — these are semantic boundaries
6. Creating chunks at those boundaries

This is algorithmic and structure-agnostic: it works on any text, not just
structured JSON. For a thesis defense, this is a stronger story than
field-aware chunking because it generalizes to unstructured corpora.
"""
import re
import numpy as np
from typing import List, Dict

_SENTENCE_DELIMITERS = re.compile(r'[।.?!]\s+')
_MIN_SENTENCES = 2
_WINDOW_SENTENCES = 3
_SIMILARITY_DROP_THRESHOLD = 0.15
_ABSOLUTE_SIM_THRESHOLD = 0.60


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, preserving Kannada danda (।)."""
    text = re.sub(r'\s+', ' ', text).strip()
    raw = _SENTENCE_DELIMITERS.split(text)
    return [s.strip() for s in raw if s.strip()]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


class SemanticChunker:
    def __init__(self, embed_model, window_sentences: int = _WINDOW_SENTENCES,
                 drop_threshold: float = _SIMILARITY_DROP_THRESHOLD,
                 abs_threshold: float = _ABSOLUTE_SIM_THRESHOLD,
                 max_chunk_chars: int = 1200):
        self.embed_model = embed_model
        self.window_sentences = window_sentences
        self.drop_threshold = drop_threshold
        self.abs_threshold = abs_threshold
        self.max_chunk_chars = max_chunk_chars

    def _embed_window(self, sentences: List[str], start: int, end: int) -> np.ndarray:
        window_text = ' '.join(sentences[start:end])
        output = self.embed_model.encode([window_text], return_dense=True, return_sparse=False)
        return output['dense_vecs'][0]

    def _find_boundaries(self, sentences: List[str]) -> List[int]:
        """Find sentence indices where semantic boundaries occur."""
        n = len(sentences)
        if n <= self.window_sentences * 2:
            return [0, n]

        similarities = []
        for i in range(n - self.window_sentences):
            vec_a = self._embed_window(sentences, i, i + self.window_sentences)
            vec_b = self._embed_window(sentences, i + 1, i + 1 + self.window_sentences)
            sim = cosine_similarity(vec_a, vec_b)
            similarities.append(sim)

        boundaries = [0]

        for i in range(1, len(similarities)):
            prev_sim = similarities[i - 1]
            curr_sim = similarities[i]
            boundary_idx = i + self.window_sentences

            # Condition 1: Sharp drop + low absolute similarity
            drop = prev_sim - curr_sim
            if drop > self.drop_threshold and curr_sim < self.abs_threshold:
                if boundary_idx > boundaries[-1] + _MIN_SENTENCES:
                    boundaries.append(boundary_idx)

            # Condition 2: Very low absolute similarity (emergency split)
            elif curr_sim < 0.40 and boundary_idx > boundaries[-1] + _MIN_SENTENCES:
                boundaries.append(boundary_idx)

        boundaries.append(n)
        return boundaries

    def chunk_sentences(self, sentences: List[str], header: str = "",
                        category: str = "general") -> List[Dict]:
        """Chunk sentences using semantic boundaries."""
        boundaries = self._find_boundaries(sentences)
        chunks = []

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk_sentences = sentences[start:end]
            body = ' '.join(chunk_sentences)

            text = (header + '\n' + body) if header else body

            # Hard ceiling
            if len(text) > self.max_chunk_chars:
                truncated = []
                current_len = len(header) + 1 if header else 0
                for s in chunk_sentences:
                    if current_len + len(s) + 1 > self.max_chunk_chars:
                        break
                    truncated.append(s)
                    current_len += len(s) + 1
                body = ' '.join(truncated)
                text = (header + '\n' + body) if header else body

            if text.strip():
                chunks.append({"text": text.strip(), "category": category})

        return chunks

    def chunk_text(self, text: str, header: str = "",
                   category: str = "general") -> List[Dict]:
        """One-shot semantic chunking of raw text."""
        sentences = split_sentences(text)
        if not sentences:
            return []
        return self.chunk_sentences(sentences, header=header, category=category)


def semantic_chunk_document(header: str, body: str, embed_model,
                            category: str = "general",
                            max_chunk_chars: int = 1200) -> List[Dict]:
    """Drop-in function for vector_db.py."""
    chunker = SemanticChunker(embed_model, max_chunk_chars=max_chunk_chars)
    return chunker.chunk_text(body, header=header, category=category)
