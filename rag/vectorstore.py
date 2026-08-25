"""
A minimal in-memory vector store: chunks + their embeddings kept as a plain
numpy array, ranked by cosine similarity at query time. No FAISS/Chroma/etc —
at the scale of a handful of documents this is fast enough and, more
importantly, keeps the retrieval math fully visible rather than hidden
inside an index library.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from .chunking import Chunk
from .embeddings import embed_texts, embed_query


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity, range [-1, 1], higher = more relevant


class VectorStore:
    def __init__(self, backend: Optional[str] = None):
        self.backend = backend
        self.chunks: List[Chunk] = []
        self.vectors: Optional[np.ndarray] = None  # shape (n_chunks, dim), unit-normalized

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        if not chunks:
            self.vectors = None
            return
        self.vectors = embed_texts([c.text for c in chunks], backend=self.backend)

    def is_ready(self) -> bool:
        return self.vectors is not None and len(self.chunks) > 0

    def search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        """
        Cosine similarity search: since both the stored vectors and the
        query vector are unit-normalized, cosine similarity is just their
        dot product. We compute it against every stored vector (a single
        matrix-vector multiply) and take the top_k highest-scoring chunks.
        """
        if not self.is_ready():
            return []

        query_vec = embed_query(query, backend=self.backend)
        similarities = self.vectors @ query_vec  # (n_chunks,) cosine similarities

        k = min(top_k, len(self.chunks))
        top_indices = np.argsort(-similarities)[:k]

        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(similarities[i]))
            for i in top_indices
        ]
