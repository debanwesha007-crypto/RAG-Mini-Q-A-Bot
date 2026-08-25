"""
Embedding backend, pluggable between:
  - "local"  : sentence-transformers (all-MiniLM-L6-v2), runs on-device, free, no API key
  - "openai" : OpenAI's embeddings API (text-embedding-3-small), needs OPENAI_API_KEY

Select via the EMBEDDING_BACKEND env var ("local" by default) or by passing
backend= explicitly to embed_texts / embed_query.
"""
from __future__ import annotations
import os
from functools import lru_cache
from typing import List
import numpy as np

LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_MODEL_NAME = "text-embedding-3-small"

DEFAULT_BACKEND = os.environ.get("EMBEDDING_BACKEND", "local")


@lru_cache(maxsize=1)
def _get_local_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(LOCAL_MODEL_NAME)


def _embed_local(texts: List[str]) -> np.ndarray:
    model = _get_local_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit vectors -> dot product == cosine similarity
        show_progress_bar=False,
    )
    return vectors.astype("float32")


def _embed_openai(texts: List[str]) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    resp = client.embeddings.create(model=OPENAI_MODEL_NAME, input=texts)
    vectors = np.array([d.embedding for d in resp.data], dtype="float32")
    # normalize so cosine similarity == dot product, same convention as the local path
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def embed_texts(texts: List[str], backend: str | None = None) -> np.ndarray:
    backend = backend or DEFAULT_BACKEND
    if backend == "local":
        return _embed_local(texts)
    if backend == "openai":
        return _embed_openai(texts)
    raise ValueError(f"Unknown embedding backend: {backend}")


def embed_query(text: str, backend: str | None = None) -> np.ndarray:
    return embed_texts([text], backend=backend)[0]
