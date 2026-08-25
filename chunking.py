"""
Split loaded documents into overlapping chunks suitable for embedding.
Chunking is done by characters with sentence-boundary snapping, which is
simple, dependency-free, and works reasonably well across PDF/DOCX/TXT text.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import re

from .loaders import RawDoc

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    id: str
    source: str
    location: str
    text: str


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return SENTENCE_SPLIT_RE.split(text)


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[str]:
    """Greedily pack sentences into ~chunk_size-character chunks with overlap."""
    sentences = _split_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > chunk_size and current:
            chunk_str = " ".join(current)
            chunks.append(chunk_str)
            # build overlap: keep trailing sentences that fit within `overlap` chars
            overlap_sents = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_len += len(s)
            current = overlap_sents
            current_len = overlap_len

        current.append(sent)
        current_len += len(sent)

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_documents(
    raw_docs: List[RawDoc],
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    counter = 0
    for doc in raw_docs:
        pieces = chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)
        for piece in pieces:
            counter += 1
            chunks.append(
                Chunk(
                    id=f"chunk_{counter}",
                    source=doc.source,
                    location=doc.location,
                    text=piece,
                )
            )
    return chunks
