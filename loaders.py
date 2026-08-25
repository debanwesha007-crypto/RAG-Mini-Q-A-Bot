"""
Document loaders: convert PDF / DOCX / TXT files into plain text,
tagged with source filename and page/paragraph info where available.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class RawDoc:
    source: str          # filename
    location: str        # e.g. "page 3" or "paragraph 12" or "whole file"
    text: str


def load_pdf(path: str) -> List[RawDoc]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    name = Path(path).name
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            docs.append(RawDoc(source=name, location=f"page {i}", text=text))
    return docs


def load_docx(path: str) -> List[RawDoc]:
    import docx

    d = docx.Document(path)
    name = Path(path).name
    docs = []
    buffer = []
    para_start = 1
    for i, para in enumerate(d.paragraphs, start=1):
        t = para.text.strip()
        if t:
            buffer.append(t)
        # Flush every ~10 non-empty paragraphs to keep location tags meaningful
        if len(buffer) >= 10:
            docs.append(
                RawDoc(
                    source=name,
                    location=f"paragraphs {para_start}-{i}",
                    text="\n".join(buffer),
                )
            )
            buffer = []
            para_start = i + 1
    if buffer:
        docs.append(
            RawDoc(
                source=name,
                location=f"paragraphs {para_start}-end",
                text="\n".join(buffer),
            )
        )
    return docs


def load_txt(path: str) -> List[RawDoc]:
    name = Path(path).name
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [RawDoc(source=name, location="whole file", text=text)]


def load_document(path: str) -> List[RawDoc]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    if ext in (".txt", ".md"):
        return load_txt(path)
    raise ValueError(f"Unsupported file type: {ext}")
