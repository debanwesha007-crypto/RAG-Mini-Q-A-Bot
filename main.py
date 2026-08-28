"""
Command-line interface for the RAG Q&A tool — useful for quick testing
without launching the Streamlit UI.

Usage:
    python main.py                        # loads ./sample_docs, interactive prompt
    python main.py --docs path/to/folder  # index a different folder
    python main.py --question "..."       # ask one question and exit
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.loaders import load_document
from rag.chunking import chunk_documents
from rag.vectorstore import VectorStore
from rag.llm import generate_answer, DEFAULT_MODEL

SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md"}


def build_index(docs_folder: str, chunk_size: int = 800, overlap: int = 150) -> VectorStore:
    folder = Path(docs_folder)
    if not folder.is_dir():
        print(f"Error: '{docs_folder}' is not a folder.")
        sys.exit(1)

    files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in SUPPORTED_EXTS]
    if not files:
        print(f"No supported documents (.pdf/.docx/.txt/.md) found in '{docs_folder}'.")
        sys.exit(1)

    print(f"Loading {len(files)} document(s) from {docs_folder}...")
    raw_docs = []
    for f in files:
        raw_docs.extend(load_document(str(f)))

    chunks = chunk_documents(raw_docs, chunk_size=chunk_size, overlap=overlap)
    print(f"Split into {len(chunks)} chunks. Embedding locally...")

    store = VectorStore()
    store.build(chunks)
    print("Index ready.\n")
    return store


def ask(store: VectorStore, question: str, top_k: int, model: str) -> None:
    results = store.search(question, top_k=top_k)

    print("Retrieved passages:")
    context_blocks = []
    for i, r in enumerate(results, start=1):
        tag = f"[{r.chunk.source} — {r.chunk.location}]"
        context_blocks.append(f"{tag}\n{r.chunk.text}")
        print(f"  {i}. {tag}  (similarity {r.score:.3f})")
        preview = r.chunk.text[:100].replace("\n", " ")
        print(f"     {preview}...")
    print()

    answer = generate_answer(question, context_blocks, model=model)
    print("Answer:")
    print(answer)
    print()


def main():
    parser = argparse.ArgumentParser(description="RAG Document Q&A (CLI)")
    parser.add_argument("--docs", default="sample_docs", help="Folder of documents to index")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--question", default=None, help="Ask one question and exit")
    args = parser.parse_args()

    store = build_index(args.docs)

    if args.question:
        ask(store, args.question, args.top_k, args.model)
        return

    print("Type a question (or 'quit' to exit).\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break
        ask(store, question, args.top_k, args.model)


if __name__ == "__main__":
    main()
