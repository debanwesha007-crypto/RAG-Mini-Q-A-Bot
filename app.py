"""
RAG Document Q&A — Streamlit app.

Pipeline shown explicitly to the user:
  1. Upload documents (PDF / DOCX / TXT)
  2. Documents are chunked and embedded locally, indexed in FAISS
  3. On a question: retrieve top-k relevant chunks (shown with scores)
  4. Only those retrieved chunks are passed to the local LLM to answer

Run with:
    streamlit run app.py
"""
import sys
import tempfile
from pathlib import Path

# Make sure this script's own folder is on sys.path, so the local 'rag'
# package is always found regardless of the working directory the app
# was launched from (fixes "ModuleNotFoundError: No module named 'rag'"
# when launched via a shortcut, IDE run button, or a synced/odd path).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from rag.loaders import load_document
from rag.chunking import chunk_documents
from rag.vectorstore import VectorStore
from rag.llm import generate_answer, DEFAULT_MODEL

st.set_page_config(page_title="RAG Document Q&A", layout="wide")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore(backend="local")
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

st.title("📚 RAG Document Q&A")
st.caption(
    "Retrieval-augmented Q&A: your question is matched against your documents "
    "first, and only the matched passages are sent to the LLM to generate an answer."
)

# ---------------- Sidebar: settings ----------------
with st.sidebar:
    st.header("Settings")
    embedding_backend = st.selectbox(
        "Embedding backend", options=["local", "openai"], index=0,
        help="'local' = sentence-transformers, free, no key. 'openai' needs OPENAI_API_KEY set.",
    )
    top_k = st.slider("Chunks to retrieve (top-k)", min_value=1, max_value=10, value=4)
    chunk_size = st.slider("Chunk size (characters)", 300, 1500, 800, step=100)
    overlap = st.slider("Chunk overlap (characters)", 0, 400, 150, step=50)
    model_name = st.text_input("Ollama model name", value=DEFAULT_MODEL)
    st.markdown("---")
    st.markdown(
        "**Local LLM required:** install [Ollama](https://ollama.com/download), "
        f"then run `ollama pull {DEFAULT_MODEL}` once in a terminal."
    )

# ---------------- Step 1: Upload & index ----------------
st.header("1. Upload documents")
uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT files",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

if st.button("Build index from uploaded documents", type="primary"):
    if not uploaded_files:
        st.warning("Upload at least one document first.")
    else:
        with st.spinner("Loading and chunking documents..."):
            all_raw_docs = []
            for uf in uploaded_files:
                suffix = Path(uf.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.getbuffer())
                    tmp_path = tmp.name
                raw_docs = load_document(tmp_path)
                # Re-tag with the original filename (tempfile has a random name)
                for d in raw_docs:
                    d.source = uf.name
                all_raw_docs.extend(raw_docs)

            chunks = chunk_documents(all_raw_docs, chunk_size=chunk_size, overlap=overlap)

        with st.spinner(f"Embedding {len(chunks)} chunks..."):
            st.session_state.vector_store = VectorStore(backend=embedding_backend)
            st.session_state.vector_store.build(chunks)
            st.session_state.indexed_files = [uf.name for uf in uploaded_files]

        st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} document(s).")

if st.session_state.indexed_files:
    st.info("Currently indexed: " + ", ".join(st.session_state.indexed_files))

st.markdown("---")

# ---------------- Step 2: Ask ----------------
st.header("2. Ask a question")
question = st.text_input("Your question")
ask_clicked = st.button("Get answer")

if ask_clicked:
    if not st.session_state.vector_store.is_ready():
        st.warning("Build an index from your documents first (Step 1).")
    elif not question.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Retrieving relevant passages..."):
            results = st.session_state.vector_store.search(question, top_k=top_k)

        st.subheader("🔍 Retrieved passages")
        st.caption("These are the actual chunks retrieved from your documents, ranked by similarity.")
        context_blocks = []
        for i, r in enumerate(results, start=1):
            tag = f"[{r.chunk.source} — {r.chunk.location}]"
            context_blocks.append(f"{tag}\n{r.chunk.text}")
            with st.expander(f"{i}. {tag}  ·  similarity {r.score:.3f}"):
                st.write(r.chunk.text)

        st.subheader("🧠 Generated answer")
        with st.spinner(f"Asking local model ({model_name})..."):
            answer = generate_answer(question, context_blocks, model=model_name)
        st.write(answer)
