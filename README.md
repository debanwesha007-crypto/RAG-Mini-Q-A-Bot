# RAG Mini Q&A Bot — NimbusNote Docs

A retrieval-augmented Q&A tool built for the MLSA-SRM recruit task. Given the
three NimbusNote docs (`sample_docs/`), it retrieves the passages most
relevant to a question **before** generating an answer, and shows you that
retrieval step — it isn't a thin wrapper that just forwards your question to
an LLM.

## Pipeline

```
sample_docs/*.md
      │
      ▼
Load & chunk text ──► Embed each chunk (sentence-transformers, local & free)
      │                        │
      │                        ▼
      │              In-memory matrix of unit-normalized vectors
      │                        │
Your question ──► Embed question ──► cosine similarity vs every chunk ──► top-k
                                                                            │
                                                                            ▼
                                                     Only those top-k chunks
                                                     + question go to the LLM
                                                                            │
                                                                            ▼
                                                                Final answer
```

Retrieval is plain numpy: chunks and their embeddings are unit vectors kept
in a matrix, so cosine similarity is a single matrix–vector dot product
(`rag/vectorstore.py`). No FAISS/Chroma — at this scale (3 short docs) that
would be overhead, not benefit, and this way the ranking math is fully
visible rather than hidden inside an index library.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires **Python 3.10+**.

### Embeddings — two backends, pick one

- **Local (default, free, no key)** — `sentence-transformers`
  (`all-MiniLM-L6-v2`), downloaded once from HuggingFace on first run and
  cached locally after that.
- **OpenAI** — set `EMBEDDING_BACKEND=openai` and `OPENAI_API_KEY=<your key>`
  in your environment to use `text-embedding-3-small` instead.

### Answer generation — local LLM via Ollama

1. Install Ollama: https://ollama.com/download
2. Pull a small model once: `ollama pull llama3.2:3b`
3. Ollama runs a local server automatically (`http://localhost:11434`) — no
   API key, nothing leaves your machine.

## Running it

**CLI (fastest way to try it / evaluate it):**

```bash
python main.py                                  # indexes ./sample_docs, interactive prompt
python main.py --question "Is there a student discount?"
```

**Web UI (Streamlit):**

```bash
streamlit run app.py
```

Upload the three files from `sample_docs/` (or any other PDF/DOCX/TXT set),
build the index, then ask a question. The retrieved passages are shown with
their similarity scores and source/location tags before the generated
answer.

## Example questions to try against the NimbusNote docs

- "How much is the Pro plan and what do I get for it?"
- "Is there a student discount, and does it apply to Team too?"
- "Why would two versions of the same note show up?"
- "My image upload isn't working, what should I check?"
- "What happens to my data if I downgrade from Pro to Free?"

## Project structure

```
rag-qa-tool/
├── app.py                  # Streamlit UI
├── main.py                 # CLI entry point
├── sample_docs/            # the 3 NimbusNote starter docs
├── rag/
│   ├── loaders.py           # PDF / DOCX / TXT(/MD) → text
│   ├── chunking.py          # text → overlapping chunks
│   ├── embeddings.py        # local (sentence-transformers) or OpenAI embeddings
│   ├── vectorstore.py       # in-memory cosine-similarity search (numpy)
│   └── llm.py                # local LLM call via Ollama
├── requirements.txt
└── README.md
```

## Design notes

- **Why chunk at all?** Embedding a whole doc as one vector blurs together
  everything in it (pricing details and troubleshooting steps end up in the
  same vector); chunking lets the search return the specific paragraph that
  answers the question.
- **Why show retrieved passages in the UI?** It's the whole point of the
  exercise — you can see exactly which source and location backed the
  answer, rather than trusting a black box.
- **Why local embeddings by default?** Free, no API key, and fast enough at
  this document scale; OpenAI's embeddings API is there as a drop-in swap if
  preferred.

## Possible extensions

- Add a re-ranking step (cross-encoder) after the initial cosine-similarity
  retrieval for higher precision on larger document sets.
- Add hybrid search (BM25 + embeddings) for exact keyword/jargon matching.
- Persist the vector store to disk so it doesn't rebuild every session.
