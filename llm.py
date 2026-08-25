"""
Local LLM wrapper. Uses Ollama's local HTTP API (http://localhost:11434),
which runs a model entirely on your machine — no API key, no cloud call.

Prerequisite (done once, outside this app):
    1. Install Ollama: https://ollama.com/download
    2. Pull a small model:  ollama pull llama3.2:3b
    3. Ollama runs a local server automatically after install.
"""
from __future__ import annotations
from typing import List
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions using ONLY the "
    "provided context passages from the user's documents. "
    "If the answer is not contained in the context, say so plainly instead "
    "of guessing. When you use a passage, refer to it by its [source] tag."
)


def build_prompt(question: str, context_blocks: List[str]) -> str:
    context_str = "\n\n".join(context_blocks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- CONTEXT PASSAGES ---\n{context_str}\n"
        f"--- END CONTEXT ---\n\n"
        f"Question: {question}\n"
        f"Answer (use only the context above; cite [source] tags where relevant):"
    )


def generate_answer(
    question: str,
    context_blocks: List[str],
    model: str = DEFAULT_MODEL,
    timeout: int = 120,
) -> str:
    prompt = build_prompt(question, context_blocks)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return (
            "⚠️ Could not reach the local LLM (Ollama). Make sure Ollama is "
            f"installed and running, and that you've pulled a model, e.g.\n\n"
            f"    ollama pull {model}\n\n"
            "Then try again."
        )
    except Exception as e:
        return f"⚠️ Error generating answer: {e}"
