"""
Thin wrapper around the persisted Chroma index.

Kept as its own module (same design principle as RivalIQ's rag/retriever.py)
so the rest of the app depends on a stable `get_retriever()` / `retrieve()`
interface, not on Chroma-specific details - if you ever swap Chroma for
FAISS or Pinecone, only this file changes.
"""

from __future__ import annotations
from pathlib import Path
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = ROOT / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings = None
_vectorstore = None


def _get_vectorstore() -> Chroma:
    """Lazily loads the embedding model and the persisted index (both are
    reused across calls rather than reloaded every time - reloading the
    embedding model per query would be needlessly slow)."""
    global _embeddings, _vectorstore
    if _vectorstore is None:
        if not PERSIST_DIR.exists():
            raise RuntimeError(
                f"No index found at {PERSIST_DIR}. Run `python ingest/build_index.py` first."
            )
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = Chroma(
            persist_directory=str(PERSIST_DIR),
            embedding_function=_embeddings,
            collection_name="acme_policies",
        )
    return _vectorstore


def retrieve(query: str, k: int = 4) -> List[Document]:
    """Top-k similarity search over the policy corpus. Returns LangChain
    Document objects, each with `.page_content` and `.metadata` (source file)."""
    vectorstore = _get_vectorstore()
    return vectorstore.similarity_search(query, k=k)


def retrieve_with_scores(query: str, k: int = 4):
    """Same as retrieve(), but also returns the similarity distance for each
    result - useful for the evaluation harness and for a relevance threshold."""
    vectorstore = _get_vectorstore()
    return vectorstore.similarity_search_with_relevance_scores(query, k=k)


def format_context(docs: List[Document]) -> str:
    """Turns retrieved chunks into the context block injected into the prompt.
    Includes the source filename so the LLM (and the user) can see where each
    fact came from - basic traceability, same spirit as citing sources."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = Path(doc.metadata.get("source", "unknown")).name
        blocks.append(f"[Source {i}: {source}]\n{doc.page_content}")
    return "\n\n".join(blocks)
