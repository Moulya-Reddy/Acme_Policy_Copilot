"""
Builds the local vector index from the policy documents in data/.

Everything here is FREE and runs entirely on your machine:
- Document loading: plain-text loaders, no paid parsing API.
- Chunking: LangChain's RecursiveCharacterTextSplitter (local, no API call).
- Embeddings: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`, downloaded once
  (~90MB) from Hugging Face and then run locally on CPU. No embedding API key,
  no per-call cost, works fine on a Mac (Intel or Apple Silicon) with no GPU.
- Vector store: Chroma, an open-source, local, file-backed vector database.
  No server to run, no cloud account, no cost. Data is persisted to ./chroma_db/.

Run this once (or whenever data/ changes):
    python ingest/build_index.py
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = ROOT / "data"
PERSIST_DIR = ROOT / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # free, local, ~90MB, CPU-friendly

# Chunking parameters.
# 800 chars (~150-200 tokens) is a reasonable chunk size for short policy paragraphs;
# 120 chars of overlap protects against a rule and its exception being split apart
# across two chunks (see the README's chunking notes for why this matters).
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def load_documents():
    """Loads every .md/.txt file under data/ as a LangChain Document."""
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    if not docs:
        raise RuntimeError(f"No .md files found in {DATA_DIR}. Add policy documents first.")
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        # Splitting on markdown headers first keeps each policy section's rules
        # together in one chunk as long as possible, before falling back to
        # paragraph/sentence/word boundaries for anything still too long.
    )
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} document(s) -> split into {len(chunks)} chunks.")
    return chunks


def build_index():
    docs = load_documents()
    chunks = chunk_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Chroma persists to disk at PERSIST_DIR - rerunning this script rebuilds the
    # index from scratch (fine for a POC-scale corpus of a few documents).
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
        collection_name="acme_policies",
    )
    print(f"Vector index built and persisted to {PERSIST_DIR}")
    return vectorstore


if __name__ == "__main__":
    build_index()
