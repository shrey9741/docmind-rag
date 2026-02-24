"""
core/vectorstore.py
===================
Handles all vector store operations:
- Loading the HuggingFace embedding model
- Splitting text into chunks
- Building and saving a FAISS index
- Loading a saved FAISS index from disk
- Loading store metadata
"""

import os
import pickle

import streamlit as st

from config.settings import (
    VECTOR_DB_PATH,
    EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    """
    Load and cache the HuggingFace sentence embedding model.
    Cached with @st.cache_resource so it loads only once per session.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(texts: list, metadatas: list):
    """
    Full indexing pipeline:
    1. Split each document into overlapping chunks
    2. Embed all chunks using the sentence transformer
    3. Store embeddings in a FAISS index
    4. Save the index to disk for persistence

    Args:
        texts     : list of extracted document texts
        metadatas : list of dicts with source filename per document

    Returns:
        FAISS vector store object, or None if no chunks were created
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    # ── Step 1: Split into chunks ─────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks, chunk_meta = [], []
    for text, meta in zip(texts, metadatas):
        splits = splitter.split_text(text)
        chunks.extend(splits)
        chunk_meta.extend([meta] * len(splits))

    if not chunks:
        return None

    # ── Step 2 & 3: Embed and build FAISS index ───────────────────────────────
    embeddings   = get_embeddings()
    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=chunk_meta,
    )

    # ── Step 4: Save to disk ──────────────────────────────────────────────────
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    vector_store.save_local(VECTOR_DB_PATH)

    # Save metadata separately for display in the sidebar
    source_set = list({m["source"] for m in chunk_meta})
    with open(os.path.join(VECTOR_DB_PATH, "meta.pkl"), "wb") as f:
        pickle.dump({"num_chunks": len(chunks), "sources": source_set}, f)

    return vector_store


def load_vector_store():
    """
    Load a previously saved FAISS vector store from disk.
    Returns None if no saved store exists yet.
    """
    from langchain_community.vectorstores import FAISS

    index_file = os.path.join(VECTOR_DB_PATH, "index.faiss")
    if not os.path.exists(index_file):
        return None

    embeddings = get_embeddings()
    return FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def load_store_meta() -> dict:
    """
    Load metadata (chunk count, source filenames) from disk.
    Returns empty dict if no metadata file exists.
    """
    meta_file = os.path.join(VECTOR_DB_PATH, "meta.pkl")
    if not os.path.exists(meta_file):
        return {}
    with open(meta_file, "rb") as f:
        return pickle.load(f)
