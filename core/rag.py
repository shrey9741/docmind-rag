"""
core/rag.py
===========
The core RAG (Retrieval Augmented Generation) pipeline.

Steps:
1. Receive user question
2. Search FAISS for TOP_K most relevant chunks
3. Build a grounded prompt with retrieved context
4. Pass prompt to Flan-T5 to generate an answer
5. Return answer, source filenames, and raw chunks
"""

from config.settings import TOP_K
from core.llm import load_llm


def answer_question(question: str, vector_store) -> dict:
    """
    Full RAG pipeline for answering a user's question.

    Args:
        question     : the user's question string
        vector_store : loaded FAISS vector store object

    Returns:
        dict with keys:
            - answer  : generated answer string
            - sources : list of source filenames
            - chunks  : list of raw retrieved text chunks
    """

    # ── Step 1: Retrieve relevant chunks via similarity search ────────────────
    docs = vector_store.similarity_search(question, k=TOP_K)

    if not docs:
        return {
            "answer":  "I couldn't find relevant information in the uploaded documents.",
            "sources": [],
            "chunks":  []
        }

    # ── Step 2: Build context from retrieved chunks ───────────────────────────
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    sources  = list({doc.metadata.get("source", "unknown") for doc in docs})

    # ── Step 3: Build a grounded prompt ──────────────────────────────────────
    # We explicitly instruct the model to ONLY use the provided context.
    # This prevents hallucination by restricting the model to retrieved facts.
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say "
        "'I don't know based on the provided documents.'\n\n"
        "Context:\n" + context + "\n\n"
        "Question: " + question + "\n\n"
        "Answer:"
    )

    # ── Step 4: Generate answer using Flan-T5 ────────────────────────────────
    llm    = load_llm()
    answer = llm.invoke(prompt).strip()

    # ── Step 5: Return structured result ─────────────────────────────────────
    return {
        "answer":  answer,
        "sources": sources,
        "chunks":  [doc.page_content for doc in docs]
    }
