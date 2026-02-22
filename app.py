"""
RAG (Retrieval Augmented Generation) Application
================================================
A Streamlit-based chatbot that answers questions based on uploaded documents.
Uses HuggingFace embeddings, FAISS vector store, and Flan-T5 for generation.
No OpenAI API keys required — 100% free and open-source.
"""

import os
import pickle
import tempfile
import time
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="DocMind · RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark editorial aesthetic ───────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;1,300&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'DM Mono', monospace; }

.stApp {
    background: #0d0d0f;
    color: #e8e6e0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111114;
    border-right: 1px solid #2a2a30;
}
[data-testid="stSidebar"] * { color: #c9c7c0 !important; }

/* ── Headings ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #16161a;
    border: 1px dashed #3a3a44;
    border-radius: 8px;
    padding: 1rem;
}
[data-testid="stFileUploader"]:hover { border-color: #7c6af7; }

/* ── Buttons ── */
.stButton > button {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    background: #7c6af7;
    color: #fff;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1.4rem;
    letter-spacing: 0.04em;
    transition: background 0.2s, transform 0.1s;
}
.stButton > button:hover {
    background: #9b8df9;
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }

/* ── Chat messages ── */
.user-msg, .bot-msg {
    padding: 0.9rem 1.2rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    font-size: 0.92rem;
    line-height: 1.65;
    max-width: 85%;
}
.user-msg {
    background: #1e1e26;
    border-left: 3px solid #7c6af7;
    margin-left: auto;
    text-align: right;
}
.bot-msg {
    background: #161619;
    border-left: 3px solid #3dd68c;
}
.msg-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    opacity: 0.55;
}

/* ── Source chips ── */
.source-chip {
    display: inline-block;
    background: #1f1f28;
    border: 1px solid #2e2e3a;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    margin: 2px 2px 0 0;
    color: #9a8ff0;
}

/* ── Status badges ── */
.badge-ready {
    display: inline-block;
    background: #0e2a1e;
    color: #3dd68c;
    border: 1px solid #1a4030;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
}
.badge-empty {
    display: inline-block;
    background: #1e1a10;
    color: #c9a227;
    border: 1px solid #352e14;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
}

/* ── Input box ── */
.stTextInput > div > div > input, .stTextArea textarea {
    background: #16161a !important;
    border: 1px solid #2a2a34 !important;
    border-radius: 8px !important;
    color: #e8e6e0 !important;
    font-family: 'DM Mono', monospace !important;
}
.stTextInput > div > div > input:focus {
    border-color: #7c6af7 !important;
    box-shadow: 0 0 0 2px rgba(124,106,247,0.18) !important;
}

/* ── Divider ── */
hr { border-color: #1e1e26; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #13131a;
    border: 1px solid #22222e;
    border-radius: 8px;
}

/* ── Metric ── */
[data-testid="stMetric"] {
    background: #13131a;
    border: 1px solid #22222e;
    border-radius: 8px;
    padding: 0.7rem 1rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0d0f; }
::-webkit-scrollbar-thumb { background: #2e2e3a; border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Lazy imports (avoid crashing before deps are installed) ─────────────────
@st.cache_resource(show_spinner=False)
def load_heavy_deps():
    """Load ML dependencies once and cache them for the session."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    return {
        "TextSplitter": RecursiveCharacterTextSplitter,
        "FAISS": FAISS,
        "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
        "HuggingFacePipeline": HuggingFacePipeline,
        "pipeline": pipeline,
        "AutoTokenizer": AutoTokenizer,
        "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
        "torch": torch,
    }


# ── Constants ───────────────────────────────────────────────────────────────
VECTOR_DB_PATH = "vector_store"          # local folder for FAISS persistence
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL      = "google/flan-t5-base"  # lightweight; swap for flan-t5-large if RAM allows
CHUNK_SIZE     = 600
CHUNK_OVERLAP  = 80
TOP_K          = 4                        # number of chunks to retrieve


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TEXT EXTRACTION                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF file using PyPDF2."""
    import PyPDF2, io
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(
        page.extract_text() or "" for page in reader.pages
    )


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract paragraphs from a DOCX file using python-docx."""
    import docx, io
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(para.text for para in doc.paragraphs)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode plain text files (UTF-8 with fallback)."""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")


def extract_text(uploaded_file) -> str:
    """Route to the correct extractor based on file extension."""
    ext  = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.read()
    if ext == ".pdf":
        return extract_text_from_pdf(data)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(data)
    elif ext == ".txt":
        return extract_text_from_txt(data)
    else:
        st.warning("Unsupported file type: " + ext)
        return ""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VECTOR STORE                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@st.cache_resource(show_spinner=False)
def get_embeddings(embed_model: str):
    """Load and cache the HuggingFace embedding model."""
    deps = load_heavy_deps()
    return deps["HuggingFaceEmbeddings"](
        model_name=embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(texts, metadatas):
    """
    Split texts into chunks, embed them, and create a FAISS vector store.
    Also saves the store to disk for persistence across restarts.
    """
    deps = load_heavy_deps()

    # ── 1. Split text into overlapping chunks ───────────────────────────────
    splitter = deps["TextSplitter"](
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

    # ── 2. Embed & store ────────────────────────────────────────────────────
    embeddings   = get_embeddings(EMBED_MODEL)
    vector_store = deps["FAISS"].from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=chunk_meta,
    )

    # ── 3. Persist to disk ──────────────────────────────────────────────────
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    vector_store.save_local(VECTOR_DB_PATH)

    # Save metadata alongside (for display purposes)
    source_set = list({m["source"] for m in chunk_meta})
    meta_data  = {"num_chunks": len(chunks), "sources": source_set}
    with open(os.path.join(VECTOR_DB_PATH, "meta.pkl"), "wb") as f:
        pickle.dump(meta_data, f)

    return vector_store


def load_vector_store():
    """Load a previously saved FAISS vector store from disk (if it exists)."""
    deps       = load_heavy_deps()
    index_file = os.path.join(VECTOR_DB_PATH, "index.faiss")
    if not os.path.exists(index_file):
        return None
    embeddings = get_embeddings(EMBED_MODEL)
    return deps["FAISS"].load_local(
        VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True
    )


def load_store_meta() -> dict:
    """Load saved metadata (chunk count, source list) from disk."""
    meta_file = os.path.join(VECTOR_DB_PATH, "meta.pkl")
    if not os.path.exists(meta_file):
        return {}
    with open(meta_file, "rb") as f:
        return pickle.load(f)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LLM                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@st.cache_resource(show_spinner=False)
def load_llm():
    """
    Load Flan-T5 (seq2seq) via HuggingFace Transformers and wrap it in a
    LangChain-compatible HuggingFacePipeline.
    """
    deps      = load_heavy_deps()
    tokenizer = deps["AutoTokenizer"].from_pretrained(LLM_MODEL)
    model     = deps["AutoModelForSeq2SeqLM"].from_pretrained(LLM_MODEL)

    pipe = deps["pipeline"](
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        do_sample=False,
        temperature=1.0,
    )
    return deps["HuggingFacePipeline"](pipeline=pipe)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  RAG QUERY                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def answer_question(question: str, vector_store) -> dict:
    """
    Full RAG pipeline:
      1. Retrieve TOP_K relevant chunks via similarity search.
      2. Build a context string from those chunks.
      3. Prompt the LLM with context + question.
      4. Return the answer and the source chunks.
    """
    # ── Retrieve ─────────────────────────────────────────────────────────────
    docs = vector_store.similarity_search(question, k=TOP_K)
    if not docs:
        return {
            "answer": "I couldn't find relevant information in the uploaded documents.",
            "sources": [],
            "chunks": []
        }

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    sources  = list({doc.metadata.get("source", "unknown") for doc in docs})

    # ── Build prompt ──────────────────────────────────────────────────────────
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say 'I don't know based on the provided documents.'\n\n"
        "Context:\n" + context + "\n\n"
        "Question: " + question + "\n\n"
        "Answer:"
    )

    # ── Generate ──────────────────────────────────────────────────────────────
    llm    = load_llm()
    answer = llm.invoke(prompt).strip()

    return {
        "answer":  answer,
        "sources": sources,
        "chunks":  [d.page_content for d in docs]
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION STATE                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_vector_store()

if "store_meta" not in st.session_state:
    st.session_state.store_meta = load_store_meta()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR — FILE UPLOAD & CONTROLS                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown(
        "<h2 style='font-family:Syne;font-size:1.4rem;margin-bottom:0'>🧠 DocMind</h2>"
        "<p style='font-size:0.75rem;opacity:0.45;margin-top:2px'>RAG · Open-Source · Local</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("**Upload Documents**")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, or TXT",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("⚡ Process & Index Documents", use_container_width=True):
            with st.spinner("Extracting text..."):
                texts, metas = [], []
                for f in uploaded_files:
                    text = extract_text(f)
                    if text.strip():
                        texts.append(text)
                        metas.append({"source": f.name})

            if texts:
                with st.spinner("Building vector index (first run downloads models)..."):
                    vs = build_vector_store(texts, metas)
                    st.session_state.vector_store = vs
                    st.session_state.store_meta   = load_store_meta()
                st.success("Indexed " + str(len(texts)) + " document(s)")
            else:
                st.error("No extractable text found.")

    st.markdown("---")

    # ── Status ────────────────────────────────────────────────────────────────
    meta = st.session_state.store_meta
    if st.session_state.vector_store and meta:
        st.markdown(
            "<span class='badge-ready'>● Store ready</span>",
            unsafe_allow_html=True,
        )
        num_chunks  = meta.get("num_chunks", "?")
        num_sources = len(meta.get("sources", []))
        st.markdown(
            "<p style='font-size:0.75rem;opacity:0.5;margin-top:0.5rem'>"
            + str(num_chunks) + " chunks · "
            + str(num_sources) + " source(s)</p>",
            unsafe_allow_html=True,
        )
        with st.expander("Sources in store"):
            for s in meta.get("sources", []):
                st.markdown(
                    "<span class='source-chip'>" + s + "</span>",
                    unsafe_allow_html=True
                )
    else:
        st.markdown(
            "<span class='badge-empty'>○ No store loaded</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("💾 Reload DB", use_container_width=True):
            st.session_state.vector_store = load_vector_store()
            st.session_state.store_meta   = load_store_meta()
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.7rem;opacity:0.3;line-height:1.6'>"
        "Models: all-MiniLM-L6-v2 · Flan-T5-base<br>"
        "Vector store: FAISS (local)<br>"
        "No API keys required.</p>",
        unsafe_allow_html=True,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN PANEL                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.markdown(
    "<h1 style='font-family:Syne;font-size:2.1rem;margin-bottom:0.1rem'>"
    "Document Q&amp;A <span style='color:#7c6af7'>Assistant</span></h1>"
    "<p style='opacity:0.4;font-size:0.85rem;margin-top:0'>Ask anything about your uploaded documents.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Chat history display ────────────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        st.markdown(
            "<div style='text-align:center;opacity:0.25;padding:3rem 0;font-size:0.9rem'>"
            "Upload documents in the sidebar, then ask a question below."
            "</div>",
            unsafe_allow_html=True,
        )

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            # ── User bubble ──────────────────────────────────────────────────
            st.markdown(
                "<div class='user-msg'>"
                "<div class='msg-label'>You</div>"
                + msg["content"] +
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            # ── Bot bubble ───────────────────────────────────────────────────
            # Build sources HTML separately to avoid backslash-in-fstring issue
            sources_html = "".join(
                "<span class='source-chip'>📄 " + s + "</span>"
                for s in msg.get("sources", [])
            )

            if sources_html:
                sources_div = "<div style='margin-top:0.6rem'>" + sources_html + "</div>"
            else:
                sources_div = ""

            st.markdown(
                "<div class='bot-msg'>"
                "<div class='msg-label'>DocMind</div>"
                + msg["content"]
                + sources_div
                + "</div>",
                unsafe_allow_html=True,
            )

            # Optional: show retrieved chunks
            if msg.get("chunks"):
                with st.expander("View retrieved context chunks"):
                    for i, chunk in enumerate(msg["chunks"], 1):
                        chunk_html = (
                            "<div style='background:#0f0f14;border:1px solid #1e1e28;"
                            "border-radius:6px;padding:0.6rem 0.9rem;margin-bottom:0.4rem;"
                            "font-size:0.8rem;opacity:0.75'>"
                            "<b style='opacity:0.4'>Chunk " + str(i) + "</b><br>"
                            + chunk +
                            "</div>"
                        )
                        st.markdown(chunk_html, unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── Input row ──────────────────────────────────────────────────────────────
col_input, col_send = st.columns([6, 1])
with col_input:
    user_question = st.text_input(
        "Ask a question",
        placeholder="What does the document say about...?",
        label_visibility="collapsed",
        key="user_input",
    )
with col_send:
    send = st.button("Send →", use_container_width=True)

# ── Handle submission ───────────────────────────────────────────────────────
if (send or user_question) and user_question.strip():
    question = user_question.strip()

    if not st.session_state.vector_store:
        st.error("Please upload and index documents first (sidebar).")
    else:
        # Append user message
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )

        # Generate answer
        with st.spinner("Thinking..."):
            result = answer_question(question, st.session_state.vector_store)

        # Append bot message
        st.session_state.chat_history.append(
            {
                "role":    "bot",
                "content": result["answer"],
                "sources": result["sources"],
                "chunks":  result.get("chunks", []),
            }
        )
        st.rerun()