"""
RAG (Retrieval Augmented Generation) Application
================================================
DocMind — A beautiful, modern RAG chatbot with dark/light mode toggle.
Uses HuggingFace embeddings, FAISS vector store, and Flan-T5 for generation.
No OpenAI API keys required — 100% free and open-source.

FEATURES:
- Multi-file upload (PDF, DOCX, TXT)
- FAISS vector store with persistence
- Chat history with source attribution
- Chat export button
- Analytics dashboard
- Suggested questions after indexing
- Dark / Light mode toggle
- Stunning modern UI with animations
"""

import os
import pickle
from pathlib import Path

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind · RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state for theme ──────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ── Theme variables ──────────────────────────────────────────────────────────
if st.session_state.dark_mode:
    BG_PRIMARY      = "#0a0a0f"
    BG_SECONDARY    = "#111118"
    BG_CARD         = "#16161f"
    BG_INPUT        = "#1a1a24"
    BORDER          = "#2a2a3a"
    BORDER_LIGHT    = "#222230"
    TEXT_PRIMARY    = "#f0eeea"
    TEXT_SECONDARY  = "#8b8a9a"
    TEXT_MUTED      = "#4a4a5a"
    ACCENT          = "#6c63ff"
    ACCENT_LIGHT    = "#8b85ff"
    ACCENT_GLOW     = "rgba(108,99,255,0.15)"
    GREEN           = "#00d97e"
    GREEN_BG        = "rgba(0,217,126,0.08)"
    AMBER           = "#ffb340"
    AMBER_BG        = "rgba(255,179,64,0.08)"
    RED             = "#ff5e5e"
    USER_BG         = "#1a1a2e"
    USER_BORDER     = "#6c63ff"
    BOT_BG          = "#0f1a16"
    BOT_BORDER      = "#00d97e"
    METRIC_BG       = "#13131e"
    SIDEBAR_BG      = "#0d0d14"

    SHADOW          = "rgba(0,0,0,0.5)"
else:
    BG_PRIMARY      = "#f5f4f7"
    BG_SECONDARY    = "#eeedf2"
    BG_CARD         = "#ffffff"
    BG_INPUT        = "#ffffff"
    BORDER          = "#dddbe5"
    BORDER_LIGHT    = "#e8e6f0"
    TEXT_PRIMARY    = "#1a1825"
    TEXT_SECONDARY  = "#6b6880"
    TEXT_MUTED      = "#a09dba"
    ACCENT          = "#6c63ff"
    ACCENT_LIGHT    = "#5a52e0"
    ACCENT_GLOW     = "rgba(108,99,255,0.12)"
    GREEN           = "#00a85e"
    GREEN_BG        = "rgba(0,168,94,0.08)"
    AMBER           = "#d4820a"
    AMBER_BG        = "rgba(212,130,10,0.08)"
    RED             = "#d93636"
    USER_BG         = "#eeecff"
    USER_BORDER     = "#6c63ff"
    BOT_BG          = "#edfaf4"
    BOT_BORDER      = "#00a85e"
    METRIC_BG       = "#ffffff"
    SIDEBAR_BG      = "#f0eff5"

    SHADOW          = "rgba(100,90,160,0.12)"

# ── Inject CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cabinet+Grotesk:wght@400;500;700;800;900&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [class*="css"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px;
}}

.stApp {{
    background: {BG_PRIMARY} !important;
    color: {TEXT_PRIMARY} !important;
    transition: background 0.3s ease, color 0.3s ease;
}}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {SIDEBAR_BG} !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] > div {{
    padding: 1.5rem 1rem;
}}
[data-testid="stSidebar"] * {{
    color: {TEXT_PRIMARY} !important;
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background: {BG_CARD} !important;
    border: 2px dashed {BORDER} !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: {ACCENT} !important;
}}

/* Force file uploader inner elements to match theme */
[data-testid="stFileUploader"] > div {{
    background: {BG_CARD} !important;
    border-radius: 12px !important;
}}
[data-testid="stFileUploader"] section {{
    background: {BG_CARD} !important;
    border: none !important;
    border-radius: 12px !important;
}}
[data-testid="stFileUploader"] section > div {{
    background: {BG_CARD} !important;
    border-radius: 12px !important;
}}
[data-testid="stFileUploaderDropzone"] {{
    background: {BG_CARD} !important;
    border: 2px dashed {BORDER} !important;
    border-radius: 12px !important;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: {ACCENT} !important;
    background: {ACCENT_GLOW} !important;
}}
[data-testid="stFileUploaderDropzone"] > div {{
    background: transparent !important;
}}
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p {{
    color: {TEXT_SECONDARY} !important;
}}
[data-testid="stFileUploaderDropzone"] small {{
    color: {TEXT_MUTED} !important;
}}
/* Browse files button inside uploader */
[data-testid="stFileUploaderDropzone"] button {{
    background: {BG_INPUT} !important;
    color: {TEXT_PRIMARY} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    transition: all 0.2s !important;
    box-shadow: none !important;
}}
[data-testid="stFileUploaderDropzone"] button:hover {{
    border-color: {ACCENT} !important;
    color: {ACCENT} !important;
    background: {ACCENT_GLOW} !important;
    transform: none !important;
}}

/* ── All Buttons ── */
.stButton > button {{
    font-family: 'Cabinet Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    background: {ACCENT} !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.2rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px {ACCENT_GLOW} !important;
}}
.stButton > button:hover {{
    background: {ACCENT_LIGHT} !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px {ACCENT_GLOW} !important;
}}
.stButton > button:active {{
    transform: translateY(0px) !important;
}}

/* ── Form submit button ── */
.stFormSubmitButton > button {{
    font-family: 'Cabinet Grotesk', sans-serif !important;
    font-weight: 800 !important;
    font-size: 0.9rem !important;
    background: linear-gradient(135deg, {ACCENT}, {ACCENT_LIGHT}) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px {ACCENT_GLOW} !important;
}}
.stFormSubmitButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px {ACCENT_GLOW} !important;
}}

/* ── Text Input ── */
.stTextInput > div > div > input {{
    background: {BG_INPUT} !important;
    border: 2px solid {BORDER} !important;
    border-radius: 12px !important;
    color: {TEXT_PRIMARY} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
    padding: 0.7rem 1rem !important;
    transition: all 0.2s ease !important;
}}
.stTextInput > div > div > input:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px {ACCENT_GLOW} !important;
    outline: none !important;
}}
.stTextInput > div > div > input::placeholder {{
    color: {TEXT_MUTED} !important;
}}

/* ── Metrics ── */
[data-testid="stMetric"] {{
    background: {METRIC_BG} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.4rem !important;
    box-shadow: 0 4px 20px {SHADOW} !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px {SHADOW} !important;
}}
[data-testid="stMetricLabel"] {{
    color: {TEXT_SECONDARY} !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}}
[data-testid="stMetricValue"] {{
    color: {ACCENT} !important;
    font-family: 'Cabinet Grotesk', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 900 !important;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}}
[data-testid="stExpander"] summary {{
    color: {TEXT_SECONDARY} !important;
    font-size: 0.8rem !important;
}}

/* ── Success / Error / Info ── */
.stSuccess {{
    background: {GREEN_BG} !important;
    border: 1px solid {GREEN} !important;
    border-radius: 10px !important;
    color: {GREEN} !important;
}}
.stError {{
    background: rgba(255,94,94,0.08) !important;
    border: 1px solid {RED} !important;
    border-radius: 10px !important;
}}
.stInfo {{
    background: {ACCENT_GLOW} !important;
    border: 1px solid {ACCENT} !important;
    border-radius: 10px !important;
}}
.stWarning {{
    background: {AMBER_BG} !important;
    border: 1px solid {AMBER} !important;
    border-radius: 10px !important;
}}

/* ── Download button ── */
.stDownloadButton > button {{
    font-family: 'Cabinet Grotesk', sans-serif !important;
    font-weight: 700 !important;
    background: transparent !important;
    color: {ACCENT} !important;
    border: 2px solid {ACCENT} !important;
    border-radius: 10px !important;
    transition: all 0.2s !important;
}}
.stDownloadButton > button:hover {{
    background: {ACCENT_GLOW} !important;
    transform: translateY(-1px) !important;
}}

/* ── Divider ── */
hr {{ border-color: {BORDER} !important; margin: 1rem 0 !important; }}


/* -- Toggle switch -- */
[data-testid="stToggle"] > div {{ background:  !important; border-radius: 20px !important; transition: background 0.3s ease !important; }}


/* -- Toggle switch -- */
[data-testid="stToggle"] {{
    align-items: center;
    gap: 10px !important;
}}
[data-testid="stToggle"] p {{
    color: {TEXT_SECONDARY} !important;
    font-size: 0.75rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}}
.stToggle > div {{
    background: {BORDER} !important;
    border-radius: 20px !important;
    transition: background 0.3s ease !important;
}}
.stToggle > div[aria-checked="true"] {{
    background: {ACCENT} !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 10px; }}
::-webkit-scrollbar-thumb:hover {{ background: {ACCENT}; }}

/* ── Custom chat bubbles ── */
.user-msg {{
    background: {USER_BG};
    border: 1px solid {USER_BORDER};
    border-radius: 18px 18px 4px 18px;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0 0.6rem auto;
    max-width: 80%;
    text-align: right;
    box-shadow: 0 4px 15px {SHADOW};
    animation: slideInRight 0.3s ease;
    color: {TEXT_PRIMARY};
    font-size: 0.88rem;
    line-height: 1.6;
}}

.bot-msg {{
    background: {BOT_BG};
    border: 1px solid {BOT_BORDER};
    border-radius: 18px 18px 18px 4px;
    padding: 1rem 1.2rem;
    margin: 0.6rem auto 0.6rem 0;
    max-width: 80%;
    box-shadow: 0 4px 15px {SHADOW};
    animation: slideInLeft 0.3s ease;
    color: {TEXT_PRIMARY};
    font-size: 0.88rem;
    line-height: 1.6;
}}

.msg-label {{
    font-family: 'Cabinet Grotesk', sans-serif;
    font-size: 0.65rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    opacity: 0.5;
}}

.source-chip {{
    display: inline-block;
    background: {ACCENT_GLOW};
    border: 1px solid {ACCENT};
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.68rem;
    margin: 3px 3px 0 0;
    color: {ACCENT};
    font-weight: 500;
}}



.badge-ready {{
    display: inline-flex; align-items: center; gap: 6px;
    background: {GREEN_BG}; color: {GREEN};
    border: 1px solid {GREEN}; border-radius: 20px;
    padding: 4px 14px; font-size: 0.75rem;
    font-family: 'Cabinet Grotesk', sans-serif; font-weight: 700;
}}
.badge-empty {{
    display: inline-flex; align-items: center; gap: 6px;
    background: {AMBER_BG}; color: {AMBER};
    border: 1px solid {AMBER}; border-radius: 20px;
    padding: 4px 14px; font-size: 0.75rem;
    font-family: 'Cabinet Grotesk', sans-serif; font-weight: 700;
}}

.suggest-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: all 0.2s ease;
    color: {TEXT_PRIMARY};
    font-size: 0.82rem;
    line-height: 1.4;
}}
.suggest-card:hover {{
    border-color: {ACCENT};
    background: {ACCENT_GLOW};
    transform: translateX(4px);
}}

.section-label {{
    font-family: 'Cabinet Grotesk', sans-serif;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {TEXT_MUTED};
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.section-label::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: {BORDER};
}}

.logo-text {{
    font-family: 'Cabinet Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, {ACCENT}, {GREEN});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.hero-title {{
    font-family: 'Cabinet Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1.1;
    color: {TEXT_PRIMARY};
}}
.hero-accent {{
    background: linear-gradient(135deg, {ACCENT}, {GREEN});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.hero-sub {{
    color: {TEXT_SECONDARY};
    font-size: 0.85rem;
    margin-top: 0.3rem;
    line-height: 1.5;
}}

.empty-state {{
    text-align: center;
    padding: 4rem 2rem;
    color: {TEXT_MUTED};
    font-size: 0.85rem;
    line-height: 1.8;
}}
.empty-icon {{
    font-size: 3rem;
    margin-bottom: 1rem;
    display: block;
    opacity: 0.4;
}}

.chunk-card {{
    background: {BG_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.78rem;
    color: {TEXT_SECONDARY};
    line-height: 1.6;
}}
.chunk-num {{
    font-family: 'Cabinet Grotesk', sans-serif;
    font-size: 0.62rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    color: {TEXT_MUTED};
    margin-bottom: 0.4rem;
    text-transform: uppercase;
}}

/* ── Animations ── */
@keyframes slideInRight {{
    from {{ opacity: 0; transform: translateX(20px); }}
    to   {{ opacity: 1; transform: translateX(0); }}
}}
@keyframes slideInLeft {{
    from {{ opacity: 0; transform: translateX(-20px); }}
    to   {{ opacity: 1; transform: translateX(0); }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0.5; }}
}}

.fade-in {{ animation: fadeIn 0.4s ease; }}
.pulse   {{ animation: pulse 2s infinite; }}
</style>
""", unsafe_allow_html=True)


# ── Heavy deps (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_heavy_deps():
    """Load all ML dependencies once and cache for the session."""
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


# ── Constants ────────────────────────────────────────────────────────────────
VECTOR_DB_PATH = "vector_store"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL      = "google/flan-t5-base"
CHUNK_SIZE     = 600
CHUNK_OVERLAP  = 80
TOP_K          = 4


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TEXT EXTRACTION                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def extract_text_from_pdf(file_bytes):
    import PyPDF2, io
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_docx(file_bytes):
    import docx, io
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(para.text for para in doc.paragraphs)

def extract_text_from_txt(file_bytes):
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")

def extract_text(uploaded_file):
    ext  = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.read()
    if ext == ".pdf":         return extract_text_from_pdf(data)
    elif ext in (".docx", ".doc"): return extract_text_from_docx(data)
    elif ext == ".txt":       return extract_text_from_txt(data)
    else:
        st.warning("Unsupported file type: " + ext)
        return ""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VECTOR STORE                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@st.cache_resource(show_spinner=False)
def get_embeddings(embed_model):
    deps = load_heavy_deps()
    return deps["HuggingFaceEmbeddings"](
        model_name=embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def build_vector_store(texts, metadatas):
    deps = load_heavy_deps()
    splitter = deps["TextSplitter"](
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks, chunk_meta = [], []
    for text, meta in zip(texts, metadatas):
        splits = splitter.split_text(text)
        chunks.extend(splits)
        chunk_meta.extend([meta] * len(splits))
    if not chunks:
        return None
    embeddings   = get_embeddings(EMBED_MODEL)
    vector_store = deps["FAISS"].from_texts(texts=chunks, embedding=embeddings, metadatas=chunk_meta)
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    vector_store.save_local(VECTOR_DB_PATH)
    source_set = list({m["source"] for m in chunk_meta})
    with open(os.path.join(VECTOR_DB_PATH, "meta.pkl"), "wb") as f:
        pickle.dump({"num_chunks": len(chunks), "sources": source_set}, f)
    return vector_store

def load_vector_store():
    deps       = load_heavy_deps()
    index_file = os.path.join(VECTOR_DB_PATH, "index.faiss")
    if not os.path.exists(index_file):
        return None
    embeddings = get_embeddings(EMBED_MODEL)
    return deps["FAISS"].load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

def load_store_meta():
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
    deps      = load_heavy_deps()
    tokenizer = deps["AutoTokenizer"].from_pretrained(LLM_MODEL)
    model     = deps["AutoModelForSeq2SeqLM"].from_pretrained(LLM_MODEL)
    pipe = deps["pipeline"](
        "text2text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=300, do_sample=False, temperature=1.0,
    )
    return deps["HuggingFacePipeline"](pipeline=pipe)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  RAG QUERY                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def answer_question(question, vector_store):
    docs = vector_store.similarity_search(question, k=TOP_K)
    if not docs:
        return {"answer": "I couldn't find relevant information in the uploaded documents.", "sources": [], "chunks": []}
    context  = "\n\n---\n\n".join(doc.page_content for doc in docs)
    sources  = list({doc.metadata.get("source", "unknown") for doc in docs})
    prompt   = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say 'I don't know based on the provided documents.'\n\n"
        "Context:\n" + context + "\n\nQuestion: " + question + "\n\nAnswer:"
    )
    answer = load_llm().invoke(prompt).strip()
    return {"answer": answer, "sources": sources, "chunks": [d.page_content for d in docs]}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SUGGESTED QUESTIONS                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def generate_suggested_questions(text):
    llm    = load_llm()
    prompt = (
        "Read the following document and suggest exactly 3 short, interesting questions "
        "a user might ask. Write one question per line, no numbering:\n\n"
        + text[:2000] + "\n\nQuestions:"
    )
    result    = llm.invoke(prompt).strip()
    questions = [q.strip() for q in result.split("\n") if q.strip()]
    return questions[:3]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CHAT EXPORT                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_chat_export(chat_history):
    lines = ["DocMind — Chat Export", "=" * 40, ""]
    for msg in chat_history:
        if msg["role"] == "user":
            lines.append("YOU: " + msg["content"])
        else:
            lines.append("DOCMIND: " + msg["content"])
            if msg.get("sources"):
                lines.append("Sources: " + ", ".join(msg["sources"]))

        lines.append("")
    return "\n".join(lines)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION STATE                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if "chat_history"         not in st.session_state: st.session_state.chat_history         = []
if "vector_store"         not in st.session_state: st.session_state.vector_store         = load_vector_store()
if "store_meta"           not in st.session_state: st.session_state.store_meta           = load_store_meta()
if "suggested_questions"  not in st.session_state: st.session_state.suggested_questions  = []
if "prefill_question"     not in st.session_state: st.session_state.prefill_question     = ""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

with st.sidebar:

    # ── Logo + Theme toggle ───────────────────────────────────────────────────
    # ── Logo ─────────────────────────────────────────────────────────────────
    st.markdown("<div class='logo-text'>DocMind</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.68rem;opacity:0.4;margin-top:2px;font-family:JetBrains Mono'>"
        "RAG · Local · Open-Source</p>",
        unsafe_allow_html=True
    )

    # ── Dark / Light mode TOGGLE SWITCH ──────────────────────────────────────
    mode_label = "🌙 Dark Mode" if st.session_state.dark_mode else "☀️ Light Mode"
    st.markdown(
        "<p style='font-size:0.72rem;opacity:0.5;margin:0.6rem 0 0.1rem 0;font-family:JetBrains Mono'>"
        + mode_label + "</p>",
        unsafe_allow_html=True
    )
    toggle = st.toggle(
        "Switch theme",
        value=not st.session_state.dark_mode,
        label_visibility="collapsed",
        key="theme_toggle"
    )
    # toggle=True means Light, toggle=False means Dark
    if toggle == st.session_state.dark_mode:
        st.session_state.dark_mode = not toggle
        st.rerun()

    st.markdown("---")

    # ── Upload section ────────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Upload Documents</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "drop files here",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(
            "<p style='font-size:0.75rem;opacity:0.5;margin:0.4rem 0'>"
            + str(len(uploaded_files)) + " file(s) selected</p>",
            unsafe_allow_html=True
        )
        if st.button("⚡  Process & Index", use_container_width=True):
            with st.spinner("Extracting text..."):
                texts, metas = [], []
                for f in uploaded_files:
                    text = extract_text(f)
                    if text.strip():
                        texts.append(text)
                        metas.append({"source": f.name})
            if texts:
                with st.spinner("Embedding & indexing..."):
                    vs = build_vector_store(texts, metas)
                    st.session_state.vector_store = vs
                    st.session_state.store_meta   = load_store_meta()
                with st.spinner("Generating questions..."):
                    st.session_state.suggested_questions = generate_suggested_questions(texts[0])
                total_words = sum(len(t.split()) for t in texts)
                st.success(str(len(texts)) + " file(s) · " + str(total_words) + " words indexed")
            else:
                st.error("No extractable text found.")

    st.markdown("---")

    # ── Store status ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Knowledge Base</div>", unsafe_allow_html=True)
    meta = st.session_state.store_meta
    if st.session_state.vector_store and meta:
        st.markdown("<span class='badge-ready'>● Ready</span>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.72rem;margin-top:0.6rem;opacity:0.6'>"
            + str(meta.get("num_chunks", "?")) + " chunks stored · "
            + str(len(meta.get("sources", []))) + " source(s)</p>",
            unsafe_allow_html=True,
        )
        with st.expander("View sources"):
            for s in meta.get("sources", []):
                st.markdown("<span class='source-chip'>📄 " + s + "</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge-empty'>○ Empty</span>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.72rem;margin-top:0.5rem;opacity:0.4'>"
            "Upload and index documents to begin.</p>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Controls ──────────────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Controls</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑  Clear Chat", use_container_width=True):
            st.session_state.chat_history    = []
            st.session_state.prefill_question = ""
            st.rerun()
    with c2:
        if st.button("🔄  Reload DB", use_container_width=True):
            st.session_state.vector_store = load_vector_store()
            st.session_state.store_meta   = load_store_meta()
            st.rerun()

    if st.session_state.chat_history:
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        st.download_button(
            label="📥  Export Chat",
            data=build_chat_export(st.session_state.chat_history),
            file_name="docmind_chat.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.65rem;opacity:0.25;line-height:1.7;font-family:JetBrains Mono'>"
        "Embeddings · all-MiniLM-L6-v2<br>"
        "LLM · Flan-T5-base<br>"
        "Store · FAISS (local disk)<br>"
        "No API keys · No cloud</p>",
        unsafe_allow_html=True,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN PANEL — HEADER                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.markdown(
    "<div class='fade-in'>"
    "<div class='hero-title'>Document <span class='hero-accent'>Intelligence</span></div>"
    "<div class='hero-sub'>Upload any document · Ask anything · Get grounded answers instantly.</div>"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ANALYTICS DASHBOARD                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

meta            = st.session_state.store_meta
total_questions = len([m for m in st.session_state.chat_history if m["role"] == "user"])
total_answers   = len([m for m in st.session_state.chat_history if m["role"] == "bot"])
total_docs      = len(meta.get("sources", []))
total_chunks    = meta.get("num_chunks", 0)

m1, m2, m3, m4 = st.columns(4)
m1.metric("📄  Docs Indexed",   total_docs)
m2.metric("🧩  Chunks Stored",  total_chunks)
m3.metric("❓  Questions Asked", total_questions)
m4.metric("💬  Answers Given",  total_answers)

st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
st.markdown("---")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SUGGESTED QUESTIONS                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if st.session_state.suggested_questions:
    st.markdown("<div class='section-label'>💡 Suggested Questions</div>", unsafe_allow_html=True)
    sq_cols = st.columns(len(st.session_state.suggested_questions))
    for i, question in enumerate(st.session_state.suggested_questions):
        with sq_cols[i]:
            if st.button("❓  " + question, key="sug_" + str(i), use_container_width=True):
                st.session_state.prefill_question = question
                st.rerun()
    st.markdown("---")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CHAT HISTORY                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.markdown("<div class='section-label'>💬 Conversation</div>", unsafe_allow_html=True)

with st.container():
    if not st.session_state.chat_history:
        st.markdown(
            "<div class='empty-state'>"
            "<span class='empty-icon'>🧠</span>"
            "No conversation yet.<br>"
            "Upload a document in the sidebar, then ask your first question below."
            "</div>",
            unsafe_allow_html=True,
        )

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                "<div class='user-msg'>"
                "<div class='msg-label'>You</div>"
                + msg["content"]
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            # Sources
            sources_html = "".join(
                "<span class='source-chip'>📄 " + s + "</span>"
                for s in msg.get("sources", [])
            )
            sources_div = (
                "<div style='margin-top:0.6rem'>" + sources_html + "</div>"
                if sources_html else ""
            )
            st.markdown(
                "<div class='bot-msg'>"
                "<div class='msg-label'>DocMind</div>"
                + msg["content"] + sources_div
                + "</div>",
                unsafe_allow_html=True,
            )

            if msg.get("chunks"):
                with st.expander("🔍 View retrieved context"):
                    for i, chunk in enumerate(msg["chunks"], 1):
                        st.markdown(
                            "<div class='chunk-card'>"
                            "<div class='chunk-num'>Chunk " + str(i) + "</div>"
                            + chunk + "</div>",
                            unsafe_allow_html=True,
                        )

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  INPUT FORM                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.markdown("<div class='section-label'>✍️ Ask a Question</div>", unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    col_input, col_send = st.columns([5, 1])
    with col_input:
        user_question = st.text_input(
            "question",
            value=st.session_state.prefill_question,
            placeholder="What does the document say about...?",
            label_visibility="collapsed",
        )
    with col_send:
        submitted = st.form_submit_button("Send →", use_container_width=True)

# Clear prefill after render
if st.session_state.prefill_question:
    st.session_state.prefill_question = ""

# ── Process answer ──────────────────────────────────────────────────────────
if submitted and user_question.strip():
    question = user_question.strip()
    if not st.session_state.vector_store:
        st.error("Please upload and index documents first using the sidebar.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Searching documents and generating answer..."):
            result = answer_question(question, st.session_state.vector_store)
        st.session_state.chat_history.append({
            "role":    "bot",
            "content": result["answer"],
            "sources": result["sources"],
            "chunks":  result.get("chunks", []),
        })
        st.rerun()