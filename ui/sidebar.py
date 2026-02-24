"""
ui/sidebar.py
=============
Renders the entire sidebar UI component.
Imported and called once in app.py.

Responsibilities:
- Logo and theme toggle
- File upload and indexing
- Knowledge base status
- Chat controls (clear, reload, export)
"""

import streamlit as st

from core.extractor   import extract_text
from core.vectorstore import build_vector_store, load_vector_store, load_store_meta
from utils.helpers    import generate_suggested_questions, build_chat_export


def render_sidebar() -> None:
    """
    Renders the complete sidebar.
    Reads and writes st.session_state directly.
    Returns nothing — all state changes happen via session_state.
    """
    with st.sidebar:

        # ── Logo ──────────────────────────────────────────────────────────────
        st.markdown(
            "<div class='logo-text'>DocMind</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='font-size:0.68rem;opacity:0.4;margin-top:2px;"
            "font-family:JetBrains Mono'>RAG · Local · Open-Source</p>",
            unsafe_allow_html=True
        )

        # ── Dark / Light Mode Toggle ──────────────────────────────────────────
        mode_label = "🌙 Dark Mode" if st.session_state.dark_mode else "☀️ Light Mode"
        st.markdown(
            "<p style='font-size:0.72rem;opacity:0.5;margin:0.6rem 0 0.1rem 0;"
            "font-family:JetBrains Mono'>" + mode_label + "</p>",
            unsafe_allow_html=True
        )
        toggle = st.toggle(
            "Switch theme",
            value=not st.session_state.dark_mode,
            label_visibility="collapsed",
            key="theme_toggle"
        )
        # toggle=True → Light Mode, toggle=False → Dark Mode
        if toggle == st.session_state.dark_mode:
            st.session_state.dark_mode = not toggle
            st.rerun()

        st.markdown("---")

        # ── File Upload ───────────────────────────────────────────────────────
        st.markdown(
            "<div class='section-label'>Upload Documents</div>",
            unsafe_allow_html=True
        )
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
                _process_files(uploaded_files)

        st.markdown("---")

        # ── Knowledge Base Status ─────────────────────────────────────────────
        _render_store_status()

        st.markdown("---")

        # ── Controls ──────────────────────────────────────────────────────────
        _render_controls()

        st.markdown("---")

        # ── Footer ────────────────────────────────────────────────────────────
        st.markdown(
            "<p style='font-size:0.65rem;opacity:0.25;line-height:1.7;"
            "font-family:JetBrains Mono'>"
            "Embeddings · all-MiniLM-L6-v2<br>"
            "LLM · Flan-T5-base<br>"
            "Store · FAISS (local disk)<br>"
            "No API keys · No cloud</p>",
            unsafe_allow_html=True,
        )


# ── Private helper functions ─────────────────────────────────────────────────

def _process_files(uploaded_files) -> None:
    """Extract text, build vector store, generate suggested questions."""
    with st.spinner("Extracting text..."):
        texts, metas = [], []
        for f in uploaded_files:
            text = extract_text(f)
            if text.strip():
                texts.append(text)
                metas.append({"source": f.name})

    if not texts:
        st.error("No extractable text found.")
        return

    with st.spinner("Embedding & indexing..."):
        vs = build_vector_store(texts, metas)
        st.session_state.vector_store = vs
        st.session_state.store_meta   = load_store_meta()

    with st.spinner("Generating suggested questions..."):
        st.session_state.suggested_questions = generate_suggested_questions(texts[0])

    total_words = sum(len(t.split()) for t in texts)
    st.success(
        str(len(texts)) + " file(s) · " + str(total_words) + " words indexed"
    )


def _render_store_status() -> None:
    """Render the knowledge base status section."""
    st.markdown(
        "<div class='section-label'>Knowledge Base</div>",
        unsafe_allow_html=True
    )
    meta = st.session_state.store_meta

    if st.session_state.vector_store and meta:
        st.markdown(
            "<span class='badge-ready'>● Ready</span>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='font-size:0.72rem;margin-top:0.6rem;opacity:0.6'>"
            + str(meta.get("num_chunks", "?")) + " chunks · "
            + str(len(meta.get("sources", []))) + " source(s)</p>",
            unsafe_allow_html=True,
        )
        with st.expander("View sources"):
            for s in meta.get("sources", []):
                st.markdown(
                    "<span class='source-chip'>📄 " + s + "</span>",
                    unsafe_allow_html=True
                )
    else:
        st.markdown(
            "<span class='badge-empty'>○ Empty</span>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='font-size:0.72rem;margin-top:0.5rem;opacity:0.4'>"
            "Upload and index documents to begin.</p>",
            unsafe_allow_html=True
        )


def _render_controls() -> None:
    """Render Clear Chat, Reload DB, and Export Chat buttons."""
    st.markdown(
        "<div class='section-label'>Controls</div>",
        unsafe_allow_html=True
    )
    c1, c2 = st.columns(2)

    with c1:
        if st.button("🗑  Clear Chat", use_container_width=True):
            st.session_state.chat_history     = []
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
