"""
app.py
======
DocMind RAG Application — Main Entry Point.

This file is intentionally kept minimal.
It only:
1. Configures the Streamlit page
2. Loads and injects the external CSS file
3. Initializes session state
4. Calls UI component functions

All logic lives in separate modules:
    config/settings.py   → constants
    styles/main.css      → all CSS styles
    ui/theme.py          → dark/light color palettes
    ui/sidebar.py        → sidebar component
    ui/chat.py           → chat panel component
    core/extractor.py    → file text extraction
    core/vectorstore.py  → FAISS vector store
    core/llm.py          → Flan-T5 language model
    core/rag.py          → RAG query pipeline
    utils/helpers.py     → suggested questions, chat export
"""

import streamlit as st

from config.settings  import APP_TITLE, APP_ICON, CSS_FILE_PATH
from ui.theme         import get_theme_colors
from ui.sidebar       import render_sidebar
from ui.chat          import render_analytics, render_suggested_questions, render_chat_history, render_input_form
from core.vectorstore import load_vector_store, load_store_meta


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE CONFIG                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION STATE                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if "dark_mode"           not in st.session_state: st.session_state.dark_mode           = True
if "chat_history"        not in st.session_state: st.session_state.chat_history        = []
if "vector_store"        not in st.session_state: st.session_state.vector_store        = load_vector_store()
if "store_meta"          not in st.session_state: st.session_state.store_meta          = load_store_meta()
if "suggested_questions" not in st.session_state: st.session_state.suggested_questions = []
if "prefill_question"    not in st.session_state: st.session_state.prefill_question    = ""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LOAD & INJECT CSS                                                       ║
# ║  Reads the external CSS file and injects theme colors into placeholders ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_css(css_path: str, dark_mode: bool) -> None:
    """
    Read the CSS file from disk, replace {PLACEHOLDER} tokens
    with actual theme color values, and inject into Streamlit.

    Args:
        css_path  : relative path to the CSS file
        dark_mode : current theme mode
    """
    with open(css_path, "r") as f:
        css_template = f.read()

    # Get color palette for current theme
    colors = get_theme_colors(dark_mode)

    # Replace each {PLACEHOLDER} with its actual color value
    css = css_template
    for key, value in colors.items():
        css = css.replace("{" + key + "}", value)

    st.markdown("<style>" + css + "</style>", unsafe_allow_html=True)


load_css(CSS_FILE_PATH, st.session_state.dark_mode)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  RENDER UI COMPONENTS                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Sidebar (upload, theme toggle, controls) ─────────────────────────────────
render_sidebar()

# ── Main panel header ─────────────────────────────────────────────────────────
st.markdown(
    "<div class='fade-in'>"
    "<div class='hero-title'>Document <span class='hero-accent'>Intelligence</span></div>"
    "<div class='hero-sub'>Upload any document · Ask anything · Get grounded answers instantly.</div>"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

# ── Analytics dashboard ───────────────────────────────────────────────────────
render_analytics(st.session_state.store_meta)
st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
st.markdown("---")

# ── Suggested questions ───────────────────────────────────────────────────────
render_suggested_questions()

# ── Chat history ──────────────────────────────────────────────────────────────
render_chat_history()
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────
render_input_form()
