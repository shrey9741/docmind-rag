"""
ui/chat.py
==========
Renders the main chat panel UI components:
- Analytics dashboard (metric cards)
- Suggested questions
- Chat message history
- Question input form

Imported and called in app.py.
"""

import streamlit as st

from core.rag import answer_question


def render_analytics(meta: dict) -> None:
    """
    Render the 4 analytics metric cards at the top of the page.

    Args:
        meta: store metadata dict with num_chunks and sources keys
    """
    total_questions = len([m for m in st.session_state.chat_history if m["role"] == "user"])
    total_answers   = len([m for m in st.session_state.chat_history if m["role"] == "bot"])
    total_docs      = len(meta.get("sources", []))
    total_chunks    = meta.get("num_chunks", 0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📄  Docs Indexed",    total_docs)
    m2.metric("🧩  Chunks Stored",   total_chunks)
    m3.metric("❓  Questions Asked",  total_questions)
    m4.metric("💬  Answers Given",   total_answers)


def render_suggested_questions() -> None:
    """
    Render clickable suggested question buttons.
    When clicked, the question is stored in prefill_question
    so it auto-fills the input form on rerun.
    """
    if not st.session_state.suggested_questions:
        return

    st.markdown(
        "<div class='section-label'>💡 Suggested Questions</div>",
        unsafe_allow_html=True
    )
    cols = st.columns(len(st.session_state.suggested_questions))
    for i, question in enumerate(st.session_state.suggested_questions):
        with cols[i]:
            if st.button("❓  " + question, key="sug_" + str(i), use_container_width=True):
                st.session_state.prefill_question = question
                st.rerun()
    st.markdown("---")


def render_chat_history() -> None:
    """
    Render all messages in the chat history.
    User messages appear on the right, bot messages on the left.
    Each bot message has an expandable section showing raw retrieved chunks.
    """
    st.markdown(
        "<div class='section-label'>💬 Conversation</div>",
        unsafe_allow_html=True
    )

    with st.container():
        if not st.session_state.chat_history:
            # Empty state placeholder
            st.markdown(
                "<div class='empty-state'>"
                "<span class='empty-icon'>🧠</span>"
                "No conversation yet.<br>"
                "Upload a document in the sidebar, then ask your first question below."
                "</div>",
                unsafe_allow_html=True,
            )
            return

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                _render_user_bubble(msg)
            else:
                _render_bot_bubble(msg)


def render_input_form() -> None:
    """
    Render the question input form at the bottom of the page.
    Uses st.form() to prevent duplicate submissions on Streamlit reruns.

    FIX: Suggested question clicks are handled OUTSIDE the form.
    When a suggested question is clicked, it goes directly to _handle_submission()
    without needing the Send button. The form handles only manually typed questions.
    """
    st.markdown(
        "<div class='section-label'>✍️ Ask a Question</div>",
        unsafe_allow_html=True
    )

    # ── FIX: If a suggested question was clicked, process it directly ─────────
    # This bypasses the form entirely so the prefill issue doesn't occur
    if st.session_state.prefill_question:
        pending = st.session_state.prefill_question
        st.session_state.prefill_question = ""   # clear immediately
        _handle_submission(pending)              # process and rerun
        return                                   # stop here — rerun will re-render form

    # ── Normal form for manually typed questions ──────────────────────────────
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_send = st.columns([5, 1])
        with col_input:
            user_question = st.text_input(
                "question",
                placeholder="What does the document say about...?",
                label_visibility="collapsed",
            )
        with col_send:
            submitted = st.form_submit_button("Send →", use_container_width=True)

    # Process manually typed question
    if submitted and user_question.strip():
        _handle_submission(user_question.strip())


# ── Private helper functions ─────────────────────────────────────────────────

def _render_user_bubble(msg: dict) -> None:
    """Render a single user message bubble."""
    st.markdown(
        "<div class='user-msg'>"
        "<div class='msg-label'>You</div>"
        + msg["content"]
        + "</div>",
        unsafe_allow_html=True,
    )


def _render_bot_bubble(msg: dict) -> None:
    """Render a single bot message bubble with sources and chunk expander."""
    # Build source chips HTML
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
        + msg["content"]
        + sources_div
        + "</div>",
        unsafe_allow_html=True,
    )

    # Expandable context chunks
    if msg.get("chunks"):
        with st.expander("🔍 View retrieved context"):
            for i, chunk in enumerate(msg["chunks"], 1):
                st.markdown(
                    "<div class='chunk-card'>"
                    "<div class='chunk-num'>Chunk " + str(i) + "</div>"
                    + chunk
                    + "</div>",
                    unsafe_allow_html=True,
                )


def _handle_submission(question: str) -> None:
    """
    Handle a submitted question:
    1. Check vector store is ready
    2. Append user message to history
    3. Run RAG pipeline
    4. Append bot response to history
    5. Rerun to refresh UI
    """
    if not st.session_state.vector_store:
        st.error("Please upload and index documents first using the sidebar.")
        return

    # Append user message
    st.session_state.chat_history.append(
        {"role": "user", "content": question}
    )

    # Run RAG and get answer
    with st.spinner("Searching documents and generating answer..."):
        result = answer_question(question, st.session_state.vector_store)

    # Append bot response
    st.session_state.chat_history.append({
        "role":    "bot",
        "content": result["answer"],
        "sources": result["sources"],
        "chunks":  result.get("chunks", []),
    })

    st.rerun()