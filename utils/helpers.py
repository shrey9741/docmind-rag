"""
utils/helpers.py
================
Utility / helper functions used by the Streamlit UI.
Kept separate from core logic to maintain clean separation of concerns.

Functions:
- generate_suggested_questions : uses LLM to suggest questions from document
- build_chat_export            : converts chat history to downloadable text
"""
import streamlit as st
from core.llm import load_llm

@st.cache_data(show_spinner=False)
def generate_suggested_questions(text: str) -> list:
    """
    Use Flan-T5 to generate 3 suggested questions based on document content.
    Cached by document text so the LLM is only called ONCE per unique document —
    subsequent reruns (button clicks, theme toggles, etc.) return instantly.

    Args:
        text : first document's extracted text (first 2000 chars used)
    Returns:
        list of up to 3 question strings
    """
    llm = load_llm()
    prompt = (
        "Read the following document and suggest exactly 3 short, interesting questions "
        "a user might want to ask about it. Write one question per line, no numbering:\n\n"
        + text[:2000]
        + "\n\nQuestions:"
    )
    result    = llm.invoke(prompt).strip()
    questions = [q.strip() for q in result.split("\n") if q.strip()]
    return questions[:3]

def build_chat_export(chat_history: list) -> str:
    """
    Convert the chat history list into a plain text string for download.

    Args:
        chat_history : list of message dicts with role and content keys
    Returns:
        Formatted string ready to be written to a .txt file
    """
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