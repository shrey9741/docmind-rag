"""
core/extractor.py
=================
Handles text extraction from uploaded files.
Supports PDF, DOCX, and TXT formats.
Each format has its own dedicated function.
"""

from pathlib import Path


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract all text from a PDF file page by page using PyPDF2.
    Returns empty string for pages with no extractable text (e.g. scanned images).
    """
    import PyPDF2
    import io
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract text from a DOCX file paragraph by paragraph using python-docx.
    """
    import docx
    import io
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(para.text for para in doc.paragraphs)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """
    Decode a plain text file.
    Tries UTF-8 first, falls back to latin-1 for legacy files.
    """
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")


def extract_text(uploaded_file) -> str:
    """
    Router function — reads the file extension and calls the
    appropriate extractor. Returns empty string for unsupported types.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Extracted text as a single string
    """
    ext  = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.read()

    if ext == ".pdf":
        return extract_text_from_pdf(data)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(data)
    elif ext == ".txt":
        return extract_text_from_txt(data)
    else:
        return ""
