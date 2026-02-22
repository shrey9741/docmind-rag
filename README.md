# 🧠 DocMind — RAG Q&A Assistant

A fully local, open-source Retrieval Augmented Generation (RAG) chatbot built with Streamlit, FAISS, HuggingFace Transformers, and LangChain.  
**No OpenAI API key. No cloud calls. Runs entirely on your machine.**

---

## Tech Stack

| Layer | Library / Model |
|---|---|
| UI | Streamlit |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS (local, persistent) |
| LLM | `google/flan-t5-base` (Seq2Seq, ~250 MB) |
| PDF parsing | PyPDF2 |
| DOCX parsing | python-docx |

---

## Quick Start

### 1. Clone / place files
```
your-project/
├── app.py
└── requirements.txt
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> First install downloads PyTorch (~700 MB) and HuggingFace models (~300 MB). Subsequent runs are fast.

### 4. Run the app
```bash
streamlit run app.py
```

---

## Usage

1. **Upload** PDF, DOCX, or TXT files in the left sidebar.
2. Click **⚡ Process & Index Documents** — models download automatically on first use.
3. **Ask questions** in the chat box at the bottom.
4. The app retrieves the 4 most relevant text chunks and feeds them to Flan-T5 to generate an answer.
5. The FAISS index is saved to `./vector_store/` — reload it anytime with **💾 Reload DB**.

---

## Configuration (top of `app.py`)

| Constant | Default | Description |
|---|---|---|
| `VECTOR_DB_PATH` | `"vector_store"` | Folder for FAISS persistence |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `LLM_MODEL` | `flan-t5-base` | Generation model |
| `CHUNK_SIZE` | `600` | Chars per chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between chunks |
| `TOP_K` | `4` | Retrieved chunks per query |

Swap `flan-t5-base` → `flan-t5-large` or `flan-t5-xl` for better answers (needs more RAM/VRAM).

---

## Notes

- First run downloads ~1 GB of model weights; cached locally by HuggingFace.
- CPU inference is supported; a GPU will be used automatically if available.
- The vector store persists across restarts in the `vector_store/` directory.
