# DocMind 🧠 — RAG-Powered Document Intelligence System

> Upload any document. Ask anything. Get grounded answers instantly — 100% offline, zero API cost.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-red?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-1.13+-orange?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## 📌 Table of Contents

- [What is DocMind?](#what-is-docmind)
- [How It Works](#how-it-works)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture Deep Dive](#architecture-deep-dive)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Limitations](#limitations)
- [Changelog](#changelog)
- [Future Improvements](#future-improvements)

---

## What is DocMind?

DocMind is a **fully offline, open-source document Q&A system** built using the RAG (Retrieval Augmented Generation) pipeline. It solves the problem of information overload — instead of manually reading through large PDFs, contracts, or research papers, users simply upload their document and ask questions in plain English.

Unlike general chatbots that hallucinate answers from training data, DocMind answers **strictly from your document** — making it reliable, trustworthy, and suitable for sensitive documents like legal contracts, medical records, and confidential reports.

### Why DocMind over ChatGPT or Adobe?

| Feature | DocMind | ChatGPT | Adobe Acrobat |
|---|---|---|---|
| Cost | Free | ~$20/month | ~$25/month |
| Privacy | 100% Local | Cloud | Cloud |
| Works Offline | Yes | No | No |
| File Types | PDF, DOCX, TXT | PDF only | PDF only |
| Hallucination | Prevented | Possible | Possible |
| Open Source | Yes | No | No |
| API Key Required | No | Yes | Yes |

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        INDEXING PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│  Upload Document (PDF/DOCX/TXT)                                 │
│          ↓                                                      │
│  Text Extraction (PyPDF2 / python-docx / plain text)           │
│          ↓                                                      │
│  Split into 600-char chunks with 80-char overlap               │
│  (RecursiveCharacterTextSplitter)                               │
│          ↓                                                      │
│  Embed each chunk → 384-dimensional vector                      │
│  (all-MiniLM-L6-v2 Sentence Transformer)                       │
│          ↓                                                      │
│  Store all vectors in FAISS index → saved to disk              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│  User asks a question                                           │
│          ↓                                                      │
│  Question embedded → 384-dimensional vector                    │
│          ↓                                                      │
│  FAISS similarity search → TOP 4 most relevant chunks          │
│  (cosine similarity)                                            │
│          ↓                                                      │
│  Build grounded prompt:                                         │
│  "Answer ONLY from context below..." + chunks + question       │
│          ↓                                                      │
│  Flan-T5 generates answer from context only                    │
│          ↓                                                      │
│  Display answer with source filename                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

- **Multi-format Support** — Upload PDF, DOCX, DOC, and TXT files
- **Semantic Search** — Finds meaning, not just keywords — understands context
- **Zero Hallucination** — Answers strictly grounded in your document content
- **100% Offline** — After first model download, no internet required ever
- **Zero API Cost** — No OpenAI, no Anthropic, no paid APIs of any kind
- **Persistent Vector Store** — FAISS index saved to disk, survives app restarts
- **Suggested Questions** — Auto-generated questions after document indexing, cached per document
- **Chat Export** — Download your entire conversation as a .txt file
- **Analytics Dashboard** — Real-time metrics: docs indexed, chunks stored, questions asked
- **Dark / Light Mode** — Toggle between themes with smooth transitions
- **Source Attribution** — Every answer shows which document it came from
- **Context Viewer** — Expand any answer to see the raw retrieved chunks
- **Production Architecture** — Modular codebase following separation of concerns

---

## Project Structure

```
docmind-rag/
│
├── app.py                    # Main entry point — page config, CSS injection, renders UI
├── requirements.txt          # All dependencies with versions
├── .gitignore
│
├── config/
│   ├── __init__.py
│   └── settings.py           # All constants — models, chunk size, TOP_K, paths
│
├── styles/
│   └── main.css              # Complete CSS with {PLACEHOLDER} tokens for theming
│
├── ui/
│   ├── __init__.py
│   ├── theme.py              # Dark/light color palettes as dictionaries
│   ├── sidebar.py            # Sidebar component — upload, toggle, controls
│   └── chat.py               # Chat panel — analytics, suggestions, history, input
│
├── core/
│   ├── __init__.py
│   ├── extractor.py          # PDF, DOCX, TXT text extraction
│   ├── vectorstore.py        # FAISS build, save, load operations
│   ├── llm.py                # Flan-T5 model loading and caching
│   └── rag.py                # Full RAG query pipeline
│
└── utils/
    ├── __init__.py
    └── helpers.py            # Suggested questions generation, chat export
```

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| UI Framework | Streamlit 1.36 | Interactive web application |
| Orchestration | LangChain 0.3 | Connects all pipeline components |
| Embedding Model | all-MiniLM-L6-v2 | Converts text to 384-dim vectors |
| Vector Database | FAISS 1.13 | Fast similarity search on embeddings |
| Language Model | Google Flan-T5-base | Answer generation from context |
| PDF Extraction | PyPDF2 | Reads and extracts text from PDFs |
| DOCX Extraction | python-docx | Reads Word document paragraphs |
| Text Splitting | RecursiveCharacterTextSplitter | Intelligent document chunking |
| Model Hub | HuggingFace Hub 0.27 | Downloads and caches models locally |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- 8GB RAM minimum (models require ~2-3GB at runtime)
- ~400MB disk space for model cache

### Step 1 — Clone the Repository

```bash
git clone https://github.com/shrey9741/docmind-rag.git
cd docmind-rag
```

### Step 2 — Create Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```bash
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** This installs ~111 packages including PyTorch. May take 5-10 minutes depending on internet speed.

### Step 4 — Run the App

```bash
streamlit run app.py
```

Open your browser and go to:
```
http://localhost:8501
```

---

## Usage

### Step 1 — Upload a Document
Click the file uploader in the sidebar and select a PDF, DOCX, or TXT file.

### Step 2 — Index the Document
Click **⚡ Process & Index**. The app will:
- Extract text from your document
- Split it into chunks
- Generate embeddings
- Save the FAISS index to disk
- Auto-generate 3 suggested questions (cached — runs only once per unique document)

### Step 3 — Ask Questions
Type your question in the input box at the bottom and click **Send →**

Or click any of the **💡 Suggested Questions** to auto-fill and send instantly.

### Step 4 — View Results
- The answer appears as a chat bubble
- Source filename is shown as a chip below the answer
- Click **🔍 View retrieved context** to see the raw chunks used

### Step 5 — Export Chat (Optional)
Click **📥 Export Chat** in the sidebar to download the full conversation as a .txt file.

---

## Architecture Deep Dive

### Why RAG over Fine-tuning?

Fine-tuning permanently modifies model weights — expensive, slow, and requires retraining for every new document. RAG retrieves relevant information at query time without changing the model — works with any new document instantly, no training required.

### Why FAISS over ChromaDB or Pinecone?

- **ChromaDB** requires a running server process
- **Pinecone** is cloud-based — sends your data to external servers, breaks offline privacy design
- **FAISS** is lightweight, serverless, saves directly to disk, and is extremely fast at our scale

### Why all-MiniLM-L6-v2?

Best tradeoff for CPU deployment:
- 90MB — fits comfortably in 8GB RAM
- 384 dimensions — sufficient for document chunks
- Fast inference on CPU
- Top performer on MTEB semantic similarity benchmark

### Why Flan-T5-base?

- Free — no API cost
- Local — data never leaves your machine
- Encoder-decoder architecture — ideal for Q&A tasks
- Instruction-tuned — follows "answer only from context" instruction reliably
- 250MB — manageable size for CPU inference

### Caching Strategy

Performance is optimised using Streamlit's caching decorators:

```python
# Embedding model — loaded once, reused for all documents
@st.cache_resource
def get_embeddings(): ...

# LLM — loaded once, reused for all queries
@st.cache_resource
def load_llm(): ...

# Suggested questions — generated once per unique document, never re-runs on rerun
@st.cache_data
def generate_suggested_questions(text: str): ...
```

This ensures models are never reloaded on Streamlit reruns (button clicks, theme toggles, etc.), keeping the app fast after the initial load.

### Chunking Strategy

```
RecursiveCharacterTextSplitter
├── chunk_size    = 600 characters (~100-150 words)
├── chunk_overlap = 80 characters (prevents boundary cutoffs)
└── separators    = ["\n\n", "\n", ". ", " ", ""]
                    paragraph → line → sentence → word → character
```

Recursive splitting preserves natural text boundaries — it tries paragraph breaks first, falls back to sentence breaks, ensuring chunks are semantically coherent.

### Hallucination Prevention

The prompt explicitly constrains the model:
```
"Answer the question using ONLY the context below.
If the answer is not in the context, say
'I don't know based on the provided documents.'"
```

This dual instruction — answer only from context + admit when unknown — prevents the model from using its general training knowledge.

### Theme System

`styles/main.css` uses `{PLACEHOLDER}` tokens:
```css
.stApp {
    background: {BG_PRIMARY} !important;
    color: {TEXT_PRIMARY} !important;
}
```

At runtime, `app.py` reads the CSS file, replaces each `{PLACEHOLDER}` with actual hex values from `ui/theme.py`, and injects the styled CSS — completely decoupling styling from Python logic.

> **Note for VSCode users:** The `{PLACEHOLDER}` tokens will show CSS lint errors in the editor. This is expected — they are replaced at runtime and do not affect functionality. To silence them, add `.vscode/settings.json` with `{ "css.validate": false }`.

---

## Configuration

All constants are centralized in `config/settings.py`:

```python
# Models
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL   = "google/flan-t5-base"

# Chunking
CHUNK_SIZE    = 600   # characters per chunk
CHUNK_OVERLAP = 80    # overlap between consecutive chunks

# Retrieval
TOP_K = 4             # number of chunks retrieved per query

# Generation
MAX_NEW_TOKENS = 300  # max length of generated answer
```

To swap models or tune chunking, only this one file needs to change.

---

## Deployment

### Deploy on Streamlit Cloud

**Step 1** — Push your code to GitHub:
```bash
git add .
git commit -m "Initial commit"
git push
```

**Step 2** — Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub

**Step 3** — Click **New App** and configure:
```
Repository  : shrey9741/docmind-rag
Branch      : main
Main module : app.py
```

**Step 4** — Click **Deploy** and wait 5-10 minutes for model downloads

Your app will be live at:
```
https://docmind-rag.streamlit.app/
```

> **Note:** The `vector_store/` folder is gitignored. Users of the deployed app must upload and index their own documents.

> **Dependency note:** `numpy` must NOT be pinned explicitly in `requirements.txt`. LangChain 0.3 on Python 3.12+ requires `numpy<2.0.0` and will resolve a compatible version (1.26.x) automatically.

---

## Limitations

| Limitation | Details |
|---|---|
| Answer Quality | Flan-T5-base is small — answers may be short or incomplete |
| No Memory | Each question is independent — no multi-turn conversation context |
| Scanned PDFs | PyPDF2 cannot extract text from image-based scanned PDFs |
| Speed | CPU inference takes 5-15 seconds per query on cold start |
| Scale | FAISS flat index slows down with millions of documents |
| No Auth | No user authentication — single shared knowledge base |
| Free Tier RAM | Streamlit Cloud free tier has ~1GB RAM — avoid files larger than 5MB |

---

## Changelog

### v1.1.0 — Performance & Bug Fixes
- **Fixed:** `numpy==2.1.0` pin removed from `requirements.txt` — was causing `ResolutionImpossible` error on Streamlit Cloud (LangChain 0.3 requires `numpy<2.0.0` on Python 3.12+)
- **Fixed:** `@st.cache_data` added to `generate_suggested_questions()` — previously re-ran the LLM on every Streamlit interaction, causing major slowdowns
- **Fixed:** Sidebar collapse button was permanently hidden after clicking due to `header { visibility: hidden }` CSS rule — fixed by explicitly keeping `[data-testid="collapsedControl"]` visible
- **Improved:** Suggested questions prompt rewritten for better output quality — Flan-T5 now generates clearer, more factual questions from document content
- **Improved:** Updated `HuggingFaceEmbeddings` and `HuggingFacePipeline` imports to use `langchain_huggingface` package — silences LangChain deprecation warnings

### v1.0.0 — Initial Release
- Full RAG pipeline with FAISS vector store
- Flan-T5-base for answer generation
- Dark/light theme system
- Suggested questions, chat export, analytics dashboard

---

## Future Improvements

- **Better LLM** — Swap Flan-T5 for Mistral-7B or Llama-2 for higher quality answers
- **Conversation Memory** — Add LangChain ConversationBufferWindowMemory for multi-turn Q&A
- **OCR Support** — Handle scanned PDFs using pytesseract
- **Streaming Responses** — Stream tokens one by one for faster perceived speed
- **Hybrid Search** — Combine FAISS vector search with BM25 keyword search
- **Re-ranking** — Add cross-encoder re-ranking for better retrieval precision
- **URL Scraping** — Index websites using requests + BeautifulSoup4
- **Evaluation** — Integrate RAGAS framework for pipeline quality measurement
- **Multi-user** — User authentication with isolated vector stores per user

---

## Author

**Shrey** — Built as a learning project to understand RAG pipelines end to end.

- GitHub: [shrey9741](https://github.com/shrey9741)
- LinkedIn: [shrey-kumar](https://linkedin.com/in/shrey-kumar)
- Live Demo: [docmind-rag.streamlit.app](https://docmind-rag.streamlit.app/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*If you found this project helpful, consider giving it a ⭐ on GitHub!*
