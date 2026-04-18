# RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that lets you query your documents through a chat-based Web UI backed by a FastAPI service. It ingests PDFs, Word docs, Excel sheets, CSVs, and text files — chunks and embeds them into ChromaDB — then answers questions using Groq's LLM with cited sources.

---

## Architecture

```
Documents (PDF, DOCX, XLSX, TXT, CSV)
        │
        ▼
   Data Loader  ──►  Text Splitter  ──►  Embedding Model (all-MiniLM-L6-v2)
                                                  │
                                                  ▼
                                          ChromaDB Vector Store
                                                  │
                  User Query ──► Query Embedding ─┘
                                                  │
                                          Similarity Search (Top-K)
                                                  │
                                          Context Assembly
                                                  │
                                     Groq LLM (llama-3.1-8b-instant)
                                                  │
                                          Answer + Citations
```

## Project Structure

```
├── src/
│   ├── embedding_manager.py   # SentenceTransformer embedding wrapper
│   ├── vector_store.py        # ChromaDB store management
│   ├── rag_retriever.py       # Semantic search / retrieval
│   ├── rag_pipeline.py        # Query orchestration (simple / advanced / formatted)
│   ├── grok_llm.py            # Groq LLM wrapper
│   ├── utility.py             # Document splitting helpers
│   └── libs/
│       ├── data_loader.py     # Multi-format document loader
│       ├── embedding.py
│       ├── search.py
│       └── vectorstore.py
├── data/
│   ├── constants.py           # Model names, collection name, API key fallback
│   ├── pdf/                   # Place PDF documents here
│   ├── text_files/            # Place TXT documents here
│   ├── excelsheet/            # Place Excel documents here
│   └── vector_store/          # ChromaDB persisted store (auto-created)
├── notebook/                  # Jupyter notebooks for each pipeline stage
├── api.py                     # FastAPI backend (port 8000)
├── webui.py                   # HTML Web UI served via FastAPI (port 7860)
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) + [Docker Compose](https://docs.docker.com/compose/) — for the containerised setup
- **or** Python 3.11+ — for running locally
- A [Groq API key](https://console.groq.com/keys)

---

## Quick Start — Docker (recommended)

```bash
# 1. Clone the repo
git clone <repo-url>
cd <repo-folder>

# 2. Configure environment
cp .env.example .env
# Open .env and set your GROQ_API_KEY

# 3. Build and run
docker compose up --build
```

| Service | URL |
|---------|-----|
| Web UI | http://localhost:7860 |
| API (Swagger) | http://localhost:8000/docs |
| API (ReDoc) | http://localhost:8000/redoc |

To stop:

```bash
docker compose down
```

The `./data` folder is mounted as a volume so your vector store and documents persist across container rebuilds.

---

## Quick Start — Local (without Docker)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 4. Start the API (Terminal 1)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# 5. Start the Web UI (Terminal 2)
uvicorn webui:app --host 0.0.0.0 --port 7860
```

---

## API Reference

The API is documented interactively via Swagger UI at **`http://localhost:8000/docs`**.

### Endpoints

#### `GET /health`
Returns API status and the number of indexed chunks.

```json
{ "status": "ok", "vector_store_chunks": 525 }
```

---

#### `POST /query`
Ask a question against the knowledge base.

**Request body:**
```json
{
  "question": "What is the attention mechanism?",
  "top_k": 5,
  "min_score": 0.2,
  "summarize": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | required | Natural language question |
| `top_k` | int | 5 | Number of chunks to retrieve (1–20) |
| `min_score` | float | 0.2 | Minimum similarity score (0.0–1.0) |
| `summarize` | bool | false | Append a 2-sentence summary to the answer |

**Response:**
```json
{
  "question": "What is the attention mechanism?",
  "answer": "The attention mechanism allows... \n\nCitations:\n[1] attention.pdf (page 2)",
  "sources": [
    {
      "source": "attention.pdf",
      "page": 2,
      "score": 0.8731,
      "preview": "Attention is a mechanism that..."
    }
  ],
  "summary": null
}
```

---

#### `POST /ingest`
Load and index documents from a local folder into the vector store.

**Request body:**
```json
{ "data_dir": "data" }
```

**Response:**
```json
{
  "status": "ok",
  "documents_loaded": 102,
  "chunks": 525
}
```

> Documents are auto-ingested on startup if the vector store is empty. Re-running ingest adds chunks on top — clear `data/vector_store/` manually for a full re-index.

---

## Web UI Features

- Chat interface for conversational Q&A over your documents
- Retrieved sources panel with similarity scores and text previews
- Adjustable retrieval settings per query (Top-K, min score)
- Optional answer summarization
- Ingest panel to re-index documents without restarting

---

## Ingesting Documents

Place your documents in the appropriate sub-folders under `data/` before starting:

| Format | Folder |
|--------|--------|
| PDF | `data/pdf/` |
| TXT | `data/text_files/` |
| Excel (`.xlsx`, `.xls`) | `data/excelsheet/` |

The API auto-ingests on startup if the vector store is empty. You can also trigger it manually via the `/ingest` endpoint or the Ingest panel in the Web UI.

Alternatively, run the ingestion notebooks in `notebook/`:

| Notebook | Purpose |
|----------|---------|
| `pdf_loader.ipynb` | Load and index PDF files |
| `document_loader.ipynb` | Load TXT / DOCX files |
| `excel_loader.ipynb` | Load Excel files |
| `confluence_loader.ipynb` | Load pages from Confluence |

---

## Configuration

| Variable | Where | Description |
|----------|-------|-------------|
| `GROQ_API_KEY` | `.env` | Your Groq API key (required) |
| `vector_db_name` | `data/constants.py` | ChromaDB collection name |
| `embedding_model_name` | `data/constants.py` | SentenceTransformer model |
| `groq_model_name` | `data/constants.py` | Groq LLM model ID |

Default LLM: `llama-3.1-8b-instant`. Other available Groq models: `llama-3.3-70b-versatile`, `qwen/qwen3-32b`.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| LLM | [Groq](https://groq.com) — `langchain-groq` |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Store | [ChromaDB](https://www.trychroma.com) |
| Document Loading | `langchain-community` |
| API | [FastAPI](https://fastapi.tiangolo.com) + Uvicorn |
| API Docs | Swagger UI (`/docs`) · ReDoc (`/redoc`) |
| Orchestration | [LangChain](https://python.langchain.com) |
