# RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that lets you query your documents through a chat-based Web UI. It ingests PDFs, Word docs, Excel sheets, CSVs, and text files — chunks and embeds them into ChromaDB — then answers questions using Groq's LLM with cited sources.

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
                                        Groq LLM (gemma2-9b-it)
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
│   ├── constants.py           # Model names, collection name
│   ├── pdf/                   # Place PDF documents here
│   ├── text_files/            # Place TXT documents here
│   ├── excelsheet/            # Place Excel documents here
│   └── vector_store/          # ChromaDB persisted store (auto-created)
├── notebook/                  # Jupyter notebooks for each pipeline stage
├── webui.py                   # Gradio web UI
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

# 4. Open the Web UI
# http://localhost:7860
```

To stop:

```bash
docker compose down
```

The `./data` folder is mounted as a volume, so your vector store and documents persist across container rebuilds.

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

# 4. Launch the Web UI
python webui.py
# Open http://localhost:7860
```

---

## Ingesting Documents

Place your documents in the appropriate sub-folders under `data/` before starting:

| Format | Folder |
|--------|--------|
| PDF | `data/pdf/` |
| TXT | `data/text_files/` |
| Excel (`.xlsx`, `.xls`) | `data/excelsheet/` |

Then run the ingestion notebooks in order from the `notebook/` folder:

| Notebook | Purpose |
|----------|---------|
| `pdf_loader.ipynb` | Load and index PDF files |
| `document_loader.ipynb` | Load TXT / DOCX files |
| `excel_loader.ipynb` | Load Excel files |
| `confluence_loader.ipynb` | Load pages from Confluence |

Or use the `src/libs/data_loader.py` `load_all_documents()` function directly in a script.

---

## Web UI Features

- **Chat interface** — conversational question answering over your documents
- **Retrieved sources panel** — shows which document chunks were used, with similarity scores and previews
- **Retrieval settings** (adjustable per query):
  - Top-K — number of document chunks to retrieve (1–10)
  - Min similarity score — filters out low-confidence chunks
- **Summarize toggle** — adds a 2-sentence summary below the full answer
- **Clear** — resets the conversation and query history

---

## Configuration

| Variable | Where | Description |
|----------|-------|-------------|
| `GROQ_API_KEY` | `.env` | Your Groq API key (required) |
| `vector_db_name` | `data/constants.py` | ChromaDB collection name |
| `embedding_model_name` | `data/constants.py` | SentenceTransformer model |

Default model: `gemma2-9b-it` via Groq. Other supported Groq models: `llama3-70b-8192`, `mixtral-8x7b-32768`.

---

## Pipeline Flow Diagrams

Visual diagrams are available in the `flow_images/` folder:

- `ragpipelineflow.jpg` — end-to-end RAG flow
- `queryingestionpipelineflow.jpg` — ingestion pipeline
- `querygenerationflow.jpg` — query generation flow

---

## Tech Stack

| Component | Library |
|-----------|---------|
| LLM | [Groq](https://groq.com) — `langchain-groq` |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Store | [ChromaDB](https://www.trychroma.com) |
| Document Loading | `langchain-community` |
| Web UI | [Gradio](https://gradio.app) |
| Orchestration | [LangChain](https://python.langchain.com) |
