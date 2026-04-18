"""
RAG FastAPI backend
Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from data.constants import groq_api_key, vector_db_name, embedding_model_name, groq_model_name
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStore
from src.rag_retriever import RAGRetriever
from src.rag_pipeline import RAGPipeline
from src.grok_llm import GroqLLM
from src.libs.data_loader import load_all_documents
from src.libs.confluence_loader import load_confluence_documents
from src.libs.sharepoint_loader import load_sharepoint_documents
from src.utility import split_documents
from src.auth import (
    SESSION_COOKIE, create_session_cookie, get_current_user, require_auth,
    google_login_url, google_callback,
    microsoft_login_url, microsoft_callback,
    GOOGLE_CLIENT_ID, MS_CLIENT_ID,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Shared pipeline state
# ---------------------------------------------------------------------------

pipeline: RAGPipeline | None = None
vector_store: VectorStore | None = None
embedding_manager: EmbeddingManager | None = None


def build_pipeline():
    global pipeline, vector_store, embedding_manager

    api_key = os.environ.get("GROQ_API_KEY") or groq_api_key
    embedding_manager = EmbeddingManager(model_name=embedding_model_name)
    vector_store = VectorStore(
        collection_name=vector_db_name,
        persist_directory="data/vector_store",
    )
    retriever = RAGRetriever(vector_store=vector_store, embedding_manager=embedding_manager)
    llm = GroqLLM(model_name=groq_model_name, api_key=api_key)
    pipeline = RAGPipeline(retriever=retriever, llm=llm)


def auto_ingest_if_empty():
    """Ingest documents from data/ automatically if the vector store is empty."""
    if vector_store is None:
        return
    count = vector_store.collection.count()
    if count == 0:
        print("[startup] Vector store is empty — ingesting documents from data/ ...")
        _do_ingest("data")
    else:
        print(f"[startup] Vector store already has {count} chunks — skipping auto-ingest.")


def _do_ingest(data_dir: str) -> dict:
    """Load, chunk, embed and store all documents from data_dir."""
    docs = load_all_documents(data_dir)
    if not docs:
        return {"status": "no_documents", "chunks": 0}

    chunks = split_documents(docs)

    # Tag source_file in metadata if missing
    for chunk in chunks:
        if "source_file" not in chunk.metadata:
            src = chunk.metadata.get("source", "unknown")
            chunk.metadata["source_file"] = os.path.basename(str(src))

    texts = [c.page_content for c in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)

    # VectorStore.add_documents expects LangChain Document objects + embeddings ndarray
    vector_store.add_documents(chunks, embeddings)

    return {"status": "ok", "documents_loaded": len(docs), "chunks": len(chunks)}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    build_pipeline()
    auto_ingest_if_empty()
    yield


app = FastAPI(
    title="RAG Pipeline API",
    version="1.0.0",
    description="""
## RAG (Retrieval-Augmented Generation) API

Query your document knowledge base using semantic search + LLM-generated answers.

### Workflow
1. **`/ingest`** — load and index documents from the `data/` folder into ChromaDB
2. **`/query`** — ask a question; the API retrieves relevant chunks and generates an answer via Groq LLM
3. **`/health`** — check API status and how many chunks are indexed

### Notes
- Documents are auto-ingested on startup if the vector store is empty
- Supported file types: PDF, TXT, CSV, Excel (.xlsx), Word (.docx)
- LLM: Groq `llama-3.1-8b-instant`
- Embeddings: `all-MiniLM-L6-v2` (384-dim)
""",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Auth",      "description": "Login / logout via Google or Microsoft"},
        {"name": "Search",    "description": "Query the RAG pipeline"},
        {"name": "Ingestion", "description": "Load and index documents into the vector store"},
        {"name": "System",    "description": "Health and status checks"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:7860",
        "http://127.0.0.1:7860",
        # allow any origin in dev — remove in production
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="The question to ask against the document knowledge base",
        examples=["What is the attention mechanism?"]
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of document chunks to retrieve")
    min_score: float = Field(default=0.2, ge=0.0, le=1.0, description="Minimum similarity score threshold (0.0–1.0)")
    summarize: bool = Field(default=False, description="If true, appends a 2-sentence summary to the answer")


class SourceItem(BaseModel):
    source: str = Field(description="Source file name")
    page: str | int = Field(description="Page number within the source file")
    score: float = Field(description="Similarity score (0.0–1.0)")
    preview: str = Field(description="Short text preview of the retrieved chunk")


class QueryResponse(BaseModel):
    question: str = Field(description="The original question")
    answer: str = Field(description="LLM-generated answer with citations")
    sources: list[SourceItem] = Field(description="Document chunks used to generate the answer")
    summary: str | None = Field(default=None, description="Optional 2-sentence summary (only if summarize=true)")


class IngestRequest(BaseModel):
    data_dir: str = Field(
        default="data",
        description="Path to the folder containing documents to ingest",
        examples=["data"]
    )


class IngestResponse(BaseModel):
    status: str = Field(description="'ok' on success, 'no_documents' if folder was empty")
    documents_loaded: int = Field(default=0, description="Number of raw document pages/files loaded")
    chunks: int = Field(default=0, description="Number of text chunks stored in the vector store")


class ConfluenceIngestRequest(BaseModel):
    url: str = Field(description="Confluence base URL, e.g. https://yourproject.atlassian.net")
    url_extended: str = Field(description="Confluence wiki URL, e.g. https://yourproject.atlassian.net/wiki")
    username: str = Field(description="Atlassian account email")
    token: str = Field(description="Atlassian API token")


class SharePointIngestRequest(BaseModel):
    tenant_id: str = Field(description="Azure AD tenant ID")
    client_id: str = Field(description="Azure AD app client ID")
    client_secret: str = Field(description="Azure AD app client secret")
    site_id: str = Field(description="SharePoint site ID (from Graph API)")

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    description="Returns API status and the number of document chunks currently indexed in the vector store.",
)
def health():
    return {"status": "ok", "vector_store_chunks": vector_store.collection.count() if vector_store else 0}


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["Search"],
    summary="Query the knowledge base",
    description="""
Submit a natural language question. The API will:
1. Embed the question using `all-MiniLM-L6-v2`
2. Retrieve the top-K most similar document chunks from ChromaDB
3. Pass the chunks as context to Groq LLM (`llama-3.1-8b-instant`)
4. Return the answer with source citations

Optionally set `summarize=true` to get a 2-sentence summary appended to the answer.
""",
    responses={
        200: {"description": "Answer generated successfully"},
        503: {"description": "Pipeline not initialised"},
    },
)
def query(req: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    result = pipeline.query_formatted(
        question=req.question,
        top_k=req.top_k,
        min_score=req.min_score,
        summarize=req.summarize,
    )

    sources = [
        SourceItem(
            source=s["source"],
            page=s["page"],
            score=round(s["score"], 4),
            preview=s["preview"],
        )
        for s in result.get("sources", [])
    ]

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=sources,
        summary=result.get("summary"),
    )


@app.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Ingestion"],
    summary="Ingest documents into the vector store",
    description="""
Loads all supported documents from the specified folder, splits them into chunks,
generates embeddings, and stores them in ChromaDB.

**Supported formats:** PDF, TXT, CSV, Excel (.xlsx), Word (.docx), JSON

This is called automatically on startup if the vector store is empty.
Re-running ingest will **add** chunks on top of existing ones — clear the vector store manually if you want a full re-index.
""",
    responses={
        200: {"description": "Ingestion completed"},
        500: {"description": "Ingestion failed"},
        503: {"description": "Pipeline not initialised"},
    },
)
def ingest(req: IngestRequest = IngestRequest()):
    if embedding_manager is None or vector_store is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    try:
        result = _do_ingest(req.data_dir)
        return IngestResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

def _auth_enabled(provider: str) -> bool:
    if provider == "google":
        return bool(GOOGLE_CLIENT_ID)
    if provider == "microsoft":
        return bool(MS_CLIENT_ID)
    return False


@app.get("/auth/me", tags=["Auth"], summary="Current user info")
def auth_me(request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"authenticated": False})
    return JSONResponse({"authenticated": True, **user})


@app.get("/auth/google/login", tags=["Auth"], summary="Redirect to Google login")
def google_login(request: Request):
    if not _auth_enabled("google"):
        raise HTTPException(status_code=501, detail="Google OAuth not configured")
    state = secrets.token_urlsafe(16)
    return RedirectResponse(google_login_url(state))


@app.get("/auth/google/callback", tags=["Auth"], summary="Google OAuth2 callback")
def google_cb(code: str, request: Request):
    try:
        user = google_callback(code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Google auth failed: {e}")
    response = RedirectResponse(url="http://localhost:7860")
    response.set_cookie(SESSION_COOKIE, create_session_cookie(user), httponly=True, max_age=28800)
    return response


@app.get("/auth/microsoft/login", tags=["Auth"], summary="Redirect to Microsoft login")
def microsoft_login(request: Request):
    if not _auth_enabled("microsoft"):
        raise HTTPException(status_code=501, detail="Microsoft OAuth not configured")
    state = secrets.token_urlsafe(16)
    return RedirectResponse(microsoft_login_url(state))


@app.get("/auth/microsoft/callback", tags=["Auth"], summary="Microsoft OAuth2 callback")
def microsoft_cb(code: str, request: Request):
    try:
        user = microsoft_callback(code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Microsoft auth failed: {e}")
    response = RedirectResponse(url="http://localhost:7860")
    response.set_cookie(SESSION_COOKIE, create_session_cookie(user), httponly=True, max_age=28800)
    return response


@app.get("/auth/logout", tags=["Auth"], summary="Logout and clear session")
def logout():
    response = RedirectResponse(url="http://localhost:7860")
    response.delete_cookie(SESSION_COOKIE)
    return response


# ---------------------------------------------------------------------------
# Confluence ingest endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/ingest/confluence",
    response_model=IngestResponse,
    tags=["Ingestion"],
    summary="Ingest documents from Confluence",
    description="Loads all spaces from a Confluence instance, chunks and embeds the pages into ChromaDB.",
)
def ingest_confluence(req: ConfluenceIngestRequest):
    if embedding_manager is None or vector_store is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    try:
        docs = load_confluence_documents(req.url, req.url_extended, req.username, req.token)
        if not docs:
            return IngestResponse(status="no_documents")
        chunks = split_documents(docs)
        texts = [c.page_content for c in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        vector_store.add_documents(chunks, embeddings)
        return IngestResponse(status="ok", documents_loaded=len(docs), chunks=len(chunks))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# SharePoint ingest endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/ingest/sharepoint",
    response_model=IngestResponse,
    tags=["Ingestion"],
    summary="Ingest documents from SharePoint",
    description="""
Connects to a SharePoint site via Microsoft Graph API, downloads all supported files
(PDF, DOCX, XLSX, TXT), and indexes them into ChromaDB.

Requires an Azure AD app registration with `Sites.Read.All` permission.
""",
)
def ingest_sharepoint(req: SharePointIngestRequest):
    if embedding_manager is None or vector_store is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    try:
        docs = load_sharepoint_documents(req.tenant_id, req.client_id, req.client_secret, req.site_id)
        if not docs:
            return IngestResponse(status="no_documents")
        chunks = split_documents(docs)
        texts = [c.page_content for c in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        vector_store.add_documents(chunks, embeddings)
        return IngestResponse(status="ok", documents_loaded=len(docs), chunks=len(chunks))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
