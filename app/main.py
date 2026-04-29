"""
app/main.py
===========
FastAPI application entry point.

All routes in one place:
  POST /upload              — Phase 1+2: ingest & embed a document
  GET  /ask                 — Full pipeline (classify → RAG or direct) + cache
  GET  /ask-stream          — SSE streaming version of /ask (tokens as they arrive)
  GET  /admin/traces        — Phase latency traces for admin dashboard
  GET  /admin/evals         — RAGAS score history for admin chart
  GET  /admin/history       — Chat history for admin table
  GET  /admin/config        — Read current pipeline parameters
  POST /admin/config        — Hot-update pipeline parameters at runtime
  DELETE /admin/cache       — Wipe Redis cache (call after re-indexing documents)

Changes accumulated across the project:
  - OpenAI embeddings (text-embedding-3-small) replacing BAAI/bge-m3
  - Intent classification (Phase 0) wired into both /ask and /ask-stream
  - intent + confidence exposed in /ask response
  - save_chat() called after every /ask so admin history is populated
  - init_db() called on startup to create SQLite tables
  - clear_cache() exposed as DELETE /admin/cache
  - FileResponse + HTTPException removed (unused imports)
  - No typing.Dict / typing.List — Python 3.9+ built-in types used throughout
"""

import shutil
import hashlib
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.pipeline.ingestion import DocumentIngester
from app.pipeline.embedder import Embedder
from app.pipeline.classifier import IntentClassifier
from app.pipeline.retriever import Retriever
from app.pipeline.generator import Generator
from app.pipeline.observer import get_traces
from app.agents.rag_agent import run_rag
from app.llm.base import get_llm
from app.db import init_db, get_history, get_eval_history, save_chat
from app.config import settings
from app.cache import get_cache, set_cache, clear_cache


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="RAG Learn")

# Serve ui/index.html and ui/admin.html at /ui/*
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

# Module-level singletons — instantiated once, reused across all requests
ingester   = DocumentIngester()
embedder   = Embedder()
classifier = IntentClassifier()
retriever  = Retriever()
generator  = Generator()


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    """
    Runs once when the API server starts.
    Creates SQLite tables if they don't exist yet.
    Without this, the first save_chat() call would fail.
    """
    init_db()


# ---------------------------------------------------------------------------
# POST /upload  —  Phase 1 (Ingestion) + Phase 2 (Embedding)
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Accept a PDF / DOCX / TXT / MD file, run it through the ingestion
    and embedding pipeline, and store the resulting vectors in Qdrant.

    Returns stats so the UI can show how many chunks were created.
    """
    dest = Path("data/uploads") / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    pages = ingester.load(str(dest))       # Phase 1: parse + clean
    stats = embedder.embed_and_store(pages) # Phase 2: chunk + embed + store

    return {
        "status":   "indexed",
        "filename": file.filename,
        **stats    # chunks, avg_chunk_len — shown in UI after upload
    }


# ---------------------------------------------------------------------------
# GET /ask  —  Full pipeline (non-streaming)
# ---------------------------------------------------------------------------

@app.get("/ask")
async def ask(q: str):
    """
    Full RAG pipeline:
      Phase 0: classify intent
        → RAG_QUERY       : Phase 3 retrieve → Phase 4 generate → Phase 5 evaluate
        → CHITCHAT        : direct LLM reply, no retrieval
        → GENERAL_KNOWLEDGE: direct LLM reply, no retrieval
        → OUT_OF_SCOPE    : canned refusal, zero LLM cost

    Response shape:
      answer      : str   — the final answer text
      sources     : list  — cited source chunks (RAG_QUERY only)
      chunks      : list  — all retrieved chunks with rerank + RRF scores
      eval        : dict  — RAGAS scores (faithfulness, relevancy, precision)
      intent      : str   — classified intent, shown as badge in UI
      confidence  : float — classifier confidence 0–1
      cached      : bool  — True if served from Redis cache

    Cache key is the MD5 hash of the question string.
    TTL is 300 seconds (defined in cache.py).
    """
    cache_key = hashlib.md5(q.encode()).hexdigest()

    # Serve from cache if available — skips the entire pipeline
    if cached := get_cache(cache_key):
        cached["cached"] = True
        return cached

    # Run the full LangGraph pipeline (classify → route → respond)
    result = run_rag(q)

    response = {
        "answer":     result["generation"]["answer"],
        "sources":    result["generation"]["sources"],
        "chunks":     result["chunks"],      # full retrieved chunks with scores
        "eval":       result["eval_scores"],
        "intent":     result["intent"],      # e.g. "RAG_QUERY" — badge in UI
        "confidence": result["confidence"],  # classifier confidence score
        "cached":     False,
    }

    # Persist to SQLite so admin history table is populated
    save_chat(
        question=q,
        answer=response["answer"],
        intent=response["intent"],
        cached=False,
    )

    set_cache(cache_key, response)
    return response


# ---------------------------------------------------------------------------
# GET /ask-stream  —  SSE streaming version
# ---------------------------------------------------------------------------

@app.get("/ask-stream")
async def ask_stream(q: str):
    """
    Same pipeline as /ask but streams tokens to the browser as they
    are generated by the LLM, instead of waiting for the full answer.

    Uses Server-Sent Events (SSE):
      - Each token is pushed as:  data: <token>\n\n
      - End of stream signal:     data: [DONE]\n\n

    The browser connects with EventSource or fetch + ReadableStream.

    Phase 0 (classification) still runs so OUT_OF_SCOPE messages
    are refused immediately without wasting retrieval budget.
    The RAGAS evaluation phase is skipped in stream mode because
    we don't have the full answer until the stream ends — evaluation
    is only available via /ask.
    """
    # Phase 0: classify — still gate non-RAG intents even in stream mode
    classification = classifier.classify(q)

    # Short-circuit for non-RAG intents — stream the direct reply
    if not classification["should_rag"]:
        direct = classification["direct_reply"] or ""

        def direct_stream():
            # Stream word by word so the UI still feels live
            for word in direct.split(" "):
                yield f"data: {word} \n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(direct_stream(), media_type="text/event-stream")

    # Phase 3: retrieve relevant chunks
    chunks = retriever.retrieve(q)

    # Build the same structured prompt as generator.py uses in /ask
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        label = f"[{i}] {chunk['source']} p.{chunk['page']}"
        context_parts.append(f"{label}\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    prompt = (
        "You are a helpful assistant. Answer using ONLY the context below.\n"
        "If the context does not contain the answer, say "
        "'I don't have enough information to answer this.'\n\n"
        f"=== Context ===\n{context}\n\n"
        f"=== Question ===\n{q}\n\n"
        "=== Answer ==="
    )

    # Phase 4: stream tokens from LLM
    llm = get_llm()

    def event_stream():
        for token in llm.generate_stream(prompt):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Admin routes  —  all consumed by ui/admin.html
# ---------------------------------------------------------------------------

@app.get("/admin/traces")
async def traces():
    """
    Returns the last 50 phase trace entries from the in-memory TRACE_STORE
    in observer.py. Each entry has: phase, duration_ms, status, params.
    admin.html polls this every 10 seconds and renders the trace list.
    """
    return get_traces()


@app.get("/admin/evals")
async def evals():
    """
    Returns RAGAS score rows from the eval_results SQLite table.
    admin.html plots these as a line chart over time.
    """
    return get_eval_history()


@app.get("/admin/history")
async def history():
    """
    Returns recent rows from the chat_history SQLite table.
    admin.html renders these in the history table.
    """
    return get_history()


@app.get("/admin/config")
async def get_config():
    """
    Returns the current settings object as a dict.
    admin.html reads this on page load to pre-fill the sliders.
    """
    return settings.dict()


@app.post("/admin/config")
async def update_config(cfg: dict):
    """
    Hot-updates pipeline parameters at runtime without restarting the server.
    The admin sliders call this on Save.

    Accepted keys (must match Settings field names, case-insensitive):
      chunk_size, chunk_overlap, top_k, top_n, hybrid_alpha, eval_enabled

    Note: chunk_size and chunk_overlap only affect NEW uploads.
    Existing vectors in Qdrant were embedded with the old settings.
    After changing chunk size, re-upload your documents.
    """
    for key, value in cfg.items():
        field = key.upper()
        if hasattr(settings, field):
            setattr(settings, field, value)

    return {"updated": cfg, "current": settings.dict()}


@app.delete("/admin/cache")
async def delete_cache():
    """
    Wipes all Redis cache entries.
    Call this after re-uploading documents so stale cached answers
    don't get served for questions whose context has changed.
    admin.html save button includes a note reminding users to do this.
    """
    clear_cache()
    return {"status": "cache cleared"}