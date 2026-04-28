from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from app.pipeline.ingestion import DocumentIngester
from app.pipeline.embedder import Embedder
from app.agents.rag_agent import run_rag
from app.pipeline.observer import get_traces
from app.db import get_history, get_eval_history
from app.config import settings
from app.cache import get_cache, set_cache
import shutil, hashlib
from pathlib import Path

app = FastAPI(title="RAG Building")
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

ingester = DocumentIngester()
embedder = Embedder()

# ---- UPLOAD ----
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Phase 1+2: Ingest document and embed it."""
    dest = Path("data/uploads") / file.filename
    dest.parent.mkdir(exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    
    pages = ingester.load(str(dest))
    stats = embedder.embed_and_store(pages)
    return {
        "status": "indexed",
        "filename": file.filename,
        **stats
    }

# ---- QUERY ----
@app.get("/ask")
async def ask(q: str):
    """Full pipeline: classify → (retrieve → generate → evaluate) OR direct reply."""
    # Check cache first
    cache_key = hashlib.md5(q.encode()).hexdigest()
    if cached := get_cache(cache_key):
        cached["cached"] = True
        return cached
    
    result = run_rag(q)

    response = {
        "answer": result["generation"]["answer"],
        "sources": result["generation"]["sources"],
        "chunks": result["chunks"],
        "eval": result["eval_scores"],
        "intent": result["intent"],
        "confidence": result["confidence"],
        "cached": False
    }
    set_cache(cache_key, response)
    return response

# ---- STREAM ----
@app.get("/ask-stream")
async def ask_stream(q: str):
    """SSE streaming — tokens appear as they're generated."""
    from app.pipeline.retriever import Retriever
    from app.llm.base import get_llm

    retriever = Retriever()
    chunks = retriever.retrieve(q)
    llm = get_llm()

    context = "\n\n".join([c["text"] for c in chunks])
    prompt = f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"

    def event_stream():
        for token in llm.generate_stream(prompt):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ---- ADMIN ----
@app.get("/admin/traces")
async def traces():
    return get_traces()

@app.get("/admin/evals")
async def evals():
    return get_eval_history()

@app.get("/admin/history")
async def history():
    return get_history()

@app.get("/admin/config")
async def get_config():
    return settings.dict()

@app.post("/admin/config")
async def update_config(cfg: dict):
    # Hot-update config at runtime (chunk_size, top_k, etc.)
    for k,v in cfg.items():
        if hasattr(settings, k.upper()):
            setattr(settings, k.upper(), v)
    return { "updated": cfg }