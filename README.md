# RAG Chatbot — Learning Edition

A learning-first RAG chatbot that makes every phase of the pipeline
visible and measurable. Built with FastAPI, Qdrant, LangGraph, and OpenAI.

## Architecture — 7 Phases

```
User message
  │
  ▼
Phase 0 — Intent Classification   classifier.py
  │  RAG_QUERY?
  ├─ Yes ──► Phase 3 — Retrieval  retriever.py   (hybrid dense+BM25+rerank)
  │              │
  │          Phase 4 — Generation generator.py   (prompt + LLM)
  │              │
  │          Phase 5 — Evaluation evaluator.py   (RAGAS scores)
  │
  └─ No ───► Direct reply (chitchat / general knowledge / out-of-scope)

Phase 1 — Ingestion               ingestion.py   (on /upload)
Phase 2 — Embedding               embedder.py    (on /upload)
Phase 6 — Observability           observer.py    (wraps all phases)
```

## Tech Stack

| Layer | Technology |
|---|---|
| API server | FastAPI + Uvicorn |
| Orchestration | LangGraph |
| Vector DB | Qdrant |
| Embedding | OpenAI text-embedding-3-small |
| LLM | Ollama (llama3) or OpenAI (gpt-4o-mini) |
| Reranker | CrossEncoder ms-marco-MiniLM (local) |
| Cache | Redis |
| Evaluation | RAGAS |
| Storage | SQLite |

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — set memory to **8 GB minimum**
- [Ollama](https://ollama.com) (if using local LLM) — run `ollama pull llama3`
- OpenAI API key (required for embedding; also for LLM if `LLM_PROVIDER=openai`)

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY
```

### 2. Start infrastructure

```bash
docker-compose up -d qdrant redis
```

Verify Qdrant is ready:
```bash
curl http://localhost:6333/collections
# → {"result":{"collections":[]}}
```

### 3. Start the API

```bash
docker-compose up -d api
```

Verify the API is running:
```bash
curl http://localhost:8000/docs
# → FastAPI Swagger UI
```

### 4. Upload a document

```bash
curl -F "file=@yourfile.pdf" http://localhost:8000/upload
# → {"status":"indexed","filename":"yourfile.pdf","chunks":47,"avg_chunk_len":412}
```

### 5. Ask a question

```bash
curl "http://localhost:8000/ask?q=What+is+this+document+about"
# → {"answer":"...","sources":[...],"chunks":[...],"eval":{...},"intent":"RAG_QUERY",...}
```

### 6. Open the UI

| URL | Description |
|---|---|
| http://localhost:8000/ui/index.html | Chat UI with retrieval trace |
| http://localhost:8000/ui/admin.html | Admin: params, traces, eval chart |
| http://localhost:6333/dashboard | Qdrant vector browser |
| http://localhost:8000/docs | FastAPI Swagger UI |

## Project Structure

```
rag-learn-final/
├── docker-compose.yml          # 3 services: api, qdrant, redis
├── Dockerfile
├── .env                        # your config (git-ignored)
├── .env.example                # safe template to commit
├── requirements.txt
├── ui/
│   ├── index.html              # chat UI
│   └── admin.html              # admin dashboard
├── data/
│   ├── uploads/                # uploaded documents
│   └── rag_learn.db            # SQLite — auto-created on first run
└── app/
    ├── main.py                 # FastAPI routes
    ├── config.py               # settings (pydantic-settings)
    ├── db.py                   # SQLite helpers
    ├── cache.py                # Redis helpers
    ├── llm/
    │   ├── base.py             # abstract LLM interface
    │   ├── ollama.py           # Ollama adapter
    │   └── openai_llm.py       # OpenAI adapter
    ├── pipeline/
    │   ├── observer.py         # Phase 6: latency tracing
    │   ├── classifier.py       # Phase 0: intent classification
    │   ├── ingestion.py        # Phase 1: document parsing
    │   ├── embedder.py         # Phase 2: chunking + embedding
    │   ├── retriever.py        # Phase 3: hybrid search + rerank
    │   ├── generator.py        # Phase 4: prompt + LLM call
    │   └── evaluator.py        # Phase 5: RAGAS scoring
    └── agents/
        └── rag_agent.py        # LangGraph workflow
```

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Ingest and embed a document |
| `GET` | `/ask?q=...` | Full pipeline — returns answer + trace + eval |
| `GET` | `/ask-stream?q=...` | SSE streaming version |
| `GET` | `/admin/traces` | Phase latency traces |
| `GET` | `/admin/evals` | RAGAS score history |
| `GET` | `/admin/history` | Chat history |
| `GET` | `/admin/config` | Current pipeline parameters |
| `POST` | `/admin/config` | Hot-update parameters at runtime |
| `DELETE` | `/admin/cache` | Wipe Redis cache |

## Learning Experiments

Run these after uploading a document to see the pipeline in action:

| Experiment | How | What you learn |
|---|---|---|
| Compare intents | Send "Hi", "What is ML?", "What does the doc say?", "Write a poem" | How Phase 0 routes different messages |
| Tune chunk size | Admin UI → chunk_size 128 vs 1024 → re-upload | How chunking affects retrieval quality |
| Tune hybrid alpha | Admin UI → 0.0 vs 1.0 vs 0.5 | When keyword beats semantic search |
| Watch RAGAS scores | Ask same question with different top_k | Effect of retrieval depth on faithfulness |
| View prompt | Chat UI → click "View full prompt" | Exactly what the LLM received |
| Phase timings | Admin UI → Traces panel | Which phase is your bottleneck |

## Switching LLM Provider

Edit `.env`:

```bash
# Use OpenAI
LLM_PROVIDER=openai

# Use Ollama (local)
LLM_PROVIDER=ollama
```

Restart the API: `docker-compose restart api`

## Switching Embedding Model

Edit `.env`:
```bash
OPENAI_EMBEDDING_MODEL=text-embedding-3-large   # 3072 dims instead of 1536
```

**Important:** After changing the embedding model you must:
1. Delete the Qdrant collection via the dashboard at http://localhost:6333/dashboard
2. Re-upload all your documents
3. Wipe the cache: `curl -X DELETE http://localhost:8000/admin/cache`

## Ports

| Port | Service |
|---|---|
| 8000 | FastAPI (API + UI) |
| 6333 | Qdrant REST + dashboard |
| 6379 | Redis |
| 11434 | Ollama (host machine) |

## Troubleshooting

**`qdrant connection refused`**
Qdrant not ready yet. Run `docker-compose up -d qdrant` and wait 5 seconds.
Check: `curl http://localhost:6333/collections`

**`OPENAI_API_KEY` errors**
Required even when `LLM_PROVIDER=ollama` because OpenAI handles embeddings.
Set it in `.env`.

**RAGAS hangs**
RAGAS uses an LLM internally. If Ollama is slow, set `EVAL_ENABLED=false` in `.env`
for faster dev iteration and re-enable when you want to measure quality.

**Empty answer / "I don't have enough information"**
No documents indexed, or wrong question type. Check the Qdrant dashboard
at http://localhost:6333/dashboard to see if your collection has vectors.

**After changing chunk_size in Admin UI**
You must re-upload documents. The existing vectors were created with the old
chunk size and cannot be retroactively re-chunked.
Also clear the cache: `curl -X DELETE http://localhost:8000/admin/cache`
