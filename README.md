## project structure

/opt/rag-learn/
├── docker-compose.yml # Only 3 services: API, Qdrant, Redis
├── .env # Environment config
├── Dockerfile
├── requirements.txt
│
├── app/
│ ├── main.py # FastAPI app + all routes
│ ├── config.py # Settings (pydantic-settings)
│ │
│ ├── pipeline/ # One folder = one RAG phase
│ │ ├── ingestion.py # Phase 1: Load, parse, clean
│ │ ├── embedder.py # Phase 2: Chunk + embed
│ │ ├── retriever.py # Phase 3: Hybrid search + rerank
│ │ ├── generator.py # Phase 4: Prompt + LLM call
│ │ ├── evaluator.py # Phase 5: RAGAS scoring
│ │ └── observer.py # Phase 6: Trace & metrics
│ │
│ ├── llm/
│ │ ├── base.py # Abstract LLM interface
│ │ ├── ollama.py # Ollama adapter
│ │ └── openai.py # OpenAI adapter (drop-in swap)
│ │
│ ├── agents/
│ │ └── rag_agent.py # LangGraph workflow (4 nodes)
│ │
│ ├── db.py # SQLite: chat history + eval logs
│ └── cache.py # Redis: response cache
│
├── ui/
│ ├── index.html # Chat UI — shows retrieval trace
│ └── admin.html # Admin: config + metrics dashboard
│
└── data/
├── uploads/ # User-uploaded documents
└── rag_learn.db # SQLite database
