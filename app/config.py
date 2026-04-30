from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    LLM_PROVIDER:           str   = "ollama"
    OLLAMA_HOST:            str   = "http://host.docker.internal:11434"
    OLLAMA_MODEL:           str   = "llama3"
    OPENAI_API_KEY:         str   = ""
    OPENAI_MODEL:           str   = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str   = "text-embedding-3-small"

    # Services
    QDRANT_HOST:            str   = "http://qdrant:6333"
    REDIS_HOST:             str   = "redis"

    # RAG pipeline parameters — all hot-swappable via Admin UI
    CHUNK_SIZE:             int   = 512
    CHUNK_OVERLAP:          int   = 64
    TOP_K:                  int   = 10   # candidates fetched before reranking
    TOP_N:                  int   = 3    # final chunks passed to generator
    HYBRID_ALPHA:           float = 0.5  # 0=BM25 only, 1=vector only

    # Feature flags
    EVAL_ENABLED:           bool  = True  # set False to skip RAGAS during dev

    class Config:
        env_file = ".env"
        extra    = "ignore"


settings = Settings()
