# =============================================================
# PHASE 2: EMBEDDING
# Goal: Split pages into chunks → embed → store in Qdrant
# Key decisions: chunk_size, chunk_overlap, embedding model
# =============================================================

from langchain_text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI                # OpenAI client for embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from app.config import settings
from app.pipeline.observer import log_phase
import uuid

# OpenAI embedding model + its output dimension
# text-embedding-3-small → 1536 dims  (fast, cheap, good quality)
# text-embedding-3-large → 3072 dims  (slower, expensive, best quality)
# ⚠️  If you change the model here, you MUST:
#     1) Delete the Qdrant collection (different dims are incompatible)
#     2) Re-upload all your documents
EMBEDDING_MODEL = settings.OPENAI_EMBEDDING_MODEL   # from .env
EMBEDDING_DIM   = 3072 if "large" in EMBEDDING_MODEL else 1536
COLLECTION_NAME = "rag_docs"

openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Batch embed a list of strings using OpenAI's API.
    Batching is important — one API call for 100 chunks is
    much faster and cheaper than 100 individual calls.
    """
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts # accepts a list directly
    )
    # response.data is a list of Embedding objects, one per input string
    return [item.embedding for item in response.data]

class Embedder:

    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_HOST)
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create Qdrant collection if it doesn't exist yet."""
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM # 1536 for -small, 3072 for -large
                    distance=
                )
            )
    
    def embed_and_store(self, pages, chunk_size=None, chunk_overlap: None):
        """
        Main method called after ingestion.
        chunk_size / chunk_overlap come from config (or Admin UI override).
        """
        cs = chunk_size or settings.CHUNK_SIZE
        co = chunk_overlap or settings.CHUNK_OVERLAP

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cs,
            chunk_overlap=co,
            # Separators tried in order — tries paragraph first, then sentence, etc.
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        all_chunks = [] # (chunk_text, metadata) tuples
        stats = { "chunks": 0, "avg_chunk_len": 0 }

        with log_phase("embedding", chunk_size=cs, overlap=co, model=EMBEDDING_MODEL):
            for page in pages:
                chunks = splitter.split_text(page["text"])
                for chunk in chunks:
                    all_chunks.append((chunk, page['metadata']))

            # Embed ALL chunks in one batch call — much more efficient
            texts = [c[0] for c in all_chunks]
            vectors = embed_texts(texts) # single API round-trip

            # Build Qdrant points
            all_points = []
            for (chunk_text, meta), vector in zip(all_chunks, vectors):
                all_points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk_text,
                        "chunk_len": len(chunk_text), # visible in UI
                        **meta
                    }
                ))
            
            # Batch upsert — faster than one-by-one
            self.client.upsert(collection_name=COLLECTION_NAME, points=all_points)

            stats["chunks"] = len(all_chunks)
            stats["avg_chunk_len"] = int(sum(p.payload["chunk_len"] for p in all_points) / max(len(all_points), 1))
        
        return stats # returned to API → shown in UI