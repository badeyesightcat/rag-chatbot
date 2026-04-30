"""
Phase 2: Embedding
Goal:   Split pages into chunks → embed via OpenAI → store in Qdrant
Key decisions: chunk_size, chunk_overlap, embedding model

IMPORTANT — model/dimension contract:
  text-embedding-3-small → 1536 dims
  text-embedding-3-large → 3072 dims

  If you change the model, you MUST:
    1. Delete the Qdrant collection (qdrant_data volume or via dashboard)
    2. Re-upload all documents
  Mixing different-dimension vectors in the same collection will crash.
"""

import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.pipeline.observer import log_phase

COLLECTION_NAME = "rag_docs"

# Dimension is derived from the model name — keeps config in one place
EMBEDDING_DIM = 3072 if "large" in settings.OPENAI_EMBEDDING_MODEL else 1536

openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Batch-embed a list of strings in a single OpenAI API call.
    One round-trip for all chunks is far faster and cheaper than
    one call per chunk.
    """
    response = openai_client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


class Embedder:

    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_HOST)
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )

    def embed_and_store(
        self,
        pages: list[dict],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict:
        cs = chunk_size    or settings.CHUNK_SIZE
        co = chunk_overlap or settings.CHUNK_OVERLAP

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cs,
            chunk_overlap=co,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        all_chunks: list[tuple[str, dict]] = []

        with log_phase("embedding", chunk_size=cs, overlap=co,
                       model=settings.OPENAI_EMBEDDING_MODEL):
            for page in pages:
                for chunk_text in splitter.split_text(page["text"]):
                    all_chunks.append((chunk_text, page["metadata"]))

            texts   = [c[0] for c in all_chunks]
            vectors = embed_texts(texts)

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"text": chunk_text, "chunk_len": len(chunk_text), **meta},
                )
                for (chunk_text, meta), vector in zip(all_chunks, vectors)
            ]

            self.client.upsert(collection_name=COLLECTION_NAME, points=points)

        return {
            "chunks":        len(points),
            "avg_chunk_len": int(
                sum(p.payload["chunk_len"] for p in points) / max(len(points), 1)
            ),
        }
