"""
Phase 3: Retrieval
Two stages:
  1. Hybrid search  — dense vector (OpenAI) + sparse BM25, fused via RRF
  2. Reranking      — CrossEncoder scores the top candidates

Embedding model : OpenAI text-embedding-3-small  (API call)
Reranker model  : CrossEncoder ms-marco-MiniLM   (local, no API key)

These are TWO DIFFERENT MODELS for TWO DIFFERENT JOBS:
  - Embedder : fast, text → vector, used for approximate search
  - Reranker : slower, scores exact query-document pairs for precise ranking
"""

import numpy as np
from openai import OpenAI
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.config import settings
from app.pipeline.observer import log_phase

RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "rag_docs"

openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
reranker       = CrossEncoder(RERANKER_MODEL)


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.
    MUST use the same model as embedder.py — different models produce
    incompatible vector spaces and will silently return wrong results.
    """
    response = openai_client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=query,
    )
    return response.data[0].embedding


class Retriever:

    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_HOST)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        top_n: int | None = None,
        alpha: float | None = None,
    ) -> list[dict]:
        """
        alpha: 0.0 = pure BM25, 1.0 = pure vector, 0.5 = balanced hybrid
        Returns list of dicts with text, metadata, and scores for UI display.
        """
        k = top_k or settings.TOP_K
        n = top_n or settings.TOP_N
        a = alpha if alpha is not None else settings.HYBRID_ALPHA

        with log_phase("retrieval", query=query[:60], top_k=k, alpha=a):

            # Stage 1a: Dense vector search
            query_vec  = embed_query(query)
            dense_hits = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vec,
                limit=k,
                with_payload=True,
            )

            # Stage 1b: Sparse BM25 search
            all_docs = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                with_payload=True,
            )[0]

            corpus      = [p.payload["text"] for p in all_docs]
            bm25        = BM25Okapi([doc.split() for doc in corpus])
            bm25_scores = bm25.get_scores(query.split())

            # Reciprocal Rank Fusion
            RRF_K: int          = 60
            fused: dict[str, float] = {}

            for rank, hit in enumerate(dense_hits):
                pid        = str(hit.id)
                fused[pid] = fused.get(pid, 0) + a / (RRF_K + rank + 1)

            bm25_ranked = np.argsort(bm25_scores)[::-1][:k]
            for rank, idx in enumerate(bm25_ranked):
                pid        = str(all_docs[idx].id)
                fused[pid] = fused.get(pid, 0) + (1 - a) / (RRF_K + rank + 1)

            top_pids   = sorted(fused, key=fused.get, reverse=True)[:k]  # type: ignore[arg-type]
            candidates = {str(p.id): p for p in all_docs}
            top_docs   = [candidates[pid] for pid in top_pids if pid in candidates]

            # Stage 2: CrossEncoder reranking
            pairs         = [[query, doc.payload["text"]] for doc in top_docs]
            rerank_scores = reranker.predict(pairs)
            ranked        = sorted(
                zip(top_docs, rerank_scores), key=lambda x: x[1], reverse=True
            )

            return [
                {
                    "text":         doc.payload["text"],
                    "source":       doc.payload.get("source", "unknown"),
                    "page":         doc.payload.get("page", 0),
                    "rerank_score": round(float(score), 4),
                    "rrf_score":    round(fused.get(str(doc.id), 0), 4),
                }
                for doc, score in ranked[:n]
            ]
