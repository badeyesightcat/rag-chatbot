# =============================================================
# PHASE 3: RETRIEVAL
# Two stages:
#   1) Hybrid search — dense vector + sparse BM25, fused with RRF
#   2) Reranking    — CrossEncoder scores the top candidates
#
# Embedding model : OpenAI text-embedding-3-small (API call)
# Reranker model  : CrossEncoder ms-marco-MiniLM  (local, sentence-transformers)
# These are TWO DIFFERENT MODELS for TWO DIFFERENT JOBS:
#   - Embedder: fast, converts text → vector for Qdrant search
#   - Reranker: slower, scores query-document pairs for final ranking
# =============================================================

from openai import OpenAI                              # for query embedding
from sentence_transformers import CrossEncoder         # for reranking ONLY
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from app.config import settings
from app.pipeline.observer import log_phase
from typing import List, Dict
import numpy as np

RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # local model, no API key needed
COLLECTION_NAME = "rag_docs"

openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
reranker = CrossEncoder(RERANKER_MODEL)

def embed_query(query: str) -> list[float]:
    """
    Embed a single query string using OpenAI.
    Must use the SAME model as embedder.py — if they differ,
    the vector spaces won't match and retrieval will be garbage.
    """
    response = openai_client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=query # single string for query-time embedding
    )
    return response.data[0].embedding

class Retriever:

    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_HOST)

    def retrieve(self, query: str, top_k=None, top_n=None, alpha=None) -> List[Dict]:
        """
        alpha: 0.0 = pure BM25, 1.0 = pure vector, 0.5 = balanced
        Returns list of dicts with text, metadata, AND scores (for UI display)
        """
        k = top_k or settings.TOP_K,
        n = top_n or settings.TOP_N,
        a = alpha if alpha is not None else settings.HYBRID_ALPHA

        with log_phase("retrieval", query=query, top_k=k, alpha=a):
            # --- Stage 1a: Dense vector search ---
            # embed_query() uses the same OpenAI model as embedder.py
            # Using a different model here would break search entirely
            query_vec = embed_query(query)
            dense_hits = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vec,
                limit=k,
                with_payload=True
            )

            # --- Stage 1b: Sparse BM25 search ---
            # Retrieve all docs for BM25 (in production, you'd use Qdrant sparse vectors)
            all_docs = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                with_payload=True
            )[0]
            corpus = [p.payload["text"] for p in all_docs]
            bm25 = BM25Okapi([doc.split() for doc in corpus])
            bm25_scores = bm25.get_scores(query.split())

            # --- Reciprocal Rank Fusion (RRF) ---
            RRF_K = 60
            fused: Dict[str, float] = {}

            for rank, hit in enumerate(dense_hits):
                pid = str(hit.id)
                fused[pid] = fused.get(pid, 0) + (1-a) / (RRF_K + rank + 1)

            # Get top-k by fused score
            top_pids = sorted(fused, key=fused.get, reverse=True)[:k]
            candidates = {str(p.id): p for p in all_docs}
            top_docs = [candidates[pid] for pid in top_pids if pid in candidates]

            # --- Stage 2: Rerank with CrossEncoder ---
            # CrossEncoder is a local model (no API call, no cost)
            # It scores query-document pairs more accurately than cosine similarity
            pairs = [[query, doc.payload["text"]] for doc in top_docs]
            rerank_scores = reranker.predict(pairs)
            ranked = sorted(zip(top_docs, rerank_scores), key=lambda x: x[1], reverse=True)

            # Return top-n with ALL scores for UI transparency
            results = []
            for doc, score in ranked[:n]:
                results.append({
                    "text": doc.payload["text"],
                    "source": doc.payload.get("source", "unknown"),
                    "page": doc.payload.get("page", 0),
                    "rerank_score": round(float(score), 4), # shown in UI
                    "rrf_score": round(fused.get(str(doc.id), 0), 4)
                })
            return results