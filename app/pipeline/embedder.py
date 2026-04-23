# =============================================================
# PHASE 2: EMBEDDING
# Goal: Split pages into chunks → embed → store in Qdrant
# Key decisions: chunk_size, chunk_overlap, embedding model
# =============================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from app.config import settings
from app.pipeline.observer import log_phase
import uuid

EMBEDDING_MODEL = 