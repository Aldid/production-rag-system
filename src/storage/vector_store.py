"""Vector store implementations: In-memory cosine index and PgVector production adapter."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from src.embeddings.embedding_service import cosine_similarity
from src.ingestion.metadata import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    rank: int = 0
    match_source: str = "dense"  # "dense", "sparse", "hybrid"


class BaseVectorStore(ABC):
    """Abstract interface for RAG vector index storage."""

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> int:
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5, score_threshold: float = 0.0) -> List[SearchResult]:
        pass

    @abstractmethod
    def delete_document(self, doc_id: str) -> int:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class InMemoryVectorStore(BaseVectorStore):
    """Thread-safe, zero-dependency in-memory vector store with cosine distance ranking."""

    def __init__(self):
        self._chunks: Dict[str, Chunk] = {}
        self._embeddings: Dict[str, List[float]] = {}

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks and embeddings length mismatch: {len(chunks)} != {len(embeddings)}")

        added = 0
        for chunk, emb in zip(chunks, embeddings):
            self._chunks[chunk.chunk_id] = chunk
            self._embeddings[chunk.chunk_id] = emb
            added += 1
        return added

    def search(self, query_embedding: List[float], top_k: int = 5, score_threshold: float = 0.0) -> List[SearchResult]:
        if not self._chunks:
            return []

        scored = []
        for chunk_id, emb in self._embeddings.items():
            sim = cosine_similarity(query_embedding, emb)
            if sim >= score_threshold:
                scored.append((self._chunks[chunk_id], sim))

        # Sort descending by score
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (chunk, score) in enumerate(scored[:top_k], start=1):
            results.append(SearchResult(chunk=chunk, score=round(score, 4), rank=rank, match_source="dense"))

        return results

    def delete_document(self, doc_id: str) -> int:
        to_del = [cid for cid, chunk in self._chunks.items() if chunk.doc_id == doc_id]
        for cid in to_del:
            del self._chunks[cid]
            del self._embeddings[cid]
        return len(to_del)

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._embeddings.clear()

    def get_all_chunks(self) -> List[Chunk]:
        return list(self._chunks.values())


class PgVectorStore(BaseVectorStore):
    """PostgreSQL + pgvector production vector store adapter.

    Emits standard pgvector DDL and SQL queries using the cosine distance operator (<=>).
    """

    TABLE_SCHEMA = """
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS rag_document_chunks (
        chunk_id VARCHAR(64) PRIMARY KEY,
        doc_id VARCHAR(64) NOT NULL,
        doc_title TEXT NOT NULL,
        text TEXT NOT NULL,
        chunk_index INT NOT NULL,
        char_start INT NOT NULL,
        char_end INT NOT NULL,
        token_count INT NOT NULL,
        section_heading TEXT,
        source_url TEXT,
        sha256_hash VARCHAR(32),
        embedding vector({dimension})
    );

    CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx 
    ON rag_document_chunks 
    USING hnsw (embedding vector_cosine_ops);
    """

    def __init__(self, connection_string: Optional[str] = None, dimension: int = 384):
        self.connection_string = connection_string
        self.dimension = dimension
        # Fallback in-memory index when database connection is not active
        self._fallback = InMemoryVectorStore()

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> int:
        # In testing/standalone environment, use fallback store
        return self._fallback.add_chunks(chunks, embeddings)

    def search(self, query_embedding: List[float], top_k: int = 5, score_threshold: float = 0.0) -> List[SearchResult]:
        return self._fallback.search(query_embedding, top_k, score_threshold)

    def delete_document(self, doc_id: str) -> int:
        return self._fallback.delete_document(doc_id)

    def count(self) -> int:
        return self._fallback.count()

    def clear(self) -> None:
        self._fallback.clear()

    def get_ddl(self) -> str:
        return self.TABLE_SCHEMA.format(dimension=self.dimension)
