"""Unit tests for in-memory and pgvector store adapters."""

import pytest
from src.ingestion.metadata import Chunk
from src.storage.vector_store import InMemoryVectorStore, PgVectorStore


def test_in_memory_vector_store_operations():
    store = InMemoryVectorStore()
    chunk1 = Chunk(chunk_id="c1", doc_id="d1", doc_title="D1", text="Vector database HNSW", chunk_index=0, char_start=0, char_end=20, token_count_estimate=5)
    chunk2 = Chunk(chunk_id="c2", doc_id="d1", doc_title="D1", text="Redis cache token bucket", chunk_index=1, char_start=21, char_end=45, token_count_estimate=6)

    emb1 = [1.0, 0.0, 0.0]
    emb2 = [0.0, 1.0, 0.0]

    added = store.add_chunks([chunk1, chunk2], [emb1, emb2])
    assert added == 2
    assert store.count() == 2

    # Query matching chunk 1
    query_emb = [0.9, 0.1, 0.0]
    results = store.search(query_emb, top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].score > results[1].score

    # Deletion
    deleted = store.delete_document("d1")
    assert deleted == 2
    assert store.count() == 0


def test_pgvector_ddl_generation():
    pg = PgVectorStore(dimension=384)
    ddl = pg.get_ddl()
    assert "vector(384)" in ddl
    assert "vector_cosine_ops" in ddl
    assert "rag_document_chunks" in ddl
