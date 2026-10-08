"""Unit tests for BM25 and hybrid RRF retrieval."""

import pytest
from src.embeddings.embedding_service import embedding_service
from src.ingestion.metadata import Chunk
from src.retrieval.retriever import BM25Index, HybridRetriever
from src.storage.vector_store import InMemoryVectorStore


def test_bm25_keyword_matching():
    index = BM25Index()
    c1 = Chunk(chunk_id="c1", doc_id="d1", doc_title="D1", text="Circuit breaker with 50% error rate threshold", chunk_index=0, char_start=0, char_end=45, token_count_estimate=10)
    c2 = Chunk(chunk_id="c2", doc_id="d2", doc_title="D2", text="Token bucket rate limiter with 500 burst capacity", chunk_index=0, char_start=0, char_end=49, token_count_estimate=11)

    index.index_chunks([c1, c2])
    res = index.search("circuit breaker threshold", top_k=2)

    assert len(res) >= 1
    assert res[0][0].chunk_id == "c1"
    assert "circuit" in res[0][2]


def test_hybrid_rrf_fusion():
    store = InMemoryVectorStore()
    retriever = HybridRetriever(store)

    c1 = Chunk(chunk_id="c1", doc_id="d1", doc_title="Postgres", text="PgBouncer operates in transaction pooling mode for connections", chunk_index=0, char_start=0, char_end=60, token_count_estimate=12)
    c2 = Chunk(chunk_id="c2", doc_id="d2", doc_title="Redis", text="Redis Cluster handles automatic failover across 3 replicas", chunk_index=0, char_start=0, char_end=58, token_count_estimate=11)

    chunks = [c1, c2]
    embs = embedding_service.embed_batch([c.text for c in chunks])
    store.add_chunks(chunks, embs)
    retriever.sync_bm25_index(chunks)

    results = retriever.retrieve("PgBouncer transaction pooling", top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[0].fused_score > results[1].fused_score
