"""Unit tests for embedding service and cosine similarity."""

import pytest
from src.embeddings.embedding_service import DeterministicSemanticEmbedder, cosine_similarity, embedding_service


def test_deterministic_embedding_consistency():
    embedder = DeterministicSemanticEmbedder(dimension=128)
    text = "Enterprise rate limiting architecture with Redis cluster"
    v1 = embedder.embed_text(text)
    v2 = embedder.embed_text(text)

    assert len(v1) == 128
    assert v1 == v2  # 100% deterministic identical outputs


def test_semantic_similarity_ranking():
    embedder = DeterministicSemanticEmbedder(dimension=384)
    q = embedder.embed_text("PostgreSQL pgvector indexing performance")
    target = embedder.embed_text("HNSW and IVFFlat vector indexing in PostgreSQL")
    unrelated = embedder.embed_text("Cooking recipes for Italian pasta carbonara")

    sim_target = cosine_similarity(q, target)
    sim_unrelated = cosine_similarity(q, unrelated)

    assert sim_target > sim_unrelated
    assert sim_target > 0.3


def test_cosine_similarity_edge_cases():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v2)) == 1.0

    v_orth = [0.0, 1.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v_orth)) == 0.0

    v_zero = [0.0, 0.0, 0.0]
    assert cosine_similarity(v1, v_zero) == 0.0
