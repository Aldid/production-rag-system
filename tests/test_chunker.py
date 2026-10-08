"""Unit tests for recursive semantic chunker."""

import pytest
from src.ingestion.chunker import RecursiveSemanticChunker
from src.ingestion.metadata import Document


def test_chunk_simple_markdown():
    doc = Document(
        doc_id="test_doc_1",
        title="Sample Document",
        content="# Section 1\n\nFirst paragraph content here with some explanation.\n\n# Section 2\n\nSecond paragraph content with more details.",
    )
    chunker = RecursiveSemanticChunker(target_chunk_size=150, overlap=30)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    assert chunks[0].doc_id == "test_doc_1"
    assert "Section 1" in chunks[0].text or chunks[0].section_heading == "# Section 1"
    assert chunks[0].token_count_estimate > 0
    assert chunks[0].sha256_hash != ""


def test_chunk_empty_document():
    doc = Document(doc_id="empty", title="Empty", content="   \n\n   ")
    chunker = RecursiveSemanticChunker()
    chunks = chunker.chunk_document(doc)
    assert chunks == []


def test_token_estimation():
    assert RecursiveSemanticChunker.estimate_tokens("Hello world") >= 1
    assert RecursiveSemanticChunker.estimate_tokens("A" * 400) == 100
