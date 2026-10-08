"""Hybrid search retriever combining dense semantic vector search with BM25 sparse keyword search via Reciprocal Rank Fusion (RRF)."""

from collections import defaultdict
import math
import re
from typing import Dict, List, Set, Tuple
from pydantic import BaseModel
from src.core.config import settings
from src.embeddings.embedding_service import embedding_service
from src.ingestion.metadata import Chunk
from src.storage.vector_store import BaseVectorStore, SearchResult


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    section_heading: str | None = None
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0
    final_rank: int = 1
    matched_terms: List[str] = []


class BM25Index:
    """In-memory BM25 Okapi keyword search implementation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: Dict[str, int] = {}
        self.avg_dl: float = 0.0
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.term_freq: Dict[str, Dict[str, int]] = {}
        self.total_docs: int = 0
        self.chunks_by_id: Dict[str, Chunk] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())

    def index_chunks(self, chunks: List[Chunk]) -> None:
        self.chunks_by_id = {c.chunk_id: c for c in chunks}
        self.total_docs = len(chunks)
        if not self.total_docs:
            return

        total_length = 0
        for chunk in chunks:
            tokens = self.tokenize(chunk.text)
            length = len(tokens)
            self.doc_len[chunk.chunk_id] = length
            total_length += length

            tf: Dict[str, int] = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            self.term_freq[chunk.chunk_id] = tf

            for unique_t in set(tokens):
                self.doc_freq[unique_t] += 1

        self.avg_dl = total_length / self.total_docs if self.total_docs > 0 else 1.0

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float, List[str]]]:
        if not self.total_docs:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[str, float] = defaultdict(float)
        matched_terms: Dict[str, Set[str]] = defaultdict(set)

        for token in query_tokens:
            df = self.doc_freq.get(token, 0)
            if df == 0:
                continue

            # Standard Robertson-Spärck Jones IDF
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)

            for chunk_id, tf_map in self.term_freq.items():
                tf = tf_map.get(token, 0)
                if tf > 0:
                    dl = self.doc_len[chunk_id]
                    num = tf * (self.k1 + 1)
                    denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avg_dl))
                    scores[chunk_id] += idf * (num / denom)
                    matched_terms[chunk_id].add(token)

        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for chunk_id, score in sorted_chunks[:top_k]:
            results.append((self.chunks_by_id[chunk_id], score, sorted(list(matched_terms[chunk_id]))))

        return results


class HybridRetriever:
    """Orchestrates dense vector search + BM25 sparse keyword search via Reciprocal Rank Fusion."""

    def __init__(self, vector_store: BaseVectorStore):
        self.vector_store = vector_store
        self.bm25_index = BM25Index()

    def sync_bm25_index(self, all_chunks: List[Chunk]) -> None:
        self.bm25_index.index_chunks(all_chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.65,
        sparse_weight: float = 0.35,
        rrf_k: int = 60,
    ) -> List[RetrievedChunk]:
        """Executes hybrid retrieval using reciprocal rank fusion."""
        # 1. Dense Vector Search
        query_emb = embedding_service.embed_text(query)
        dense_results: List[SearchResult] = self.vector_store.search(
            query_embedding=query_emb,
            top_k=top_k * 2,
            score_threshold=0.0,
        )

        # 2. Sparse BM25 Search
        sparse_results = self.bm25_index.search(query, top_k=top_k * 2)

        # 3. Reciprocal Rank Fusion (RRF)
        dense_ranks = {res.chunk.chunk_id: (idx + 1, res.score, res.chunk) for idx, res in enumerate(dense_results)}
        sparse_ranks = {chunk.chunk_id: (idx + 1, score, chunk, matches) for idx, (chunk, score, matches) in enumerate(sparse_results)}

        all_chunk_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        fused_scores: Dict[str, float] = {}

        for cid in all_chunk_ids:
            score = 0.0
            if cid in dense_ranks:
                r_dense = dense_ranks[cid][0]
                score += dense_weight * (1.0 / (rrf_k + r_dense))
            if cid in sparse_ranks:
                r_sparse = sparse_ranks[cid][0]
                score += sparse_weight * (1.0 / (rrf_k + r_sparse))
            fused_scores[cid] = score

        # Sort descending by fused RRF score
        sorted_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)[:top_k]

        final_chunks: List[RetrievedChunk] = []
        for rank, cid in enumerate(sorted_ids, start=1):
            chunk = dense_ranks[cid][2] if cid in dense_ranks else sparse_ranks[cid][2]
            d_score = dense_ranks[cid][1] if cid in dense_ranks else 0.0
            s_score = sparse_ranks[cid][1] if cid in sparse_ranks else 0.0
            matches = sparse_ranks[cid][3] if cid in sparse_ranks else []

            final_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    text=chunk.text,
                    section_heading=chunk.section_heading,
                    dense_score=round(d_score, 4),
                    sparse_score=round(s_score, 4),
                    fused_score=round(fused_scores[cid], 6),
                    final_rank=rank,
                    matched_terms=matches,
                )
            )

        return final_chunks
