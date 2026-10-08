"""Structured schemas for RAG query responses and citation attribution."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class GroundednessVerdict(str, Enum):
    GROUNDED = "GROUNDED"
    PARTIALLY_GROUNDED = "PARTIALLY_GROUNDED"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    REFUSED_SECURITY = "REFUSED_SECURITY"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    section_heading: Optional[str] = None
    exact_quote: str
    relevance_score: float = 0.0


class RAGResponse(BaseModel):
    query: str
    answer: str
    verdict: GroundednessVerdict
    confidence: ConfidenceLevel
    citations: List[Citation] = Field(default_factory=list)
    retrieved_chunk_count: int = 0
    latency_ms: float = 0.0
    tokens_estimated: int = 0
    trace_id: Optional[str] = None
    sanitized: bool = False
