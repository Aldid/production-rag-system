"""Evaluation metrics calculations for RAG retrieval and answer generation quality."""

from dataclasses import dataclass, field
import re
from typing import List, Optional, Set, Tuple
from pydantic import BaseModel


class CaseEvaluationResult(BaseModel):
    case_id: str
    query: str
    category: str
    is_hit_at_1: bool
    is_hit_at_3: bool
    is_hit_at_5: bool
    reciprocal_rank: float
    retrieved_rank: Optional[int] = None
    expected_doc_id: str
    retrieved_doc_ids: List[str]
    matched_keyword_count: int
    total_keyword_count: int
    faithfulness_score: float
    answer_relevance_score: float
    citation_precision: float
    latency_ms: float
    failure_reason: Optional[str] = None


class BenchmarkSummary(BaseModel):
    total_cases: int
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float
    mean_reciprocal_rank: float
    avg_keyword_coverage: float
    avg_faithfulness: float
    avg_answer_relevance: float
    avg_citation_precision: float
    avg_latency_ms: float
    p95_latency_ms: float
    failure_case_count: int
    category_breakdown: dict = {}


class MetricsCalculator:
    """Computes exact information retrieval and answer quality evaluation metrics."""

    @staticmethod
    def compute_mrr(rank: Optional[int]) -> float:
        if rank is None or rank <= 0:
            return 0.0
        return 1.0 / rank

    @staticmethod
    def compute_keyword_overlap(answer: str, expected_keywords: List[str]) -> Tuple[int, float]:
        if not expected_keywords:
            return 0, 1.0
        ans_lower = answer.lower()
        matched = 0
        for kw in expected_keywords:
            if kw.lower() in ans_lower:
                matched += 1
        return matched, matched / len(expected_keywords)

    @staticmethod
    def compute_faithfulness(answer: str, context_chunks_text: List[str]) -> float:
        """Evaluates what fraction of answer key terms are grounded in retrieved context."""
        ans_words = re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", answer.lower())
        if not ans_words:
            return 1.0

        combined_context = " ".join(context_chunks_text).lower()
        context_words = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", combined_context))

        grounded_count = sum(1 for w in ans_words if w in context_words)
        return round(grounded_count / len(ans_words), 4)

    @staticmethod
    def compute_answer_relevance(generated_answer: str, ground_truth: str) -> float:
        """Calculates token Jaccard similarity between generated answer and ground truth."""
        gen_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", generated_answer.lower()))
        truth_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", ground_truth.lower()))
        if not truth_tokens:
            return 1.0
        intersection = len(gen_tokens & truth_tokens)
        union = len(gen_tokens | truth_tokens)
        return round(intersection / union if union > 0 else 0.0, 4)
