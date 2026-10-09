"""Evaluation benchmark harness for executing comprehensive 100-case RAG evaluation."""

from collections import defaultdict
import json
import os
import time
from typing import Dict, List, Tuple
from src.embeddings.embedding_service import embedding_service
from src.evals.dataset import REFERENCE_CORPUS, EvalCase, generate_100_eval_cases
from src.evals.metrics import BenchmarkSummary, CaseEvaluationResult, MetricsCalculator
from src.generation.rag_engine import RAGEngine
from src.ingestion.chunker import RecursiveSemanticChunker
from src.ingestion.metadata import Chunk, Document
from src.retrieval.retriever import HybridRetriever
from src.storage.vector_store import InMemoryVectorStore


class EvaluationHarness:
    """Executes offline deterministic RAG benchmark across 100 ground-truth evaluation cases."""

    def __init__(self):
        self.vector_store = InMemoryVectorStore()
        self.retriever = HybridRetriever(self.vector_store)
        self.rag_engine = RAGEngine(self.retriever)
        self.chunker = RecursiveSemanticChunker(target_chunk_size=400, overlap=80)
        self.all_chunks: List[Chunk] = []

    def setup_corpus(self) -> int:
        """Ingests and indexes the reference corpus documents."""
        self.vector_store.clear()
        self.all_chunks.clear()

        for doc_dict in REFERENCE_CORPUS:
            doc = Document(
                doc_id=doc_dict["doc_id"],
                title=doc_dict["title"],
                content=doc_dict["content"],
                source="corpus",
            )
            chunks = self.chunker.chunk_document(doc)
            self.all_chunks.extend(chunks)

        embeddings = embedding_service.embed_batch([c.text for c in self.all_chunks])
        self.vector_store.add_chunks(self.all_chunks, embeddings)
        self.retriever.sync_bm25_index(self.all_chunks)
        return len(self.all_chunks)

    def run_benchmark(self, eval_cases: List[EvalCase] = None) -> Tuple[BenchmarkSummary, List[CaseEvaluationResult]]:
        """Runs evaluation over cases and compiles aggregate benchmark metrics."""
        if not self.all_chunks:
            self.setup_corpus()

        cases = eval_cases or generate_100_eval_cases()
        results: List[CaseEvaluationResult] = []
        latencies: List[float] = []

        category_hits = defaultdict(lambda: {"total": 0, "hits_at_1": 0, "hits_at_3": 0, "mrr_sum": 0.0})

        for case in cases:
            t0 = time.time()
            rag_resp = self.rag_engine.query(user_query=case.query, top_k=5)
            lat_ms = (time.time() - t0) * 1000
            latencies.append(lat_ms)

            # Analyze retrieved chunks
            retrieved_doc_ids = [c.doc_id for c in self.retriever.retrieve(case.query, top_k=5)]
            rank = None
            if case.expected_doc_id in retrieved_doc_ids:
                rank = retrieved_doc_ids.index(case.expected_doc_id) + 1

            is_hit_1 = rank == 1
            is_hit_3 = rank is not None and rank <= 3
            is_hit_5 = rank is not None and rank <= 5
            recip_rank = MetricsCalculator.compute_mrr(rank)

            # Keyword and generation quality metrics
            matched_kw, kw_cov = MetricsCalculator.compute_keyword_overlap(rag_resp.answer, case.expected_keywords)
            retrieved_texts = [c.text for c in self.retriever.retrieve(case.query, top_k=3)]
            faithfulness = MetricsCalculator.compute_faithfulness(rag_resp.answer, retrieved_texts)
            relevance = MetricsCalculator.compute_answer_relevance(rag_resp.answer, case.ground_truth_answer)
            citation_prec = 1.0 if any(c.doc_id == case.expected_doc_id for c in rag_resp.citations) else 0.0

            failure_reason = None
            if not is_hit_3:
                failure_reason = f"Target doc '{case.expected_doc_id}' not in top 3 retrieved: {retrieved_doc_ids}"
            elif kw_cov < 0.5:
                failure_reason = f"Keyword coverage low ({matched_kw}/{len(case.expected_keywords)})"

            res = CaseEvaluationResult(
                case_id=case.case_id,
                query=case.query,
                category=case.category,
                is_hit_at_1=is_hit_1,
                is_hit_at_3=is_hit_3,
                is_hit_at_5=is_hit_5,
                reciprocal_rank=recip_rank,
                retrieved_rank=rank,
                expected_doc_id=case.expected_doc_id,
                retrieved_doc_ids=retrieved_doc_ids,
                matched_keyword_count=matched_kw,
                total_keyword_count=len(case.expected_keywords),
                faithfulness_score=faithfulness,
                answer_relevance_score=relevance,
                citation_precision=citation_prec,
                latency_ms=round(lat_ms, 2),
                failure_reason=failure_reason,
            )
            results.append(res)

            cat_stats = category_hits[case.category]
            cat_stats["total"] += 1
            if is_hit_1:
                cat_stats["hits_at_1"] += 1
            if is_hit_3:
                cat_stats["hits_at_3"] += 1
            cat_stats["mrr_sum"] += recip_rank

        total = len(cases)
        latencies_sorted = sorted(latencies)
        p95_idx = int(0.95 * total)
        p95_latency = round(latencies_sorted[p95_idx], 2) if latencies_sorted else 0.0

        cat_breakdown = {}
        for cat, stats in category_hits.items():
            tot = stats["total"]
            cat_breakdown[cat] = {
                "count": tot,
                "hit_rate_at_1": round(stats["hits_at_1"] / tot, 4),
                "hit_rate_at_3": round(stats["hits_at_3"] / tot, 4),
                "mrr": round(stats["mrr_sum"] / tot, 4),
            }

        summary = BenchmarkSummary(
            total_cases=total,
            hit_rate_at_1=round(sum(1 for r in results if r.is_hit_at_1) / total, 4),
            hit_rate_at_3=round(sum(1 for r in results if r.is_hit_at_3) / total, 4),
            hit_rate_at_5=round(sum(1 for r in results if r.is_hit_at_5) / total, 4),
            mean_reciprocal_rank=round(sum(r.reciprocal_rank for r in results) / total, 4),
            avg_keyword_coverage=round(sum(r.matched_keyword_count / r.total_keyword_count for r in results) / total, 4),
            avg_faithfulness=round(sum(r.faithfulness_score for r in results) / total, 4),
            avg_answer_relevance=round(sum(r.answer_relevance_score for r in results) / total, 4),
            avg_citation_precision=round(sum(r.citation_precision for r in results) / total, 4),
            avg_latency_ms=round(sum(latencies) / total, 2),
            p95_latency_ms=p95_latency,
            failure_case_count=sum(1 for r in results if r.failure_reason is not None),
            category_breakdown=cat_breakdown,
        )

        return summary, results

    def export_report(self, summary: BenchmarkSummary, results: List[CaseEvaluationResult], output_dir: str = "benchmarks") -> str:
        """Exports benchmark metrics to JSON and Markdown reports."""
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "evaluation_report.json")
        md_path = os.path.join(output_dir, "BENCHMARK_REPORT.md")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary.model_dump(),
                    "results": [r.model_dump() for r in results],
                },
                f,
                indent=2,
            )

        def _status(ok: bool) -> str:
            return "PASS" if ok else "FAIL"

        md_content = f"""# Production RAG System — Benchmark Evaluation Report

**Total Test Cases:** {summary.total_cases}  
**Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  
**Architecture:** Hybrid Dense-Sparse Retrieval (deterministic hashed-token embeddings + BM25) with Reciprocal Rank Fusion (RRF)  
**Corpus:** {len(REFERENCE_CORPUS)} short self-written reference documents; the {summary.total_cases} queries and expected answers were also written by the author (see `src/evals/dataset.py`). Hit-Rate is measured at document level, so a random ranking already scores ~{100 / len(REFERENCE_CORPUS):.0f}% Hit-Rate @ 1.  
**Faithfulness** is a lexical proxy (share of answer tokens found in the retrieved context), not an LLM- or human-judged score.  
**Answer relevance** (token Jaccard vs. ground-truth answer): {summary.avg_answer_relevance * 100:.2f}%

---

## 1. Executive Metrics Summary

| Metric | Measured Value | Production Target | Status |
| :--- | :---: | :---: | :---: |
| **Hit-Rate @ 1** | **{summary.hit_rate_at_1 * 100:.2f}%** | $\\ge 80.0\\%$ | **{_status(summary.hit_rate_at_1 >= 0.80)}** |
| **Hit-Rate @ 3** | **{summary.hit_rate_at_3 * 100:.2f}%** | $\\ge 95.0\\%$ | **{_status(summary.hit_rate_at_3 >= 0.95)}** |
| **Hit-Rate @ 5** | **{summary.hit_rate_at_5 * 100:.2f}%** | $\\ge 98.0\\%$ | **{_status(summary.hit_rate_at_5 >= 0.98)}** |
| **Mean Reciprocal Rank (MRR)** | **{summary.mean_reciprocal_rank:.4f}** | $\\ge 0.8500$ | **{_status(summary.mean_reciprocal_rank >= 0.85)}** |
| **Avg Keyword Grounding** | **{summary.avg_keyword_coverage * 100:.2f}%** | $\\ge 85.0\\%$ | **{_status(summary.avg_keyword_coverage >= 0.85)}** |
| **Context Faithfulness** | **{summary.avg_faithfulness * 100:.2f}%** | $\\ge 90.0\\%$ | **{_status(summary.avg_faithfulness >= 0.90)}** |
| **Citation Precision** | **{summary.avg_citation_precision * 100:.2f}%** | $\\ge 95.0\\%$ | **{_status(summary.avg_citation_precision >= 0.95)}** |
| **Average Query Latency** | **{summary.avg_latency_ms:.2f} ms** | $\\le 25.0$ ms | **{_status(summary.avg_latency_ms <= 25.0)}** |
| **P95 Query Latency** | **{summary.p95_latency_ms:.2f} ms** | $\\le 50.0$ ms | **{_status(summary.p95_latency_ms <= 50.0)}** |

---

## 2. Category Performance Breakdown

| Category | Cases | Hit-Rate @ 1 | Hit-Rate @ 3 | MRR |
| :--- | :---: | :---: | :---: | :---: |
"""
        for cat, stats in summary.category_breakdown.items():
            md_content += f"| **{cat}** | {stats['count']} | {stats['hit_rate_at_1']*100:.1f}% | {stats['hit_rate_at_3']*100:.1f}% | {stats['mrr']:.4f} |\n"

        md_content += f"""
---

## 3. Failure Cases & Outliers
Total identified edge/failure cases: **{summary.failure_case_count}**

"""
        failures = [r for r in results if r.failure_reason]
        if failures:
            for f_case in failures[:10]:
                md_content += f"- **{f_case.case_id}** ({f_case.category}): *\"{f_case.query}\"* — Reason: `{f_case.failure_reason}`\n"
        else:
            md_content += "Zero retrieval failures detected across all 100 test cases.\n"

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return md_path
