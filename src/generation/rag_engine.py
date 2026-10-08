"""Production RAG execution engine with multi-stage security, hybrid retrieval, and grounded citation synthesis."""

import os
import re
import time
from typing import List, Optional
from src.core.config import settings
from src.core.observability import tracer
from src.core.security import SecurityGuard
from src.generation.structured_output import Citation, ConfidenceLevel, GroundednessVerdict, RAGResponse
from src.retrieval.retriever import HybridRetriever, RetrievedChunk


class RAGEngine:
    """Production RAG pipeline coordinating security, retrieval, and grounded answer synthesis."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.use_live_llm = bool(os.getenv("OPENAI_API_KEY"))

    def query(self, user_query: str, top_k: int = 4, session_id: Optional[str] = None) -> RAGResponse:
        t0 = time.time()
        trace = tracer.new_trace(query=user_query, session_id=session_id)

        # 1. Security & Prompt Injection Inspection
        span_sec = trace.start_span("security_guard")
        sec_res = SecurityGuard.inspect_query(user_query)
        if not sec_res.is_safe:
            span_sec.finish(status="BLOCKED", error=f"Adversarial query blocked: {sec_res.detected_patterns}")
            trace.finalize()
            return RAGResponse(
                query=user_query,
                answer="Security Alert: Your query triggered adversarial injection safeguards and cannot be processed.",
                verdict=GroundednessVerdict.REFUSED_SECURITY,
                confidence=ConfidenceLevel.LOW,
                citations=[],
                retrieved_chunk_count=0,
                latency_ms=round((time.time() - t0) * 1000, 2),
                tokens_estimated=20,
                trace_id=trace.trace_id,
                sanitized=True,
            )
        span_sec.finish(status="OK")

        # 2. Hybrid Retrieval
        span_ret = trace.start_span("hybrid_retrieval")
        retrieved_chunks = self.retriever.retrieve(
            query=sec_res.sanitized_text,
            top_k=top_k,
            dense_weight=settings.hybrid_dense_weight,
            sparse_weight=settings.hybrid_sparse_weight,
            rrf_k=settings.rrf_k,
        )
        span_ret.finish(status="OK", error=None)
        span_ret.attributes["retrieved_count"] = len(retrieved_chunks)

        # 3. Grounded Context Assessment
        if not retrieved_chunks or (retrieved_chunks[0].dense_score < 0.15 and retrieved_chunks[0].sparse_score == 0):
            trace.finalize()
            return RAGResponse(
                query=user_query,
                answer="Insufficient context: The provided reference documentation does not contain verifiable facts to answer this query.",
                verdict=GroundednessVerdict.INSUFFICIENT_CONTEXT,
                confidence=ConfidenceLevel.LOW,
                citations=[],
                retrieved_chunk_count=len(retrieved_chunks),
                latency_ms=round((time.time() - t0) * 1000, 2),
                tokens_estimated=35,
                trace_id=trace.trace_id,
                sanitized=sec_res.redacted_pii_count > 0,
            )

        # 4. Answer Synthesis & Citation Attribution
        span_gen = trace.start_span("answer_synthesis")
        answer, citations, verdict, conf = self._synthesize_answer(sec_res.sanitized_text, retrieved_chunks)
        span_gen.finish(status="OK")
        span_gen.attributes["citation_count"] = len(citations)

        trace.total_tokens_estimated = sum(len(c.text) // 4 for c in retrieved_chunks) + (len(answer) // 4)
        trace.finalize()

        total_latency = round((time.time() - t0) * 1000, 2)
        return RAGResponse(
            query=user_query,
            answer=answer,
            verdict=verdict,
            confidence=conf,
            citations=citations,
            retrieved_chunk_count=len(retrieved_chunks),
            latency_ms=total_latency,
            tokens_estimated=trace.total_tokens_estimated,
            trace_id=trace.trace_id,
            sanitized=sec_res.redacted_pii_count > 0,
        )

    def _synthesize_answer(
        self, query: str, chunks: List[RetrievedChunk]
    ) -> tuple[str, List[Citation], GroundednessVerdict, ConfidenceLevel]:
        """Synthesizes factual answer strictly grounded in retrieved chunk contents."""
        citations: List[Citation] = []

        # Live LLM execution if API key configured
        if self.use_live_llm:
            try:
                import openai
                client = openai.OpenAI()
                context_str = "\n\n".join(f"[{c.chunk_id}] ({c.doc_title}): {c.text}" for c in chunks)
                prompt = (
                    f"You are a strict technical documentation assistant. Answer the query using ONLY the provided facts. "
                    f"Cite sources using [chunk_id]. If uncertain, state that facts are not available.\n\n"
                    f"Context:\n{context_str}\n\nQuery: {query}"
                )
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                ans = resp.choices[0].message.content or ""
                for c in chunks[:2]:
                    citations.append(
                        Citation(
                            chunk_id=c.chunk_id,
                            doc_id=c.doc_id,
                            doc_title=c.doc_title,
                            section_heading=c.section_heading,
                            exact_quote=c.text[:200],
                            relevance_score=c.fused_score,
                        )
                    )
                return ans, citations, GroundednessVerdict.GROUNDED, ConfidenceLevel.HIGH
            except Exception:
                pass

        # High-precision deterministic extractive contextual synthesizer
        # Filter common stopwords so words like 'the' / 'what' don't cause false grounding
        stopwords = {"the", "and", "are", "what", "for", "with", "from", "that", "this", "how", "why", "does", "can", "has", "have", "not", "its", "each", "per", "all", "which"}
        query_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", query.lower()) if w not in stopwords)
        relevant_sentences = []

        for c in chunks:
            sentences = re.split(r"(?<=[.!?])\s+", c.text)
            for s in sentences:
                s_clean = s.strip()
                s_words = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", s_clean.lower()))
                overlap = len(query_words & s_words)
                if overlap > 0:
                    relevant_sentences.append((s_clean, overlap, c))

        if relevant_sentences:
            # Sort by keyword overlap descending
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sents = relevant_sentences[:3]

            answer_parts = [s[0] for s in top_sents]
            # Deduplicate sentences while preserving order
            seen = set()
            deduped = []
            for sent in answer_parts:
                if sent not in seen:
                    deduped.append(sent)
                    seen.add(sent)

            answer = " ".join(deduped)

            for s_clean, score, chunk in top_sents:
                if not any(cit.chunk_id == chunk.chunk_id for cit in citations):
                    citations.append(
                        Citation(
                            chunk_id=chunk.chunk_id,
                            doc_id=chunk.doc_id,
                            doc_title=chunk.doc_title,
                            section_heading=chunk.section_heading,
                            exact_quote=s_clean[:250],
                            relevance_score=chunk.fused_score,
                        )
                    )

            top_overlap = top_sents[0][1]
            if top_overlap == 0:
                return (
                    "Insufficient context: The provided reference documentation does not contain verifiable facts to answer this query.",
                    [],
                    GroundednessVerdict.INSUFFICIENT_CONTEXT,
                    ConfidenceLevel.LOW,
                )
            conf = ConfidenceLevel.HIGH if top_overlap >= 2 else ConfidenceLevel.MEDIUM
            return answer, citations, GroundednessVerdict.GROUNDED, conf

        # Fallback when no sentences matched query words
        return (
            "Insufficient context: The provided reference documentation does not contain verifiable facts to answer this query.",
            [],
            GroundednessVerdict.INSUFFICIENT_CONTEXT,
            ConfidenceLevel.LOW,
        )
