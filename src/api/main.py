"""Production FastAPI REST interface for RAG querying, document ingestion, and evaluation benchmarking."""

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from src.core.config import settings
from src.core.observability import tracer
from src.evals.harness import EvaluationHarness
from src.generation.rag_engine import RAGEngine
from src.generation.structured_output import RAGResponse
from src.ingestion.chunker import RecursiveSemanticChunker
from src.ingestion.metadata import Document
from src.retrieval.retriever import HybridRetriever
from src.storage.vector_store import InMemoryVectorStore


# Initialize application components
vector_store = InMemoryVectorStore()
retriever = HybridRetriever(vector_store)
rag_engine = RAGEngine(retriever)
chunker = RecursiveSemanticChunker(target_chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
harness = EvaluationHarness()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production RAG & Evaluation Engine with Hybrid Retrieval, Guardrails, and Strict Evals",
)


class IngestRequest(BaseModel):
    doc_id: str
    title: str
    content: str
    source: str = "api"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    chunks_created: int
    total_index_size: int
    status: str = "SUCCESS"


class QueryRequest(BaseModel):
    query: str
    top_k: int = 4
    session_id: Optional[str] = None


def initialize_default_corpus():
    harness.setup_corpus()
    vector_store.clear()
    for chunk in harness.all_chunks:
        emb = harness.vector_store._embeddings[chunk.chunk_id]
        vector_store.add_chunks([chunk], [emb])
    retriever.sync_bm25_index(harness.all_chunks)


# Pre-populate on module initialization
initialize_default_corpus()


@app.on_event("startup")
def startup_event():
    initialize_default_corpus()


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "HEALTHY",
        "version": settings.app_version,
        "indexed_chunks": vector_store.count(),
        "environment": settings.environment,
    }


@app.post("/api/v1/query", response_model=RAGResponse, tags=["RAG"])
def query_rag(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query string cannot be empty")
    return rag_engine.query(user_query=req.query, top_k=req.top_k, session_id=req.session_id)


@app.post("/api/v1/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest_document(req: IngestRequest):
    if not req.content.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document content cannot be empty")

    doc = Document(
        doc_id=req.doc_id,
        title=req.title,
        content=req.content,
        source=req.source,
        metadata=req.metadata,
    )
    chunks = chunker.chunk_document(doc)
    from src.embeddings.embedding_service import embedding_service
    embeddings = embedding_service.embed_batch([c.text for c in chunks])
    vector_store.add_chunks(chunks, embeddings)
    retriever.sync_bm25_index(vector_store.get_all_chunks())

    return IngestResponse(
        doc_id=req.doc_id,
        chunks_created=len(chunks),
        total_index_size=vector_store.count(),
    )


@app.post("/api/v1/evaluate", tags=["Evaluation"])
def run_evaluation_benchmark():
    summary, results = harness.run_benchmark()
    report_path = harness.export_report(summary, results)
    return {
        "summary": summary.model_dump(),
        "report_generated": report_path,
        "status": "EVALUATION_COMPLETE",
    }


@app.get("/api/v1/traces", tags=["Observability"])
def get_recent_traces(limit: int = 25):
    return {"traces": tracer.get_recent_traces(limit=limit)}
