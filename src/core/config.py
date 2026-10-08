"""Configuration settings for Production RAG & AI Evaluation System."""

from enum import Enum
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    TEST = "test"
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class SimilarityMetric(str, Enum):
    COSINE = "cosine"
    L2 = "l2"
    DOT_PRODUCT = "dot"


class Settings(BaseSettings):
    """System-wide configuration with environment variable override support."""

    app_name: str = "Production RAG & Evaluation Engine"
    app_version: str = "1.0.0"
    environment: Environment = Environment.TEST
    host: str = "0.0.0.0"
    port: int = 8000

    # Ingestion & Chunking
    chunk_size: int = 400
    chunk_overlap: int = 80
    min_chunk_length: int = 40

    # Embeddings & Vector Search
    embedding_dimension: int = 384
    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE
    top_k: int = 5
    similarity_threshold: float = 0.45
    hybrid_dense_weight: float = 0.65
    hybrid_sparse_weight: float = 0.35
    rrf_k: int = 60

    # Generation & Guardrails
    max_context_tokens: int = 2048
    temperature: float = 0.1
    enable_injection_defense: bool = True
    enable_pii_redaction: bool = True
    strict_groundedness: bool = True

    # Observability
    enable_tracing: bool = True
    trace_sample_rate: float = 1.0
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Database
    database_url: Optional[str] = None
    use_in_memory_vector_store: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
