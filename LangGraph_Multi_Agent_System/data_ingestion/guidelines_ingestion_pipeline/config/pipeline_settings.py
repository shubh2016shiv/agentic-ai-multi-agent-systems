"""
Pipeline configuration using Pydantic BaseSettings.

All configuration values are loaded from environment variables with the prefix
GUIDELINES_PIPELINE_. Sensitive values (mongodb_uri) have no defaults and must
be provided via environment variables or .env file.
"""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineSettings(BaseSettings):
    """Configuration for the guidelines ingestion pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GUIDELINES_PIPELINE_",
        extra="ignore",
    )

    # -- Chunking -----------------------------------------------------------------
    child_chunk_max_tokens: int = 400
    child_chunk_min_tokens: int = 50
    # 20% overlap — balances context continuity vs. index bloat. Empirically tuned
    # on clinical guideline text where ~20% overlap minimises retrieval boundary
    # artefacts without excessive duplication.
    child_chunk_overlap_ratio: float = 0.20
    parent_chunk_max_tokens: int = 1500

    # -- Embedding ----------------------------------------------------------------
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_batch_size: int = 32
    embedding_model_max_tokens: int = 8192
    # Safety margin prevents edge-case truncation at model's hard limit.
    embedding_token_safety_margin: float = 0.90

    # -- Deduplication ------------------------------------------------------------
    # 0.97 cosine similarity catches near-duplicates (minor formatting diffs)
    # without over-filtering genuinely distinct clinical paragraphs.
    semantic_dedup_similarity_threshold: float = 0.97
    semantic_dedup_sample_chunks: int = 3

    # -- Retry --------------------------------------------------------------------
    max_embedding_retries: int = 5
    retry_base_delay_seconds: float = 1.0
    retry_max_delay_seconds: float = 30.0

    # -- ChromaDB -----------------------------------------------------------------
    chroma_persist_path: str = "./chroma_db"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection_name: str = "medical_guidelines_v1"
    chroma_write_batch_size: int = 50
    chroma_hnsw_construction_ef: int = 200
    chroma_hnsw_m: int = 48

    # -- MongoDB ------------------------------------------------------------------
    mongodb_uri: str
    mongodb_database: str = "guidelines_pipeline"
    mongodb_registry_collection: str = "ingestion_jobs"
    mongodb_parent_chunks_collection: str = "parent_chunks"

    # -- Vision LLM ---------------------------------------------------------------
    vision_llm_timeout_seconds: int = 60
    vision_llm_model: str = "claude-sonnet-4-20250514"
    figure_crop_dpi: int = 300

    # -- Parsing ------------------------------------------------------------------
    docling_min_confidence_score: float = 0.70

    # -- Pipeline provenance ------------------------------------------------------
    pipeline_version: str = "v1.0.0"
    failed_pdfs_directory: str = "./failed_queue"
    wal_file_path: str = "./wal.jsonl"

    # -- Sanitisation patterns (JACC journal artefacts) ---------------------------
    header_footer_strip_patterns: List[str] = [
        r"JACC VOL\.\s+\d+,\s+NO\.\s+\d+,\s+\d{4}",
        r"^\d{3,4}$"
    ]
