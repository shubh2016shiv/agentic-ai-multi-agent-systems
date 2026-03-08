"""
Pipeline configuration using Pydantic BaseSettings.

All configuration values are loaded from environment variables with the prefix
GUIDELINES_PIPELINE_. Sensitive values (mongodb_uri) have no defaults and must
be provided via environment variables or .env file.
"""

from typing import List
from pydantic_settings import BaseSettings


class PipelineSettings(BaseSettings):
    """Configuration for the guidelines ingestion pipeline."""

    child_chunk_max_tokens: int = 400
    child_chunk_min_tokens: int = 50
    child_chunk_overlap_ratio: float = 0.20
    parent_chunk_max_tokens: int = 1500

    embedding_model_name: str = "BAAI/bge-m3"
    embedding_batch_size: int = 32
    embedding_model_max_tokens: int = 8192
    embedding_token_safety_margin: float = 0.90

    semantic_dedup_similarity_threshold: float = 0.97
    semantic_dedup_sample_chunks: int = 3

    max_embedding_retries: int = 5
    retry_base_delay_seconds: float = 1.0
    retry_max_delay_seconds: float = 30.0

    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection_name: str = "medical_guidelines_v1"
    chroma_write_batch_size: int = 50
    chroma_hnsw_construction_ef: int = 200
    chroma_hnsw_m: int = 48

    mongodb_uri: str
    mongodb_database: str = "guidelines_pipeline"
    mongodb_registry_collection: str = "ingestion_jobs"
    mongodb_parent_chunks_collection: str = "parent_chunks"

    vision_llm_timeout_seconds: int = 60
    vision_llm_model: str = "claude-sonnet-4-20250514"
    figure_crop_dpi: int = 300

    docling_min_confidence_score: float = 0.70

    pipeline_version: str = "v1.0.0"
    failed_pdfs_directory: str = "./failed_queue"
    wal_file_path: str = "./wal.jsonl"

    header_footer_strip_patterns: List[str] = [
        r"JACC VOL\.\s+\d+,\s+NO\.\s+\d+,\s+\d{4}",
        r"^\d{3,4}$"
    ]

    class Config:
        env_file = ".env"
        env_prefix = "GUIDELINES_PIPELINE_"
        extra = "ignore"
