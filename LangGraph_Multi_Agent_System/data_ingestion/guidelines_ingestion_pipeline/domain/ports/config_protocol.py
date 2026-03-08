"""
Domain-local configuration protocol.

Domain services depend on THIS protocol instead of ``config.pipeline_settings``,
satisfying the Dependency Inversion Principle.  The concrete ``PipelineSettings``
class (in ``config/``) satisfies this protocol without the domain importing it.
"""

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class PipelineConfigProtocol(Protocol):
    """Subset of pipeline settings visible to the domain layer."""

    # -- Chunking ----------------------------------------------------------------
    child_chunk_max_tokens: int
    child_chunk_min_tokens: int
    child_chunk_overlap_ratio: float
    parent_chunk_max_tokens: int

    # -- Embedding ---------------------------------------------------------------
    embedding_model_name: str
    embedding_batch_size: int

    # -- Deduplication -----------------------------------------------------------
    semantic_dedup_similarity_threshold: float
    semantic_dedup_sample_chunks: int

    # -- Sanitisation ------------------------------------------------------------
    header_footer_strip_patterns: List[str]

    # -- Provenance --------------------------------------------------------------
    pipeline_version: str
