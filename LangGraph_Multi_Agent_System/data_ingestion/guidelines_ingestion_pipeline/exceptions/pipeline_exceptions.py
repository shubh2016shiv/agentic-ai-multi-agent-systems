"""
Pipeline exception hierarchy — public re-exports.

All exception classes are defined in ``domain.exceptions`` (the canonical
location inside the domain boundary).  This module re-exports every class
so that infrastructure and application code can continue to import from
``exceptions.pipeline_exceptions`` without change.
"""

# Canonical definitions live in the domain layer.
from ..domain.exceptions import (                      # noqa: F401
    PipelineBaseException,
    # Document-level
    DocumentIngestionError,
    PDFParseError,
    PDFCorruptedError,
    PDFPasswordProtectedError,
    PDFNoTextLayerError,
    DocumentAlreadyIngestedException,
    FigureDescriptionError,
    VisionLLMTimeoutError,
    VisionLLMQuotaExceededError,
    # Chunking
    ChunkingError,
    ChunkTokenLimitExceededError,
    EmptyChunkError,
    # Embedding
    EmbeddingError,
    EmbeddingModelUnavailableError,
    EmbeddingBatchFailureError,
    EmbeddingTokenOverflowError,
    # Vector store
    VectorStoreError,
    VectorStoreWriteError,
    VectorStoreDiskFullError,
    # Registry
    RegistryError,
    RegistryConnectionError,
    RegistryJobNotFoundError,
)
