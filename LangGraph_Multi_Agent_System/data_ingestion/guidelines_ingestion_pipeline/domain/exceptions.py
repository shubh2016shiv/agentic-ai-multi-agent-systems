"""
Domain-layer exceptions raised and caught by domain services.

These live INSIDE the domain boundary so that no domain service needs to
import from the top-level ``exceptions/`` package.  The top-level
``exceptions.pipeline_exceptions`` module re-exports these and adds
infrastructure-specific exception subclasses.
"""

from typing import List, Optional


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class PipelineBaseException(Exception):
    """Base exception for all pipeline errors."""

    def __init__(self, message: str, **context):
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
        return f"{self.__class__.__name__}: {self.message} ({context_str})"


# ---------------------------------------------------------------------------
# Document-level
# ---------------------------------------------------------------------------

class DocumentIngestionError(PipelineBaseException):
    """Base exception for document-level ingestion failures."""
    pass


class PDFParseError(DocumentIngestionError):
    """Raised when PDF parsing fails."""

    def __init__(self, doc_id: str, pdf_name: str, cause: Optional[Exception] = None):
        message = f"Failed to parse PDF '{pdf_name}'"
        super().__init__(message, doc_id=doc_id, pdf_name=pdf_name, cause=str(cause) if cause else None)
        self.doc_id = doc_id
        self.pdf_name = pdf_name
        self.cause = cause


class PDFCorruptedError(PDFParseError):
    """Raised when PDF file is corrupted or malformed."""
    pass


class PDFPasswordProtectedError(PDFParseError):
    """Raised when PDF requires password authentication."""
    pass


class PDFNoTextLayerError(PDFParseError):
    """Raised when PDF has no embedded text layer (OCR required)."""
    pass


class DocumentAlreadyIngestedException(DocumentIngestionError):
    """Raised when attempting to ingest an already-ingested document."""

    def __init__(self, doc_id: str):
        message = f"Document already ingested: {doc_id}"
        super().__init__(message, doc_id=doc_id)
        self.doc_id = doc_id


class FigureDescriptionError(DocumentIngestionError):
    """Raised when figure/flowchart description generation fails."""

    def __init__(self, figure_id: str, cause: Optional[Exception] = None):
        message = f"Failed to describe figure '{figure_id}'"
        super().__init__(message, figure_id=figure_id, cause=str(cause) if cause else None)
        self.figure_id = figure_id
        self.cause = cause


class VisionLLMTimeoutError(FigureDescriptionError):
    """Raised when vision LLM call times out."""
    pass


class VisionLLMQuotaExceededError(FigureDescriptionError):
    """Raised when vision LLM API quota is exceeded."""
    pass


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class ChunkingError(PipelineBaseException):
    """Base exception for chunking-related failures."""
    pass


class ChunkTokenLimitExceededError(ChunkingError):
    """Raised when a chunk exceeds the maximum token limit."""

    def __init__(self, chunk_id: str, token_count: int, limit: int):
        message = f"Chunk '{chunk_id}' has {token_count} tokens (limit: {limit})"
        super().__init__(message, chunk_id=chunk_id, token_count=token_count, limit=limit)
        self.chunk_id = chunk_id
        self.token_count = token_count
        self.limit = limit


class EmptyChunkError(ChunkingError):
    """Raised when a chunk contains no meaningful content."""

    def __init__(self, chunk_id: str):
        message = f"Chunk '{chunk_id}' is empty or below minimum token threshold"
        super().__init__(message, chunk_id=chunk_id)
        self.chunk_id = chunk_id


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class EmbeddingError(PipelineBaseException):
    """Base exception for embedding-related failures."""
    pass


class EmbeddingModelUnavailableError(EmbeddingError):
    """Raised when the embedding model is unavailable or down."""

    def __init__(self, model_name: str, cause: Optional[Exception] = None):
        message = f"Embedding model '{model_name}' is unavailable"
        super().__init__(message, model_name=model_name, cause=str(cause) if cause else None)
        self.model_name = model_name
        self.cause = cause


class EmbeddingBatchFailureError(EmbeddingError):
    """Raised when batch embedding fails for specific indices."""

    def __init__(self, failed_indices: List[int], cause: Optional[Exception] = None):
        message = f"Embedding failed for {len(failed_indices)} chunks at indices: {failed_indices}"
        super().__init__(message, failed_indices=failed_indices, cause=str(cause) if cause else None)
        self.failed_indices = failed_indices
        self.cause = cause


class EmbeddingTokenOverflowError(EmbeddingError):
    """Raised when chunk token count exceeds embedding model's limit."""

    def __init__(self, chunk_id: str, token_count: int, model_limit: int):
        message = f"Chunk '{chunk_id}' has {token_count} tokens (model limit: {model_limit})"
        super().__init__(message, chunk_id=chunk_id, token_count=token_count, model_limit=model_limit)
        self.chunk_id = chunk_id
        self.token_count = token_count
        self.model_limit = model_limit


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

class VectorStoreError(PipelineBaseException):
    """Base exception for vector store failures."""
    pass


class VectorStoreWriteError(VectorStoreError):
    """Raised when writing to vector store fails."""

    def __init__(self, chunk_ids: List[str], cause: Optional[Exception] = None):
        message = f"Failed to write {len(chunk_ids)} chunks to vector store"
        super().__init__(message, chunk_ids=chunk_ids, cause=str(cause) if cause else None)
        self.chunk_ids = chunk_ids
        self.cause = cause


class VectorStoreDiskFullError(VectorStoreError):
    """Raised when vector store disk is full."""

    def __init__(self, available_space: Optional[int] = None):
        message = "Vector store disk is full"
        super().__init__(message, available_space=available_space)
        self.available_space = available_space


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class RegistryError(PipelineBaseException):
    """Base exception for document registry failures."""
    pass


class RegistryConnectionError(RegistryError):
    """Raised when connection to document registry fails."""

    def __init__(self, registry_uri: str, cause: Optional[Exception] = None):
        message = f"Failed to connect to document registry at '{registry_uri}'"
        super().__init__(message, registry_uri=registry_uri, cause=str(cause) if cause else None)
        self.registry_uri = registry_uri
        self.cause = cause


class RegistryJobNotFoundError(RegistryError):
    """Raised when querying for a non-existent job."""

    def __init__(self, job_id: str):
        message = f"Job not found in registry: {job_id}"
        super().__init__(message, job_id=job_id)
        self.job_id = job_id
