"""
Three-layer deduplication service for ingestion pipeline.

Implements:
1. Document-level deduplication (SHA-256 hash check)
2. Chunk-level deduplication (content hash check)
3. Semantic near-duplicate detection (cosine similarity)
"""

from typing import List, Optional

import structlog

from ..models.chunk import ChildChunk
from ..models.document_metadata import DuplicateResolution
from ..models.ingestion_job import IngestionStatus
from ..models.parsed_document import ParsedDocument
from ..ports.document_registry_port import AbstractDocumentRegistry
from ..ports.vector_store_port import AbstractVectorStore
from ..ports.config_protocol import PipelineConfigProtocol


logger = structlog.get_logger(__name__)


class DeduplicationService:
    """
    Three-layer deduplication strategy for idempotent ingestion.
    
    Dependencies:
        - AbstractVectorStore (injected)
        - AbstractDocumentRegistry (injected)
        - PipelineConfigProtocol (injected)
    """

    def __init__(
        self,
        vector_store: AbstractVectorStore,
        registry: AbstractDocumentRegistry,
        settings: PipelineConfigProtocol,
    ):
        """
        Initialize the deduplication service.
        
        Args:
            vector_store: Vector store for chunk existence checks
            registry: Document registry for job status checks
            settings: Pipeline configuration
        """
        self.vector_store = vector_store
        self.registry = registry
        self.settings = settings

    def check_document_duplicate(self, doc_id: str) -> DuplicateResolution:
        """
        Check if a document has already been ingested.
        
        Layer 1 of deduplication strategy.
        
        Args:
            doc_id: Document ID (SHA-256 hash of PDF bytes)
        
        Returns:
            DuplicateResolution enum indicating action to take:
            - SKIP: Document fully ingested, skip it
            - RESUME: Partial ingestion, resume from failures
            - RETRY: Previous ingestion failed, retry
            - NEW: New document, proceed with ingestion
        """
        existing_job = self.registry.get_by_doc_id(doc_id)
        
        if not existing_job:
            logger.info("document_duplicate_check", doc_id=doc_id, result="NEW")
            return DuplicateResolution.NEW
        
        if existing_job.status == IngestionStatus.INGESTED:
            logger.info(
                "document_duplicate_check",
                doc_id=doc_id,
                result="SKIP",
                reason="already_ingested",
            )
            return DuplicateResolution.SKIP
        
        if existing_job.status == IngestionStatus.PARTIAL:
            logger.info(
                "document_duplicate_check",
                doc_id=doc_id,
                result="RESUME",
                failed_chunks=len(existing_job.failed_chunk_indices),
            )
            return DuplicateResolution.RESUME
        
        if existing_job.status in (IngestionStatus.FAILED, IngestionStatus.RETRY_PENDING):
            logger.info(
                "document_duplicate_check",
                doc_id=doc_id,
                result="RETRY",
                retry_count=existing_job.retry_count,
            )
            return DuplicateResolution.RETRY
        
        logger.info("document_duplicate_check", doc_id=doc_id, result="NEW")
        return DuplicateResolution.NEW

    def filter_duplicate_chunks(self, chunks: List[ChildChunk]) -> List[ChildChunk]:
        """
        Remove chunks that already exist in the vector store.
        
        Layer 2 of deduplication strategy.
        Uses a single batch lookup (``batch_chunk_exists``) instead of
        per-chunk queries to avoid the N+1 query pattern.
        
        Args:
            chunks: List of ChildChunk objects to check
        
        Returns:
            Filtered list containing only new chunks
        """
        if not chunks:
            return []

        all_ids = [c.chunk_id for c in chunks]
        existing_ids = self.vector_store.batch_chunk_exists(all_ids)

        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        duplicate_count = len(chunks) - len(new_chunks)
        
        if duplicate_count > 0:
            logger.info(
                "chunk_deduplication_complete",
                total_chunks=len(chunks),
                duplicates_filtered=duplicate_count,
                new_chunks=len(new_chunks),
            )
        
        return new_chunks

    def detect_semantic_near_duplicate(
        self,
        doc: ParsedDocument,
        sample_embeddings: List[List[float]],
    ) -> Optional[str]:
        """
        Detect if document is a semantic near-duplicate of an existing one.
        
        Layer 3 of deduplication strategy. Uses first N chunks to detect
        near-duplicates (e.g., same content with different PDF encoding).
        
        Args:
            doc: ParsedDocument to check
            sample_embeddings: Embeddings of first N chunks from document
        
        Returns:
            doc_id of near-duplicate if found, None otherwise
        """
        if not sample_embeddings:
            return None
        
        similar_chunks = self.vector_store.semantic_similarity_search(
            sample_embeddings[: self.settings.semantic_dedup_sample_chunks],
            top_k=1,
        )
        
        if not similar_chunks:
            return None
        
        for chunk in similar_chunks:
            if chunk.doc_id == doc.doc_id:
                continue
            
            logger.warning(
                "semantic_near_duplicate_detected",
                doc_id=doc.doc_id,
                pdf_name=doc.pdf_name,
                similar_doc_id=chunk.doc_id,
                similar_pdf_name=chunk.metadata.pdf_name,
                similarity_threshold=self.settings.semantic_dedup_similarity_threshold,
            )
            return chunk.doc_id
        
        return None
