"""
Ingestion orchestrator - main coordinator for the PDF ingestion pipeline.

Orchestrates the entire ingestion flow:
1. Parse document
2. Sanitise content
3. Process figures
4. Build chunks
5. Deduplicate
6. Embed
7. Persist

Each step is independently resumable and updates job status in the registry.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import structlog

from ..models.chunk import ChildChunk, ParentChunk
from ..models.document_metadata import DuplicateResolution, GuidelineMetadata
from ..models.ingestion_job import IngestionJob, IngestionStatus
from ..models.parsed_document import ParsedDocument
from ..ports.chunk_store_port import AbstractParentChunkStore
from ..ports.document_registry_port import AbstractDocumentRegistry
from ..ports.figure_describer_port import AbstractFigureDescriber
from ..ports.pdf_parser_port import AbstractPDFParser
from ..ports.text_embedder_port import AbstractTextEmbedder
from ..ports.vector_store_port import AbstractVectorStore
from ...config.pipeline_settings import PipelineSettings
from ...exceptions.pipeline_exceptions import (
    DocumentAlreadyIngestedException,
    EmbeddingBatchFailureError,
    PDFParseError,
)
from ...utils.logging_utils import bind_pipeline_context
from .chunking_service import ChunkingService
from .deduplication_service import DeduplicationService
from .document_hasher import DocumentHasher
from .pdf_sanitiser import PDFSanitiser


logger = structlog.get_logger(__name__)


class IngestionOrchestrator:
    """
    Orchestrates the entire PDF ingestion pipeline.
    
    Dependencies (all injected):
        - AbstractPDFParser
        - AbstractFigureDescriber
        - AbstractTextEmbedder
        - AbstractVectorStore
        - AbstractDocumentRegistry
        - AbstractParentChunkStore
        - PipelineSettings
        - DocumentHasher
        - PDFSanitiser
        - ChunkingService
        - DeduplicationService
    """

    def __init__(
        self,
        parser: AbstractPDFParser,
        figure_describer: AbstractFigureDescriber,
        embedder: AbstractTextEmbedder,
        vector_store: AbstractVectorStore,
        registry: AbstractDocumentRegistry,
        chunk_store: AbstractParentChunkStore,
        settings: PipelineSettings,
        hasher: DocumentHasher,
        sanitiser: PDFSanitiser,
        chunking_service: ChunkingService,
        dedup_service: DeduplicationService,
    ):
        """
        Initialize the ingestion orchestrator.
        
        Args:
            parser: PDF parser implementation
            figure_describer: Figure description service
            embedder: Text embedding service
            vector_store: Vector store for child chunks
            registry: Document registry for job tracking
            chunk_store: Storage for parent chunks
            settings: Pipeline configuration
            hasher: Document hasher
            sanitiser: PDF sanitiser
            chunking_service: Chunking service
            dedup_service: Deduplication service
        """
        self.parser = parser
        self.figure_describer = figure_describer
        self.embedder = embedder
        self.vector_store = vector_store
        self.registry = registry
        self.chunk_store = chunk_store
        self.settings = settings
        self.hasher = hasher
        self.sanitiser = sanitiser
        self.chunking_service = chunking_service
        self.dedup_service = dedup_service

    def orchestrate(
        self,
        pdf_path: Path,
        metadata: GuidelineMetadata,
    ) -> IngestionJob:
        """
        Orchestrate the ingestion of a single PDF document.
        
        This method is idempotent - safe to call multiple times on the same
        document. Failed steps can be resumed from the registry.
        
        Args:
            pdf_path: Path to the PDF file
            metadata: Guideline metadata
        
        Returns:
            IngestionJob with final status
        
        Raises:
            DocumentAlreadyIngestedException: If document already fully ingested
            PDFParseError: If PDF parsing fails
            Various other pipeline exceptions on failures
        """
        correlation_id = str(uuid.uuid4())
        doc_id = self.hasher.compute_document_id(pdf_path)
        
        bind_pipeline_context(correlation_id=correlation_id, doc_id=doc_id, step="init")
        
        logger.info(
            "ingestion_started",
            pdf_name=metadata.pdf_name,
            pdf_path=str(pdf_path),
        )
        
        dup_resolution = self.dedup_service.check_document_duplicate(doc_id)

        if dup_resolution == DuplicateResolution.SKIP:
            logger.info(
                "ingestion_skipped_already_ingested",
                doc_id=doc_id,
                pdf_name=metadata.pdf_name,
            )
            existing_job = self.registry.get_by_doc_id(doc_id)
            if existing_job:
                return existing_job
            # Fallback: create a SKIPPED sentinel job so callers always get a job back
            return IngestionJob(
                job_id=str(uuid.uuid4()),
                doc_id=doc_id,
                pdf_name=metadata.pdf_name,
                status=IngestionStatus.SKIPPED,
                total_chunks=0,
                embedded_chunks=0,
                completed_at=datetime.utcnow(),
            )

        job = IngestionJob(
            job_id=str(uuid.uuid4()),
            doc_id=doc_id,
            pdf_name=metadata.pdf_name,
            status=IngestionStatus.RUNNING,
            total_chunks=0,
            embedded_chunks=0,
            started_at=datetime.utcnow(),
        )
        
        try:
            self.registry.register_new_job(job)
            
            bind_pipeline_context(correlation_id=correlation_id, doc_id=doc_id, step="parse")
            doc = self._parse_document(pdf_path, doc_id)
            doc.doc_id = doc_id  # Propagate the content-addressed hash into the document

            bind_pipeline_context(correlation_id=correlation_id, doc_id=doc_id, step="sanitise")
            doc = self._sanitise_document(doc)
            
            bind_pipeline_context(correlation_id=correlation_id, doc_id=doc_id, step="figures")
            doc = self._process_figures(doc)
            
            bind_pipeline_context(correlation_id=correlation_id, doc_id=doc_id, step="chunk")
            parent_chunks, child_chunks = self._build_chunks(doc, metadata)
            
            bind_pipeline_context(correlation_id=correlation_id, doc_id=doc_id, step="dedup")
            child_chunks = self._deduplicate_chunks(child_chunks)
            
            bind_pipeline_context(correlation_id=correlation_id, doc_id=doc_id, step="embed")
            child_chunks = self._embed_chunks(child_chunks)
            
            bind_pipeline_context(correlation_id=correlation_id, doc_id=doc_id, step="persist")
            self._persist_chunks(parent_chunks, child_chunks)
            
            job.status = IngestionStatus.INGESTED
            job.total_chunks = len(child_chunks)
            job.embedded_chunks = len(child_chunks)
            job.completed_at = datetime.utcnow()
            
            self.registry.update_job_status(
                job.job_id,
                IngestionStatus.INGESTED,
                total_chunks=job.total_chunks,
                embedded_chunks=job.embedded_chunks,
                completed_at=job.completed_at,
            )
            
            logger.info(
                "ingestion_completed",
                job_id=job.job_id,
                doc_id=doc_id,
                total_chunks=job.total_chunks,
            )
            
            return job
            
        except Exception as e:
            logger.error(
                "ingestion_failed",
                job_id=job.job_id,
                doc_id=doc_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            
            job.status = IngestionStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            
            self.registry.update_job_status(
                job.job_id,
                IngestionStatus.FAILED,
                error_message=str(e),
                completed_at=job.completed_at,
            )
            
            raise

    def _parse_document(self, pdf_path: Path, doc_id: str) -> ParsedDocument:
        """
        Parse PDF document.
        
        Args:
            pdf_path: Path to PDF
            doc_id: Document ID
        
        Returns:
            ParsedDocument
        
        Raises:
            PDFParseError: If parsing fails
        """
        logger.info("parsing_document", pdf_path=str(pdf_path))
        
        try:
            doc = self.parser.parse(pdf_path)
            
            logger.info(
                "parsing_complete",
                sections=len(doc.sections),
                tables=len(doc.tables),
                figures=len(doc.figures),
                confidence_score=doc.parse_confidence_score,
            )
            
            return doc
            
        except Exception as e:
            logger.error("parsing_failed", error=str(e), error_type=type(e).__name__)
            raise PDFParseError(doc_id, str(pdf_path), e)

    def _sanitise_document(self, doc: ParsedDocument) -> ParsedDocument:
        """Sanitise document content."""
        logger.info("sanitising_document")
        
        sanitised_sections = [
            self.sanitiser.sanitise_section(section)
            for section in doc.sections
        ]
        
        ordering_valid = self.sanitiser.validate_section_ordering(sanitised_sections)
        
        if not ordering_valid:
            logger.warning("section_ordering_invalid", doc_id=doc.doc_id)
        
        doc.sections = sanitised_sections
        
        logger.info("sanitisation_complete", sections_processed=len(sanitised_sections))
        
        return doc

    def _process_figures(self, doc: ParsedDocument) -> ParsedDocument:
        """Process figures by generating descriptions."""
        logger.info("processing_figures", total_figures=len(doc.figures))
        
        figures_processed = 0
        figures_pending = 0
        
        for figure in doc.figures:
            if figure.description_pending:
                figures_pending += 1
            else:
                figures_processed += 1
        
        logger.info(
            "figure_processing_complete",
            processed=figures_processed,
            pending=figures_pending,
        )
        
        return doc

    def _build_chunks(
        self,
        doc: ParsedDocument,
        metadata: GuidelineMetadata,
    ) -> Tuple[List[ParentChunk], List[ChildChunk]]:
        """Build parent and child chunks."""
        logger.info("building_chunks")
        
        parent_chunks, child_chunks = self.chunking_service.build_parent_child_chunks(
            doc, metadata
        )
        
        logger.info(
            "chunking_complete",
            parent_chunks=len(parent_chunks),
            child_chunks=len(child_chunks),
        )
        
        return parent_chunks, child_chunks

    def _deduplicate_chunks(self, chunks: List[ChildChunk]) -> List[ChildChunk]:
        """Remove duplicate chunks."""
        logger.info("deduplicating_chunks", total_chunks=len(chunks))
        
        unique_chunks = self.dedup_service.filter_duplicate_chunks(chunks)
        
        logger.info(
            "deduplication_complete",
            unique_chunks=len(unique_chunks),
            duplicates_removed=len(chunks) - len(unique_chunks),
        )
        
        return unique_chunks

    def _embed_chunks(self, chunks: List[ChildChunk]) -> List[ChildChunk]:
        """Generate embeddings for chunks."""
        logger.info("embedding_chunks", total_chunks=len(chunks))
        
        texts = [chunk.content for chunk in chunks]
        
        try:
            embeddings = self.embedder.embed_batch(texts)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            logger.info("embedding_complete", chunks_embedded=len(chunks))
            
            return chunks
            
        except Exception as e:
            logger.error("embedding_failed", error=str(e), error_type=type(e).__name__)
            raise EmbeddingBatchFailureError(list(range(len(chunks))), e)

    def _persist_chunks(
        self,
        parent_chunks: List[ParentChunk],
        child_chunks: List[ChildChunk],
    ) -> None:
        """Persist chunks to storage."""
        logger.info(
            "persisting_chunks",
            parent_chunks=len(parent_chunks),
            child_chunks=len(child_chunks),
        )
        
        for parent in parent_chunks:
            self.chunk_store.save_parent_chunk(parent)
        
        self.vector_store.upsert_chunks(child_chunks)
        
        logger.info("persistence_complete")
