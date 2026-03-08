"""
Guidelines Ingestion Pipeline - Main composition root and entry point.

This is the ONLY class external callers need to interact with. It composes
all infrastructure implementations and injects them into domain services.
"""

from pathlib import Path
from typing import Callable, List

import structlog

from ..config.pipeline_settings import PipelineSettings
from ..domain.models.document_metadata import GuidelineMetadata
from ..domain.models.ingestion_job import IngestionJob, IngestionStatus
from ..domain.services.chunking_service import ChunkingService
from ..domain.services.deduplication_service import DeduplicationService
from ..domain.services.document_hasher import DocumentHasher
from ..domain.services.ingestion_orchestrator import IngestionOrchestrator
from ..domain.services.pdf_sanitiser import PDFSanitiser
from ..infrastructure.chunk_stores.mongo_parent_chunk_store import MongoParentChunkStore
from ..infrastructure.embedders.bge_m3_embedder import BGEM3Embedder
from ..infrastructure.figure_describers.vision_llm_figure_describer import (
    VisionLLMFigureDescriber,
)
from ..infrastructure.parsers.docling_pdf_parser import DoclingPDFParser
from ..infrastructure.registries.mongo_document_registry import MongoDocumentRegistry
from ..infrastructure.vector_stores.chroma_vector_store import ChromaVectorStore


logger = structlog.get_logger(__name__)


class GuidelinesIngestionPipeline:
    """
    Main entry point for the guidelines ingestion pipeline.
    
    This is the composition root - it wires together all infrastructure
    implementations and domain services. External callers interact only
    with this class.
    
    Example usage:
        settings = PipelineSettings()
        pipeline = GuidelinesIngestionPipeline(settings)
        
        metadata = GuidelineMetadata(
            guideline_org="ACC",
            guideline_year=2021,
            therapeutic_area="cardiology",
            condition_focus="hypertriglyceridemia",
            pdf_name="hyperlipidemia.pdf",
            pdf_source_path="guidelines/cardiology",
        )
        
        job = pipeline.ingest_single_pdf(
            pdf_path=Path("hyperlipidemia.pdf"),
            metadata=metadata,
        )
    """

    def __init__(self, settings: PipelineSettings):
        """
        Initialize the ingestion pipeline with all dependencies.
        
        Args:
            settings: Pipeline configuration from environment
        """
        self.settings = settings
        
        logger.info(
            "pipeline_initialization_started",
            pipeline_version=settings.pipeline_version,
        )
        
        self._initialize_infrastructure()
        self._initialize_domain_services()
        self._initialize_orchestrator()
        
        logger.info("pipeline_initialization_complete")

    def ingest_single_pdf(
        self,
        pdf_path: Path,
        metadata: GuidelineMetadata,
    ) -> IngestionJob:
        """
        Ingest a single PDF document.
        
        This method is idempotent - safe to call multiple times on the same
        PDF. If the document is already ingested, it will be skipped.
        
        Args:
            pdf_path: Path to the PDF file
            metadata: Guideline metadata (org, year, therapeutic area, etc.)
        
        Returns:
            IngestionJob with status and statistics
        
        Raises:
            DocumentAlreadyIngestedException: If document already fully ingested
            PDFParseError: If PDF parsing fails
            Various other pipeline exceptions on failures
        """
        logger.info(
            "ingest_single_pdf_started",
            pdf_name=metadata.pdf_name,
            pdf_path=str(pdf_path),
        )
        
        job = self.orchestrator.orchestrate(pdf_path, metadata)
        
        logger.info(
            "ingest_single_pdf_complete",
            job_id=job.job_id,
            status=job.status.value,
            total_chunks=job.total_chunks,
        )
        
        return job

    def ingest_directory(
        self,
        directory_path: Path,
        metadata_resolver: Callable[[Path], GuidelineMetadata],
    ) -> List[IngestionJob]:
        """
        Ingest all PDFs in a directory.
        
        Each PDF is processed independently. One PDF failure does not stop
        processing of other PDFs.
        
        Args:
            directory_path: Path to directory containing PDFs
            metadata_resolver: Function that takes a PDF path and returns
                              its GuidelineMetadata
        
        Returns:
            List of IngestionJob objects, one per PDF
        """
        logger.info("ingest_directory_started", directory=str(directory_path))
        
        pdf_files = list(directory_path.glob("*.pdf"))
        jobs = []
        
        for pdf_path in pdf_files:
            try:
                metadata = metadata_resolver(pdf_path)
                job = self.ingest_single_pdf(pdf_path, metadata)
                jobs.append(job)
            except Exception as e:
                logger.error(
                    "pdf_ingestion_failed",
                    pdf_path=str(pdf_path),
                    error=str(e),
                    error_type=type(e).__name__,
                )
                continue
        
        logger.info(
            "ingest_directory_complete",
            total_pdfs=len(pdf_files),
            successful=sum(1 for j in jobs if j.status == IngestionStatus.INGESTED),
            failed=sum(1 for j in jobs if j.status == IngestionStatus.FAILED),
        )
        
        return jobs

    def retry_failed_jobs(self) -> List[IngestionJob]:
        """
        Retry all jobs with status PARTIAL or RETRY_PENDING.
        
        Returns:
            List of IngestionJob objects after retry
        """
        logger.info("retry_failed_jobs_started")
        
        partial_jobs = self.registry.get_jobs_by_status(IngestionStatus.PARTIAL)
        retry_pending_jobs = self.registry.get_jobs_by_status(IngestionStatus.RETRY_PENDING)
        
        all_retry_jobs = partial_jobs + retry_pending_jobs
        retried_jobs = []
        
        for job in all_retry_jobs:
            logger.info(
                "retrying_job",
                job_id=job.job_id,
                doc_id=job.doc_id,
                pdf_name=job.pdf_name,
            )
            
            try:
                metadata = GuidelineMetadata(
                    guideline_org="",
                    guideline_year=0,
                    therapeutic_area="",
                    condition_focus="",
                    pdf_name=job.pdf_name,
                    pdf_source_path="",
                )
                
                retried_job = self.ingest_single_pdf(
                    pdf_path=Path(job.pdf_name),
                    metadata=metadata,
                )
                retried_jobs.append(retried_job)
            except Exception as e:
                logger.error(
                    "job_retry_failed",
                    job_id=job.job_id,
                    error=str(e),
                )
        
        logger.info(
            "retry_failed_jobs_complete",
            jobs_retried=len(retried_jobs),
        )
        
        return retried_jobs

    def retry_pending_figures(self) -> int:
        """
        Reprocess all chunks with figure_description_pending=True.
        
        Returns:
            Count of figures successfully described
        """
        logger.info("retry_pending_figures_started")
        
        logger.warning("retry_pending_figures_not_implemented")
        
        return 0

    def _initialize_infrastructure(self) -> None:
        """Initialize all infrastructure implementations."""
        self.parser = DoclingPDFParser()
        
        self.figure_describer = VisionLLMFigureDescriber(
            model_name=self.settings.vision_llm_model,
            timeout=self.settings.vision_llm_timeout_seconds,
        )
        
        self.embedder = BGEM3Embedder(
            model_name=self.settings.embedding_model_name,
        )
        
        self.vector_store = ChromaVectorStore(
            chroma_path="./chroma_db",
            collection_name=self.settings.chroma_collection_name,
            wal_file_path=self.settings.wal_file_path,
            batch_size=self.settings.chroma_write_batch_size,
            hnsw_construction_ef=self.settings.chroma_hnsw_construction_ef,
            hnsw_m=self.settings.chroma_hnsw_m,
        )
        
        self.registry = MongoDocumentRegistry(
            mongodb_uri=self.settings.mongodb_uri,
            database_name=self.settings.mongodb_database,
            collection_name=self.settings.mongodb_registry_collection,
        )
        
        self.chunk_store = MongoParentChunkStore(
            mongodb_uri=self.settings.mongodb_uri,
            database_name=self.settings.mongodb_database,
            collection_name=self.settings.mongodb_parent_chunks_collection,
        )

    def _initialize_domain_services(self) -> None:
        """Initialize all domain services."""
        self.hasher = DocumentHasher()
        
        self.sanitiser = PDFSanitiser(self.settings)
        
        self.chunking_service = ChunkingService(
            settings=self.settings,
            hasher=self.hasher,
        )
        
        self.dedup_service = DeduplicationService(
            vector_store=self.vector_store,
            registry=self.registry,
            settings=self.settings,
        )

    def _initialize_orchestrator(self) -> None:
        """Initialize the ingestion orchestrator."""
        self.orchestrator = IngestionOrchestrator(
            parser=self.parser,
            figure_describer=self.figure_describer,
            embedder=self.embedder,
            vector_store=self.vector_store,
            registry=self.registry,
            chunk_store=self.chunk_store,
            settings=self.settings,
            hasher=self.hasher,
            sanitiser=self.sanitiser,
            chunking_service=self.chunking_service,
            dedup_service=self.dedup_service,
        )
