"""
Guidelines Ingestion Pipeline - Main composition root and entry point.

This is the ONLY class external callers need to interact with. It composes
all infrastructure implementations and injects them into domain services.
"""

import json
from pathlib import Path
from typing import Callable, List

import structlog

from data_ingestion.connections.chroma_connection_manager import (
    ChromaConnectionManager,
)
from data_ingestion.connections.mongodb_connection_manager import (
    MongoDBConnectionManager,
)
from ..config.pipeline_settings import PipelineSettings
from ..domain.models.chunk import ChunkType
from ..domain.models.document_metadata import GuidelineMetadata
from ..domain.models.ingestion_job import IngestionJob, IngestionStatus
from ..domain.models.parsed_document import FigureType
from .retry_pipeline import RetryPipeline
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
            List of jobs after retry processing
        """
        logger.info("retry_failed_jobs_started")
        jobs = self.retry_pipeline.retry_all_failed_jobs()
        logger.info("retry_failed_jobs_complete", jobs_retried=len(jobs))
        return jobs

    def close(self) -> None:
        """
        Release all infrastructure connections.

        Safe to call multiple times.  Also invoked automatically when
        the pipeline is used as a context manager.
        """
        logger.info("pipeline_closing")
        self._mongo_manager.close()
        self._chroma_manager.close()
        logger.info("pipeline_closed")

    def __enter__(self) -> "GuidelinesIngestionPipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def retry_pending_figures(self) -> int:
        """
        Reprocess all chunks with figure_description_pending=True.
        
        Queries ChromaDB for chunks with chunk_type='FIGURE_DESCRIPTION_PENDING',
        retrieves the parent chunk metadata to find the PDF source, crops the
        figure from the PDF, calls the vision LLM, and updates the chunk in
        ChromaDB with the real description.
        
        Returns:
            Count of figures successfully described
        """
        logger.info("retry_pending_figures_started")
        
        try:
            results = self.vector_store.collection.get(
                where={"chunk_type": ChunkType.FIGURE_DESCRIPTION_PENDING.value},
                include=["metadatas", "documents"],
            )

            if not results["ids"]:
                logger.info("no_pending_figures_found")
                return 0

            logger.info("pending_figures_found", count=len(results["ids"]))

            pending_chunks_by_doc: dict[str, list[tuple[str, dict]]] = {}
            for chunk_id, metadata in zip(results["ids"], results["metadatas"]):
                doc_id = metadata.get("doc_id")
                if not doc_id:
                    logger.warning(
                        "pending_figure_missing_doc_id",
                        chunk_id=chunk_id,
                    )
                    continue
                pending_chunks_by_doc.setdefault(doc_id, []).append((chunk_id, metadata))

            success_count = 0

            for doc_id, pending_entries in pending_chunks_by_doc.items():
                job = self.registry.get_by_doc_id(doc_id)
                if not job or not job.metadata:
                    logger.warning(
                        "pending_figure_job_metadata_missing",
                        doc_id=doc_id,
                    )
                    continue

                pdf_path = Path(job.metadata.pdf_source_path) / job.metadata.pdf_name
                if not pdf_path.exists():
                    logger.warning(
                        "pending_figure_pdf_missing",
                        doc_id=doc_id,
                        pdf_path=str(pdf_path),
                    )
                    continue

                try:
                    parsed_doc = self.parser.parse(pdf_path)
                    parsed_doc.doc_id = doc_id
                    processed_doc = self.orchestrator._process_figures(parsed_doc, pdf_path)
                except Exception as e:
                    logger.warning(
                        "pending_figure_doc_reprocess_failed",
                        doc_id=doc_id,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    continue

                figure_lookup: dict[tuple[str, int], tuple[str, FigureType]] = {}
                for figure in processed_doc.figures:
                    if figure.description and not figure.description_pending:
                        figure_lookup[(figure.caption, figure.page_number)] = (
                            figure.description,
                            figure.figure_type,
                        )

                updates: list[str] = []
                documents: list[str] = []
                metadatas: list[dict] = []

                for chunk_id, metadata in pending_entries:
                    page_numbers_raw = metadata.get("page_numbers", "[]")
                    try:
                        page_numbers = json.loads(page_numbers_raw)
                    except (TypeError, json.JSONDecodeError):
                        page_numbers = []

                    page_number = page_numbers[0] if page_numbers else 0
                    caption = metadata.get("section_heading", "")
                    lookup_key = (caption, page_number)

                    if lookup_key not in figure_lookup:
                        logger.warning(
                            "pending_figure_description_unavailable",
                            chunk_id=chunk_id,
                            caption=caption,
                            page_number=page_number,
                        )
                        continue

                    description, figure_type = figure_lookup[lookup_key]
                    updated_metadata = dict(metadata)
                    updated_metadata["chunk_type"] = (
                        ChunkType.FLOWCHART_DESCRIPTION.value
                        if figure_type == FigureType.FLOWCHART
                        else ChunkType.FIGURE_DESCRIPTION.value
                    )
                    updated_metadata["confidence_score"] = 0.9

                    updates.append(chunk_id)
                    documents.append(f"**{caption}**\n\n{description}")
                    metadatas.append(updated_metadata)

                if updates:
                    self.vector_store.collection.update(
                        ids=updates,
                        documents=documents,
                        metadatas=metadatas,
                    )
                    success_count += len(updates)

            logger.info(
                "retry_pending_figures_complete",
                total_pending=len(results["ids"]),
                successfully_described=success_count,
            )
            return success_count

        except Exception as e:
            logger.error(
                "retry_pending_figures_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return 0

    def _initialize_infrastructure(self) -> None:
        """
        Create connection managers once and inject them into infrastructure components.

        Both MongoDB components (registry + chunk store) share a single
        ``MongoDBConnectionManager`` instance — one connection pool for the
        entire pipeline run.  The ChromaDB vector store receives its own
        ``ChromaConnectionManager``.  Neither component constructs a raw
        client itself, satisfying the Dependency Inversion Principle.
        """
        self.parser = DoclingPDFParser()

        self.figure_describer = VisionLLMFigureDescriber(
            model_name=self.settings.vision_llm_model,
            timeout=self.settings.vision_llm_timeout_seconds,
        )

        self.embedder = BGEM3Embedder(
            model_name=self.settings.embedding_model_name,
        )

        # -- Connection managers (created once, shared across components) ----
        self._mongo_manager = MongoDBConnectionManager(
            mongodb_uri=self.settings.mongodb_uri,
            database_name=self.settings.mongodb_database,
        )
        self._chroma_manager = ChromaConnectionManager(
            persist_path=self.settings.chroma_persist_path,
            hnsw_construction_ef=self.settings.chroma_hnsw_construction_ef,
            hnsw_m=self.settings.chroma_hnsw_m,
        )

        # -- Infrastructure components (injected, not self-constructing) -----
        self.vector_store = ChromaVectorStore(
            connection_manager=self._chroma_manager,
            collection_name=self.settings.chroma_collection_name,
            wal_file_path=self.settings.wal_file_path,
            batch_size=self.settings.chroma_write_batch_size,
        )

        self.registry = MongoDocumentRegistry(
            connection_manager=self._mongo_manager,
            collection_name=self.settings.mongodb_registry_collection,
        )

        self.chunk_store = MongoParentChunkStore(
            connection_manager=self._mongo_manager,
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
        self.retry_pipeline = RetryPipeline(
            registry=self.registry,
            vector_store=self.vector_store,
            orchestrator=self.orchestrator,
            settings=self.settings,
        )
