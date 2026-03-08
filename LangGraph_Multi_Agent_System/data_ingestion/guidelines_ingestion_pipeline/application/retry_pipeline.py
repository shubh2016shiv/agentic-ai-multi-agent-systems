"""
Retry pipeline for processing failed and partial ingestion jobs.

Provides targeted retry capabilities for:
- Jobs with partial failures (some chunks failed)
- Jobs marked as RETRY_PENDING
- Figures with pending descriptions
"""

from pathlib import Path
from typing import List

import structlog

from ..config.pipeline_settings import PipelineSettings
from ..domain.exceptions import RegistryError
from ..domain.models.ingestion_job import IngestionJob, IngestionStatus
from ..domain.ports.document_registry_port import AbstractDocumentRegistry
from ..domain.ports.vector_store_port import AbstractVectorStore
from ..domain.services.ingestion_orchestrator import IngestionOrchestrator


logger = structlog.get_logger(__name__)


class RetryPipeline:
    """
    Specialized pipeline for retrying failed ingestion jobs.
    
    Dependencies:
        - AbstractDocumentRegistry (injected)
        - AbstractVectorStore (injected)
        - IngestionOrchestrator (injected)
        - PipelineSettings (injected)
    """

    def __init__(
        self,
        registry: AbstractDocumentRegistry,
        vector_store: AbstractVectorStore,
        orchestrator: IngestionOrchestrator,
        settings: PipelineSettings,
    ):
        """
        Initialize the retry pipeline.
        
        Args:
            registry: Document registry for job tracking
            vector_store: Vector store for chunk queries
            orchestrator: Ingestion orchestrator to re-run jobs
            settings: Pipeline configuration
        """
        self.registry = registry
        self.vector_store = vector_store
        self.orchestrator = orchestrator
        self.settings = settings

    def retry_all_failed_jobs(self) -> List[IngestionJob]:
        """
        Retry all jobs with PARTIAL or RETRY_PENDING status.
        
        Returns:
            List of jobs after retry attempt
        """
        logger.info("retry_all_failed_jobs_started")
        
        partial_jobs = self.registry.get_jobs_by_status(IngestionStatus.PARTIAL)
        retry_jobs = self.registry.get_jobs_by_status(IngestionStatus.RETRY_PENDING)
        
        all_jobs = partial_jobs + retry_jobs
        
        logger.info(
            "retry_jobs_found",
            partial_jobs=len(partial_jobs),
            retry_pending=len(retry_jobs),
            total=len(all_jobs),
        )
        
        retried_jobs = []
        
        for job in all_jobs:
            if job.retry_count >= self.settings.max_embedding_retries:
                logger.warning(
                    "job_max_retries_exceeded",
                    job_id=job.job_id,
                    retry_count=job.retry_count,
                )
                continue
            
            if not job.metadata:
                logger.error(
                    "job_missing_metadata_cannot_retry",
                    job_id=job.job_id,
                    doc_id=job.doc_id,
                )
                continue
            
            logger.info(
                "retrying_job",
                job_id=job.job_id,
                doc_id=job.doc_id,
                failed_chunks=len(job.failed_chunk_indices),
            )
            
            try:
                pdf_path = Path(job.metadata.pdf_source_path) / job.metadata.pdf_name
                
                if not pdf_path.exists():
                    logger.error(
                        "pdf_not_found_cannot_retry",
                        job_id=job.job_id,
                        pdf_path=str(pdf_path),
                    )
                    continue
                
                updated_job = self.orchestrator.orchestrate(
                    pdf_path=pdf_path,
                    metadata=job.metadata,
                )
                
                retried_jobs.append(updated_job)
                
                logger.info(
                    "job_retry_complete",
                    job_id=updated_job.job_id,
                    status=updated_job.status.value,
                )
                
            except Exception as e:
                logger.error(
                    "job_retry_failed",
                    job_id=job.job_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                
                self.registry.update_job_status(
                    job.job_id,
                    IngestionStatus.FAILED,
                    retry_count=job.retry_count + 1,
                    error_message=str(e),
                )
        
        logger.info(
            "retry_all_failed_jobs_complete",
            jobs_retried=len(retried_jobs),
        )
        
        return retried_jobs

    def retry_specific_job(self, job_id: str) -> IngestionJob:
        """
        Retry a specific job by job ID.
        
        Args:
            job_id: Job ID to retry
        
        Returns:
            Updated IngestionJob
        
        Raises:
            RegistryJobNotFoundError: If job not found
        """
        logger.info("retry_specific_job_started", job_id=job_id)
        
        jobs = self.registry.get_jobs_by_status(IngestionStatus.PARTIAL)
        jobs += self.registry.get_jobs_by_status(IngestionStatus.RETRY_PENDING)
        
        job = next((j for j in jobs if j.job_id == job_id), None)
        
        if not job:
            from ..exceptions.pipeline_exceptions import RegistryJobNotFoundError
            raise RegistryJobNotFoundError(job_id)
        
        if not job.metadata:
            logger.error(
                "job_missing_metadata_cannot_retry",
                job_id=job.job_id,
            )
            raise RegistryError("Job metadata missing for retry", job_id=job.job_id)
        
        pdf_path = Path(job.metadata.pdf_source_path) / job.metadata.pdf_name
        
        updated_job = self.orchestrator.orchestrate(
            pdf_path=pdf_path,
            metadata=job.metadata,
        )
        
        logger.info("retry_specific_job_complete", job_id=job_id)
        
        return updated_job

    def get_retry_queue_stats(self) -> dict:
        """
        Get statistics on jobs awaiting retry.
        
        Returns:
            Dictionary with retry queue statistics
        """
        partial_jobs = self.registry.get_jobs_by_status(IngestionStatus.PARTIAL)
        retry_jobs = self.registry.get_jobs_by_status(IngestionStatus.RETRY_PENDING)
        failed_jobs = self.registry.get_jobs_by_status(IngestionStatus.FAILED)
        
        stats = {
            "partial_jobs": len(partial_jobs),
            "retry_pending": len(retry_jobs),
            "failed_jobs": len(failed_jobs),
            "total_awaiting_retry": len(partial_jobs) + len(retry_jobs),
        }
        
        logger.info("retry_queue_stats", **stats)
        
        return stats
