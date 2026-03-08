"""
Retry pipeline for processing failed and partial ingestion jobs.

Provides targeted retry capabilities for:
- Jobs with partial failures (some chunks failed)
- Jobs marked as RETRY_PENDING
- Figures with pending descriptions
"""

from typing import List

import structlog

from ..config.pipeline_settings import PipelineSettings
from ..domain.models.ingestion_job import IngestionJob, IngestionStatus
from ..domain.ports.document_registry_port import AbstractDocumentRegistry
from ..domain.ports.vector_store_port import AbstractVectorStore


logger = structlog.get_logger(__name__)


class RetryPipeline:
    """
    Specialized pipeline for retrying failed ingestion jobs.
    
    Dependencies:
        - AbstractDocumentRegistry (injected)
        - AbstractVectorStore (injected)
        - PipelineSettings (injected)
    """

    def __init__(
        self,
        registry: AbstractDocumentRegistry,
        vector_store: AbstractVectorStore,
        settings: PipelineSettings,
    ):
        """
        Initialize the retry pipeline.
        
        Args:
            registry: Document registry for job tracking
            vector_store: Vector store for chunk queries
            settings: Pipeline configuration
        """
        self.registry = registry
        self.vector_store = vector_store
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
            
            logger.info(
                "retrying_job",
                job_id=job.job_id,
                doc_id=job.doc_id,
                failed_chunks=len(job.failed_chunk_indices),
            )
            
            self.registry.update_job_status(
                job.job_id,
                IngestionStatus.RUNNING,
                retry_count=job.retry_count + 1,
            )
            
            retried_jobs.append(job)
        
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
        
        self.registry.update_job_status(
            job.job_id,
            IngestionStatus.RUNNING,
            retry_count=job.retry_count + 1,
        )
        
        logger.info("retry_specific_job_complete", job_id=job_id)
        
        return job

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
