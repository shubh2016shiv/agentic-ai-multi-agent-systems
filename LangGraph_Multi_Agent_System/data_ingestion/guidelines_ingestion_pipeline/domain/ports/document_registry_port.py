"""
Abstract base class for document registry operations.

The document registry is the source of truth for ingestion job state,
enabling idempotent re-runs, partial failure recovery, and progress tracking.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..models.ingestion_job import IngestionJob, IngestionStatus


class AbstractDocumentRegistry(ABC):
    """Interface for document registry implementations."""

    @abstractmethod
    def get_by_doc_id(self, doc_id: str) -> Optional[IngestionJob]:
        """
        Retrieve an ingestion job by document ID.
        
        Args:
            doc_id: Document ID (SHA-256 hash of PDF bytes)
        
        Returns:
            IngestionJob if found, None otherwise
        """
        raise NotImplementedError("Subclasses must implement get_by_doc_id()")

    @abstractmethod
    def register_new_job(self, job: IngestionJob) -> None:
        """
        Register a new ingestion job in the registry.
        
        Args:
            job: IngestionJob to register
        
        Raises:
            RegistryConnectionError: If connection to registry fails
        """
        raise NotImplementedError("Subclasses must implement register_new_job()")

    @abstractmethod
    def update_job_status(
        self,
        job_id: str,
        status: IngestionStatus,
        **kwargs
    ) -> None:
        """
        Update the status and optional fields of an ingestion job.
        
        Args:
            job_id: Job ID (UUID)
            status: New status
            **kwargs: Additional fields to update (e.g., embedded_chunks,
                     error_message, completed_at)
        
        Raises:
            RegistryJobNotFoundError: If job_id does not exist
            RegistryConnectionError: If connection to registry fails
        """
        raise NotImplementedError("Subclasses must implement update_job_status()")

    @abstractmethod
    def mark_chunk_as_completed(self, job_id: str, chunk_index: int) -> None:
        """
        Mark a specific chunk as successfully processed.
        
        Args:
            job_id: Job ID (UUID)
            chunk_index: Index of the chunk within the document
        
        Raises:
            RegistryJobNotFoundError: If job_id does not exist
        """
        raise NotImplementedError("Subclasses must implement mark_chunk_as_completed()")

    @abstractmethod
    def mark_chunk_as_failed(
        self,
        job_id: str,
        chunk_index: int,
        error: str
    ) -> None:
        """
        Mark a specific chunk as failed and record the error.
        
        Args:
            job_id: Job ID (UUID)
            chunk_index: Index of the chunk within the document
            error: Error message or exception string
        
        Raises:
            RegistryJobNotFoundError: If job_id does not exist
        """
        raise NotImplementedError("Subclasses must implement mark_chunk_as_failed()")

    @abstractmethod
    def get_jobs_by_status(self, status: IngestionStatus) -> List[IngestionJob]:
        """
        Retrieve all jobs with a specific status.
        
        Args:
            status: Status to filter by
        
        Returns:
            List of IngestionJob objects with the specified status
        """
        raise NotImplementedError("Subclasses must implement get_jobs_by_status()")
