"""
MongoDB document registry implementation.

Tracks ingestion job state, enabling idempotent re-runs and partial failure
recovery. This is the source of truth for pipeline execution state.
"""

from datetime import datetime
from typing import List, Optional

import structlog
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError

from ...domain.models.ingestion_job import IngestionJob, IngestionStatus
from ...domain.ports.document_registry_port import AbstractDocumentRegistry
from ...exceptions.pipeline_exceptions import (
    RegistryConnectionError,
    RegistryJobNotFoundError,
)


logger = structlog.get_logger(__name__)


class MongoDocumentRegistry(AbstractDocumentRegistry):
    """
    MongoDB-based document registry for tracking ingestion jobs.
    
    Features:
    - Persistent job state tracking
    - Idempotent job registration
    - Per-chunk failure tracking
    """

    def __init__(self, mongodb_uri: str, database_name: str, collection_name: str):
        """
        Initialize the MongoDB document registry.
        
        Args:
            mongodb_uri: MongoDB connection URI
            database_name: Database name
            collection_name: Collection name for jobs
        """
        try:
            self.client = MongoClient(mongodb_uri)
            self.db = self.client[database_name]
            self.collection = self.db[collection_name]
            
            self.collection.create_index("doc_id", unique=True)
            self.collection.create_index("status")
            self.collection.create_index("job_id", unique=True)
            
            logger.info(
                "mongo_registry_initialized",
                database=database_name,
                collection=collection_name,
            )
            
        except ConnectionFailure as e:
            raise RegistryConnectionError(registry_uri=mongodb_uri, cause=e)

    def get_by_doc_id(self, doc_id: str) -> Optional[IngestionJob]:
        """
        Retrieve an ingestion job by document ID.
        
        Args:
            doc_id: Document ID
        
        Returns:
            IngestionJob if found, None otherwise
        """
        try:
            doc = self.collection.find_one({"doc_id": doc_id})
            
            if not doc:
                return None
            
            return self._doc_to_job(doc)
            
        except Exception as e:
            logger.error("get_by_doc_id_failed", doc_id=doc_id, error=str(e))
            return None

    def register_new_job(self, job: IngestionJob) -> None:
        """
        Register a new ingestion job.
        
        Args:
            job: IngestionJob to register
        
        Raises:
            RegistryConnectionError: If connection fails
        """
        try:
            doc = self._job_to_doc(job)
            self.collection.insert_one(doc)
            
            logger.info(
                "job_registered",
                job_id=job.job_id,
                doc_id=job.doc_id,
                pdf_name=job.pdf_name,
            )
            
        except DuplicateKeyError:
            logger.warning(
                "job_already_registered",
                job_id=job.job_id,
                doc_id=job.doc_id,
            )
        except Exception as e:
            logger.error("register_job_failed", job_id=job.job_id, error=str(e))
            raise RegistryConnectionError(registry_uri="mongodb", cause=e)

    def update_job_status(
        self,
        job_id: str,
        status: IngestionStatus,
        **kwargs
    ) -> None:
        """
        Update job status and optional fields.
        
        Args:
            job_id: Job ID
            status: New status
            **kwargs: Additional fields to update
        
        Raises:
            RegistryJobNotFoundError: If job not found
        """
        try:
            update_doc = {"status": status.value}
            update_doc.update(kwargs)
            
            result = self.collection.update_one(
                {"job_id": job_id},
                {"$set": update_doc}
            )
            
            if result.matched_count == 0:
                raise RegistryJobNotFoundError(job_id)
            
            logger.info(
                "job_status_updated",
                job_id=job_id,
                status=status.value,
            )
            
        except RegistryJobNotFoundError:
            raise
        except Exception as e:
            logger.error("update_job_status_failed", job_id=job_id, error=str(e))
            raise RegistryConnectionError(registry_uri="mongodb", cause=e)

    def mark_chunk_as_completed(self, job_id: str, chunk_index: int) -> None:
        """
        Mark a chunk as successfully processed.
        
        Args:
            job_id: Job ID
            chunk_index: Chunk index
        
        Raises:
            RegistryJobNotFoundError: If job not found
        """
        try:
            result = self.collection.update_one(
                {"job_id": job_id},
                {"$inc": {"embedded_chunks": 1}}
            )
            
            if result.matched_count == 0:
                raise RegistryJobNotFoundError(job_id)
            
        except Exception as e:
            logger.error("mark_chunk_completed_failed", job_id=job_id, error=str(e))

    def mark_chunk_as_failed(
        self,
        job_id: str,
        chunk_index: int,
        error: str
    ) -> None:
        """
        Mark a chunk as failed.
        
        Args:
            job_id: Job ID
            chunk_index: Chunk index
            error: Error message
        
        Raises:
            RegistryJobNotFoundError: If job not found
        """
        try:
            result = self.collection.update_one(
                {"job_id": job_id},
                {
                    "$addToSet": {"failed_chunk_indices": chunk_index},
                    "$set": {"error_message": error}
                }
            )
            
            if result.matched_count == 0:
                raise RegistryJobNotFoundError(job_id)
            
            logger.warning(
                "chunk_marked_failed",
                job_id=job_id,
                chunk_index=chunk_index,
                error=error[:100],
            )
            
        except Exception as e:
            logger.error("mark_chunk_failed_failed", job_id=job_id, error=str(e))

    def get_jobs_by_status(self, status: IngestionStatus) -> List[IngestionJob]:
        """
        Retrieve all jobs with a specific status.
        
        Args:
            status: Status to filter by
        
        Returns:
            List of IngestionJob objects
        """
        try:
            docs = self.collection.find({"status": status.value})
            
            jobs = [self._doc_to_job(doc) for doc in docs]
            
            return jobs
            
        except Exception as e:
            logger.error("get_jobs_by_status_failed", status=status.value, error=str(e))
            return []

    def _job_to_doc(self, job: IngestionJob) -> dict:
        """Convert IngestionJob to MongoDB document."""
        return {
            "job_id": job.job_id,
            "doc_id": job.doc_id,
            "pdf_name": job.pdf_name,
            "status": job.status.value,
            "total_chunks": job.total_chunks,
            "embedded_chunks": job.embedded_chunks,
            "failed_chunk_indices": job.failed_chunk_indices,
            "figures_pending": job.figures_pending,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "error_message": job.error_message,
            "retry_count": job.retry_count,
        }

    def _doc_to_job(self, doc: dict) -> IngestionJob:
        """Convert MongoDB document to IngestionJob."""
        return IngestionJob(
            job_id=doc["job_id"],
            doc_id=doc["doc_id"],
            pdf_name=doc["pdf_name"],
            status=IngestionStatus(doc["status"]),
            total_chunks=doc.get("total_chunks", 0),
            embedded_chunks=doc.get("embedded_chunks", 0),
            failed_chunk_indices=doc.get("failed_chunk_indices", []),
            figures_pending=doc.get("figures_pending", 0),
            started_at=doc.get("started_at", datetime.utcnow()),
            completed_at=doc.get("completed_at"),
            error_message=doc.get("error_message"),
            retry_count=doc.get("retry_count", 0),
        )
