"""
Domain models for tracking ingestion job state and progress.

The IngestionJob is the source of truth for pipeline execution state, stored
in the document registry (MongoDB). It enables idempotent re-runs and tracks
per-chunk success/failure for granular retry.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from .document_metadata import GuidelineMetadata


class IngestionStatus(Enum):
    """Status of a document ingestion job."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    INGESTED = "INGESTED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    RETRY_PENDING = "RETRY_PENDING"
    SUPERSEDED = "SUPERSEDED"


@dataclass
class IngestionJob:
    """
    Tracks the state and progress of a single document ingestion.
    
    Stored in the document registry to enable:
    - Idempotent re-runs (skip if already ingested)
    - Partial failure recovery (resume from failed chunks)
    - Progress tracking and observability
    """

    job_id: str
    doc_id: str
    pdf_name: str
    status: IngestionStatus
    total_chunks: int
    embedded_chunks: int
    failed_chunk_indices: List[int] = field(default_factory=list)
    figures_pending: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    metadata: Optional[GuidelineMetadata] = None
