"""
Domain models for document metadata and deduplication resolution.

GuidelineMetadata captures provenance information about clinical PDFs to enable
precise retrieval filtering. DuplicateResolution guides the pipeline on how to
handle documents that may already exist in the system.
"""

from dataclasses import dataclass
from enum import Enum


class DuplicateResolution(Enum):
    """Resolution strategy for duplicate document detection."""

    SKIP = "SKIP"
    RESUME = "RESUME"
    RETRY = "RETRY"
    NEW = "NEW"


@dataclass
class GuidelineMetadata:
    """
    Provenance metadata for a clinical guideline PDF.
    
    Used for:
    - Retrieval filtering (e.g., only ACC guidelines, only cardiology)
    - Audit trails and versioning
    - Supersession tracking when guidelines are updated
    """

    guideline_org: str
    guideline_year: int
    therapeutic_area: str
    condition_focus: str
    pdf_name: str
    pdf_source_path: str
