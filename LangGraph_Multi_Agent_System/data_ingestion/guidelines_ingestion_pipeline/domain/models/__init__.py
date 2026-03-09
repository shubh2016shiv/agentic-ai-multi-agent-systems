"""Domain models for the ingestion pipeline."""

from .chunk import ChunkMetadata, ChunkType, ChildChunk, ParentChunk
from .document_metadata import DuplicateResolution, GuidelineMetadata
from .ingestion_job import IngestionJob, IngestionStatus
from .parsed_document import (
    BoundingBox,
    FigureType,
    ParsedDocument,
    ParsedFigure,
    ParsedSection,
    ParsedTable,
)

__all__ = [
    "BoundingBox",
    "FigureType",
    "ParsedSection",
    "ParsedTable",
    "ParsedFigure",
    "ParsedDocument",
    "ChunkType",
    "ChunkMetadata",
    "ParentChunk",
    "ChildChunk",
    "IngestionStatus",
    "IngestionJob",
    "GuidelineMetadata",
    "DuplicateResolution",
]
