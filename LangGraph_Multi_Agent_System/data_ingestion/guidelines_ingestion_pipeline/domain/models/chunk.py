"""
Domain models for chunk representation in the parent-child chunking strategy.

Child chunks are embedded and stored in ChromaDB for retrieval. Parent chunks
provide extended context and are stored in MongoDB for post-retrieval context
enrichment.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class ChunkType(Enum):
    """Types of content chunks."""

    TEXT = "TEXT"
    TABLE = "TABLE"
    FLOWCHART_DESCRIPTION = "FLOWCHART_DESCRIPTION"
    FIGURE_DESCRIPTION = "FIGURE_DESCRIPTION"
    FIGURE_DESCRIPTION_PENDING = "FIGURE_DESCRIPTION_PENDING"


@dataclass
class ChunkMetadata:
    """Metadata attached to each child chunk for retrieval filtering and audit."""

    pdf_name: str
    pdf_source_path: str
    guideline_org: str
    guideline_year: int
    therapeutic_area: str
    condition_focus: str
    parser_version: str
    embedding_model: str
    pipeline_version: str
    ingested_at: datetime
    is_superseded: bool = False


@dataclass
class ParentChunk:
    """
    Large-context chunk (1200-1500 tokens) stored in MongoDB.
    
    Not embedded. Retrieved after child chunk matching to provide extended
    context to the LLM.
    """

    parent_chunk_id: str
    doc_id: str
    content: str
    section_heading: str
    page_start: int
    page_end: int
    child_chunk_ids: List[str] = field(default_factory=list)
    token_count: int = 0


@dataclass
class ChildChunk:
    """
    Precision chunk (300-400 tokens) embedded and stored in ChromaDB.
    
    Used for semantic similarity search. Each child references its parent
    for context retrieval.
    """

    chunk_id: str
    parent_chunk_id: str
    doc_id: str
    content: str
    chunk_type: ChunkType
    page_numbers: List[int]
    section_heading: str
    section_depth: int
    chunk_index: int
    token_count: int
    confidence_score: float
    is_ocr_sourced: bool
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None
