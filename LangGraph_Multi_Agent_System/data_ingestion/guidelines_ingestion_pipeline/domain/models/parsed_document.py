"""
Domain models for parsed PDF documents.

Represents the output of the PDF parser layer, including sections, tables,
figures, and their associated metadata. These models are the input to the
chunking layer.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


@dataclass
class BoundingBox:
    """Coordinates of a region within a PDF page."""

    x: float
    y: float
    width: float
    height: float


class FigureType(Enum):
    """Types of figures found in clinical PDFs."""

    FLOWCHART = "FLOWCHART"
    TABLE_IMAGE = "TABLE_IMAGE"
    DIAGRAM = "DIAGRAM"
    PHOTO = "PHOTO"


@dataclass
class ParsedSection:
    """A single text section from the PDF."""

    section_id: str
    heading: str
    depth: int
    content: str
    page_numbers: List[int]
    token_count: int


@dataclass
class ParsedTable:
    """A table extracted from the PDF."""

    table_id: str
    caption: str
    markdown_content: str
    page_numbers: List[int]
    column_count: int
    row_count: int
    is_cross_page: bool
    validation_passed: bool


@dataclass
class ParsedFigure:
    """A figure or flowchart extracted from the PDF."""

    figure_id: str
    caption: str
    page_number: int
    bounding_box: BoundingBox
    figure_type: FigureType
    description: Optional[str] = None
    description_pending: bool = False


@dataclass
class ParsedDocument:
    """Complete parsed representation of a PDF document."""

    doc_id: str
    pdf_name: str
    pdf_source_path: str
    total_pages: int
    is_ocr_sourced: bool
    parser_version: str
    sections: List[ParsedSection] = field(default_factory=list)
    tables: List[ParsedTable] = field(default_factory=list)
    figures: List[ParsedFigure] = field(default_factory=list)
    parse_confidence_score: float = 1.0
    parse_warnings: List[str] = field(default_factory=list)
