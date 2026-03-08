"""
Abstract base class for PDF parsing.

Defines the contract that concrete PDF parsers (Docling, pdfplumber, etc.)
must implement. The domain layer depends only on this interface, never on
concrete parser implementations.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from ..models.parsed_document import ParsedDocument


class AbstractPDFParser(ABC):
    """Interface for PDF parsing implementations."""

    @abstractmethod
    def parse(self, pdf_path: Path) -> ParsedDocument:
        """
        Parse a PDF file into a structured ParsedDocument.
        
        Args:
            pdf_path: Path to the PDF file to parse
        
        Returns:
            ParsedDocument with extracted sections, tables, and figures
        
        Raises:
            PDFParseError: If parsing fails
            PDFCorruptedError: If PDF is malformed
            PDFPasswordProtectedError: If PDF requires password
            PDFNoTextLayerError: If PDF has no embedded text
        """
        raise NotImplementedError("Subclasses must implement parse()")

    @abstractmethod
    def get_parser_version(self) -> str:
        """
        Get the version string of the parser implementation.
        
        Returns:
            Version string (e.g., "docling-2.26.0")
        """
        raise NotImplementedError("Subclasses must implement get_parser_version()")
