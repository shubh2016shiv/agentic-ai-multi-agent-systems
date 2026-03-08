"""
PDFPlumber fallback parser implementation.

Used as a fallback when Docling parsing fails or produces low-confidence results.
Simpler but more reliable for basic text extraction.

Note: pdfplumber is an optional dependency. If not installed this module can still
be imported; the parser will raise PDFParseError when .parse() is called.
"""

from pathlib import Path
from typing import List

import structlog

from ...domain.models.parsed_document import (
    ParsedDocument,
    ParsedSection,
    ParsedTable,
)
from ...domain.ports.pdf_parser_port import AbstractPDFParser
from ...exceptions.pipeline_exceptions import (
    PDFParseError,
    PDFPasswordProtectedError,
)

try:
    import pdfplumber as _pdfplumber  # noqa: F401

    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False


logger = structlog.get_logger(__name__)


class PDFPlumberFallbackParser(AbstractPDFParser):
    """
    PDFPlumber-based fallback parser for simple text extraction.

    Used when Docling fails or produces low-confidence results.
    Trades sophisticated layout understanding for reliability.
    """

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """
        Parse a PDF file using PDFPlumber.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ParsedDocument with extracted content.

        Raises:
            PDFPasswordProtectedError: If PDF is password-protected.
            PDFParseError: If pdfplumber is not installed or parsing fails.
        """
        if not _PDFPLUMBER_AVAILABLE:
            raise PDFParseError(
                doc_id="",
                pdf_name=pdf_path.name,
                cause=ImportError("pdfplumber is not installed; run: pip install pdfplumber"),
            )

        import pdfplumber  # local import guards against missing dependency at call time

        logger.info("pdfplumber_parse_started", pdf_path=str(pdf_path))

        try:
            with pdfplumber.open(pdf_path) as pdf:
                sections = self._extract_sections(pdf)
                tables = self._extract_tables(pdf)

                parsed_doc = ParsedDocument(
                    doc_id="",
                    pdf_name=pdf_path.name,
                    pdf_source_path=str(pdf_path.parent),
                    total_pages=len(pdf.pages),
                    is_ocr_sourced=False,
                    parser_version=self.get_parser_version(),
                    sections=sections,
                    tables=tables,
                    figures=[],
                    parse_confidence_score=0.8,
                    parse_warnings=["Parsed with fallback parser (pdfplumber)"],
                )

                logger.info(
                    "pdfplumber_parse_complete",
                    sections=len(sections),
                    tables=len(tables),
                )

                return parsed_doc

        except PDFParseError:
            raise
        except Exception as exc:
            # Catch pdfminer's PDFPasswordIncorrect by name to avoid importing it
            if type(exc).__name__ == "PDFPasswordIncorrect":
                raise PDFPasswordProtectedError(doc_id="", pdf_name=pdf_path.name)
            logger.error("pdfplumber_parse_failed", error=str(exc), error_type=type(exc).__name__)
            raise PDFParseError(doc_id="", pdf_name=pdf_path.name, cause=exc)

    def get_parser_version(self) -> str:
        """
        Return the pdfplumber version string.

        Returns:
            Version string.
        """
        if _PDFPLUMBER_AVAILABLE:
            import pdfplumber
            return f"pdfplumber-{pdfplumber.__version__}"
        return "pdfplumber-unavailable"

    def _extract_sections(self, pdf) -> List[ParsedSection]:
        """Extract text sections as one ParsedSection per page."""
        sections: List[ParsedSection] = []

        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text or not text.strip():
                continue
            sections.append(
                ParsedSection(
                    section_id=f"page_{page_num}",
                    heading=f"Page {page_num}",
                    depth=1,
                    content=text.strip(),
                    page_numbers=[page_num],
                    token_count=len(text) // 4,
                )
            )

        return sections

    def _extract_tables(self, pdf) -> List[ParsedTable]:
        """Extract tables from all pages."""
        tables: List[ParsedTable] = []
        table_idx = 0

        for page_num, page in enumerate(pdf.pages, start=1):
            for table_data in page.extract_tables():
                if not table_data:
                    continue
                tables.append(
                    ParsedTable(
                        table_id=f"table_{table_idx}",
                        caption=f"Table {table_idx + 1}",
                        markdown_content=self._table_to_markdown(table_data),
                        page_numbers=[page_num],
                        column_count=len(table_data[0]) if table_data else 0,
                        row_count=len(table_data),
                        is_cross_page=False,
                        validation_passed=True,
                    )
                )
                table_idx += 1

        return tables

    @staticmethod
    def _table_to_markdown(table_data: List[List]) -> str:
        """Convert a 2-D list of cell values to a Markdown table string."""
        if not table_data:
            return ""
        header = table_data[0]
        rows = [
            "| " + " | ".join(str(cell or "") for cell in header) + " |",
            "|" + "|".join(["---"] * len(header)) + "|",
        ]
        for row in table_data[1:]:
            rows.append("| " + " | ".join(str(cell or "") for cell in row) + " |")
        return "\n".join(rows)
