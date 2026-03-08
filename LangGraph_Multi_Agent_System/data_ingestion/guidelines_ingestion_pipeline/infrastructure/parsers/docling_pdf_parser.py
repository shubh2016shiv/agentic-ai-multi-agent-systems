"""
Docling PDF parser implementation.

Uses Docling's DocumentConverter to convert PDFs to structured content.
Follows the same export_to_markdown() approach used by the existing
document_processing.pdf_processor module (proven against these PDFs).

OCR is disabled because all guideline PDFs in this system are born-digital
(native text layer present). Enabling OCR would load RapidOCR neural models
onto CPU, causing bad_alloc errors on pages with many images.
"""

import re
from pathlib import Path
from typing import List

import structlog
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from ...domain.models.parsed_document import (
    BoundingBox,
    FigureType,
    ParsedDocument,
    ParsedFigure,
    ParsedSection,
    ParsedTable,
)
from ...domain.ports.pdf_parser_port import AbstractPDFParser
from ...exceptions.pipeline_exceptions import (
    PDFParseError,
    PDFPasswordProtectedError,
)


logger = structlog.get_logger(__name__)

_HEADING_DEPTH = {
    1: re.compile(r"^# (.+)$"),
    2: re.compile(r"^## (.+)$"),
    3: re.compile(r"^### (.+)$"),
}
_TABLE_HEADER_RE = re.compile(r"^\|.+\|$")


def _build_converter() -> DocumentConverter:
    """
    Build a DocumentConverter configured for born-digital clinical PDFs.

    OCR is disabled because guideline PDFs always carry a native text layer.
    This avoids loading RapidOCR's ONNX models on CPU, which caused
    std::bad_alloc errors when processing large (200+ page) documents.
    """
    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.do_ocr = False         # no OCR needed for digital PDFs
    pipeline_opts.do_table_structure = True  # keep TableFormer for table extraction

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
        }
    )


class DoclingPDFParser(AbstractPDFParser):
    """
    Docling-based PDF parser for complex clinical documents.

    Uses export_to_markdown() as the stable Docling public API, then parses
    heading-delimited sections, extracts tables from TableItem objects,
    and enumerates PictureItem objects for downstream figure processing.
    """

    def __init__(self) -> None:
        """Initialize the Docling DocumentConverter with OCR disabled."""
        self._converter = _build_converter()

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """
        Parse a PDF file into a structured ParsedDocument.

        Args:
            pdf_path: Absolute or relative path to the PDF file.

        Returns:
            ParsedDocument with sections, tables, and figure stubs.

        Raises:
            PDFPasswordProtectedError: If PDF is password-protected.
            PDFParseError: For all other parsing failures.
        """
        logger.info("docling_parse_started", pdf_path=str(pdf_path))

        try:
            result = self._converter.convert(str(pdf_path))
        except PermissionError as exc:
            raise PDFPasswordProtectedError(doc_id="", pdf_name=pdf_path.name, cause=exc)
        except Exception as exc:
            logger.error("docling_conversion_failed", error=str(exc))
            raise PDFParseError(doc_id="", pdf_name=pdf_path.name, cause=exc)

        try:
            markdown = result.document.export_to_markdown()
            total_pages = len(result.document.pages)

            sections = self._parse_sections_from_markdown(markdown)
            tables = self._extract_tables(result)
            figures = self._extract_figures(result)

            parsed_doc = ParsedDocument(
                doc_id="",  # Will be set by the orchestrator after hashing
                pdf_name=pdf_path.name,
                pdf_source_path=str(pdf_path.parent),
                total_pages=total_pages,
                is_ocr_sourced=False,
                parser_version=self.get_parser_version(),
                sections=sections,
                tables=tables,
                figures=figures,
                parse_confidence_score=1.0,
                parse_warnings=[],
            )

            logger.info(
                "docling_parse_complete",
                sections=len(sections),
                tables=len(tables),
                figures=len(figures),
                total_pages=total_pages,
            )

            return parsed_doc

        except Exception as exc:
            logger.error("docling_extraction_failed", error=str(exc))
            raise PDFParseError(doc_id="", pdf_name=pdf_path.name, cause=exc)

    def get_parser_version(self) -> str:
        """
        Return the parser version string.

        Returns:
            Human-readable version identifier.
        """
        try:
            from docling import __version__ as dv
            return f"docling-{dv}"
        except Exception:
            return "docling-unknown"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_sections_from_markdown(self, markdown: str) -> List[ParsedSection]:
        """
        Parse markdown into a list of ParsedSection objects.

        Splits at every H1 / H2 / H3 heading. Body text between headings
        becomes the section content. Page numbers are estimated as 0 (not
        available from the markdown export; bounding-box extraction is Phase 4+).
        """
        sections: List[ParsedSection] = []
        current_heading = "Introduction"
        current_depth = 1
        current_lines: List[str] = []

        def _flush(heading: str, depth: int, lines: List[str]) -> None:
            content = "\n".join(lines).strip()
            if not content:
                return
            section_id = f"section_{len(sections)}"
            sections.append(
                ParsedSection(
                    section_id=section_id,
                    heading=heading,
                    depth=depth,
                    content=content,
                    page_numbers=[0],
                    token_count=len(content) // 4,
                )
            )

        for line in markdown.splitlines():
            matched = False
            for depth, pattern in _HEADING_DEPTH.items():
                m = pattern.match(line)
                if m:
                    _flush(current_heading, current_depth, current_lines)
                    current_heading = m.group(1).strip()
                    current_depth = depth
                    current_lines = []
                    matched = True
                    break
            if not matched:
                current_lines.append(line)

        _flush(current_heading, current_depth, current_lines)

        return sections

    def _extract_tables(self, result) -> List[ParsedTable]:
        """
        Extract tables from the Docling ConversionResult.

        Uses each TableItem's export_to_markdown(doc=...) method for a clean
        Markdown representation. The `doc` argument is required in newer
        Docling versions to resolve cross-references correctly.
        """
        tables: List[ParsedTable] = []

        for idx, table in enumerate(result.document.tables):
            try:
                md_content = table.export_to_markdown(doc=result.document)
                rows = [r for r in md_content.splitlines() if _TABLE_HEADER_RE.match(r)]
                row_count = max(0, len(rows) - 1)  # subtract header separator row
                col_count = len(rows[0].split("|")) - 2 if rows else 0

                page_no = 0
                if hasattr(table, "prov") and table.prov:
                    page_no = getattr(table.prov[0], "page_no", 0)

                tables.append(
                    ParsedTable(
                        table_id=f"table_{idx}",
                        caption=f"Table {idx + 1}",
                        markdown_content=md_content,
                        page_numbers=[page_no],
                        column_count=col_count,
                        row_count=row_count,
                        is_cross_page=False,
                        validation_passed=True,
                    )
                )
            except Exception as exc:
                logger.warning("table_extraction_failed", table_idx=idx, error=str(exc))

        return tables

    def _extract_figures(self, result) -> List[ParsedFigure]:
        """
        Extract figure stubs from PictureItem objects.

        Actual descriptions are populated later by the FigureDescriber service.
        """
        figures: List[ParsedFigure] = []

        for idx, picture in enumerate(result.document.pictures):
            try:
                page_no = 0
                if hasattr(picture, "prov") and picture.prov:
                    page_no = getattr(picture.prov[0], "page_no", 0)

                caption = f"Figure {idx + 1}"
                if hasattr(picture, "caption") and picture.caption:
                    caption = str(picture.caption)

                figures.append(
                    ParsedFigure(
                        figure_id=f"figure_{idx}",
                        caption=caption,
                        page_number=page_no,
                        bounding_box=BoundingBox(x=0.0, y=0.0, width=0.0, height=0.0),
                        figure_type=FigureType.DIAGRAM,
                        description=None,
                        description_pending=True,
                    )
                )
            except Exception as exc:
                logger.warning("figure_extraction_failed", figure_idx=idx, error=str(exc))

        return figures
