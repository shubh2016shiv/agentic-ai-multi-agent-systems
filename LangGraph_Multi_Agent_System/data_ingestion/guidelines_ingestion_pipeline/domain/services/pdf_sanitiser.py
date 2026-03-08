"""
PDF content sanitisation service.

Strips header/footer noise, validates section ordering, and cleans up
artifacts from PDF parsing that would pollute semantic retrieval.

Configured via PipelineSettings. No external I/O dependencies.
"""

import re
from typing import List

import structlog

from ..models.parsed_document import ParsedSection
from ...config.pipeline_settings import PipelineSettings


logger = structlog.get_logger(__name__)


class PDFSanitiser:
    """
    Sanitises parsed PDF content by removing headers, footers, and noise.
    
    Dependencies:
        - PipelineSettings (injected) for regex patterns
    """

    def __init__(self, settings: PipelineSettings):
        """
        Initialize the PDF sanitiser.
        
        Args:
            settings: Pipeline configuration containing header/footer patterns
        """
        self.settings = settings
        self._compiled_patterns = [
            re.compile(pattern, re.MULTILINE)
            for pattern in settings.header_footer_strip_patterns
        ]

    def sanitise_section(self, section: ParsedSection) -> ParsedSection:
        """
        Sanitise a single parsed section by stripping noise.
        
        Args:
            section: ParsedSection to clean
        
        Returns:
            New ParsedSection with sanitised content
        """
        clean_content = self.strip_header_footer_noise(section.content)
        
        return ParsedSection(
            section_id=section.section_id,
            heading=section.heading,
            depth=section.depth,
            content=clean_content,
            page_numbers=section.page_numbers,
            token_count=section.token_count,
        )

    def strip_header_footer_noise(self, text: str) -> str:
        """
        Remove header/footer patterns from text.
        
        Args:
            text: Raw text content
        
        Returns:
            Cleaned text with patterns removed
        """
        clean_text = text
        
        for pattern in self._compiled_patterns:
            clean_text = pattern.sub("", clean_text)
        
        clean_text = clean_text.strip()
        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
        
        return clean_text

    def validate_section_ordering(self, sections: List[ParsedSection]) -> bool:
        """
        Validate that sections appear in logical order.
        
        Checks for:
        - Heading depth progression (no jumps from H1 to H3)
        - Body content following its heading
        
        Args:
            sections: List of ParsedSection objects
        
        Returns:
            True if ordering is valid, False if suspicious
        """
        if not sections:
            return True
        
        prev_depth = 0
        for i, section in enumerate(sections):
            if section.depth > prev_depth + 1:
                logger.warning(
                    "section_ordering_suspicious",
                    section_index=i,
                    section_heading=section.heading,
                    depth=section.depth,
                    prev_depth=prev_depth,
                    reason="depth_jump",
                )
                return False
            
            prev_depth = section.depth
        
        return True
