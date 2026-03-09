"""
Abstract base class for figure description generation.

Defines the contract for vision LLM services that generate structured textual
descriptions of figures, flowcharts, and diagrams extracted from clinical PDFs.
"""

from abc import ABC, abstractmethod

from ..models.parsed_document import FigureType


class AbstractFigureDescriber(ABC):
    """Interface for figure description implementations."""

    @abstractmethod
    def describe_figure(
        self,
        image: bytes,
        figure_type: FigureType,
        caption: str
    ) -> str:
        """
        Generate a structured textual description of a figure.
        
        Args:
            image: Raw image bytes (PNG or JPEG)
            figure_type: Type of figure (FLOWCHART, DIAGRAM, TABLE_IMAGE, PHOTO)
            caption: The figure caption extracted from the PDF
        
        Returns:
            Structured textual description of the figure, following the format
            specified for the figure type (e.g., decision pathway for flowcharts,
            markdown table for table images)
        
        Raises:
            FigureDescriptionError: If description generation fails
            VisionLLMTimeoutError: If the vision LLM call times out
            VisionLLMQuotaExceededError: If API quota is exceeded
        """
        raise NotImplementedError("Subclasses must implement describe_figure()")
