"""
Vision LLM figure describer implementation.

Generates structured textual descriptions of figures, flowcharts, and diagrams
using a vision-capable LLM (e.g., Claude with vision, GPT-4V).
"""

import base64
import os

import structlog

from ...domain.models.parsed_document import FigureType
from ...domain.ports.figure_describer_port import AbstractFigureDescriber
from ...exceptions.pipeline_exceptions import (
    FigureDescriptionError,
    VisionLLMQuotaExceededError,
    VisionLLMTimeoutError,
)
from ...utils.retry_utils import retry_with_exponential_backoff


logger = structlog.get_logger(__name__)


class VisionLLMFigureDescriber(AbstractFigureDescriber):
    """
    Vision LLM-based figure description generator.
    
    Uses a vision-capable LLM to generate structured textual descriptions
    of clinical flowcharts, diagrams, and figures.
    """

    def __init__(self, model_name: str = "claude-sonnet-4-20250514", timeout: int = 60):
        """
        Initialize the vision LLM figure describer.
        
        Args:
            model_name: Name of the vision LLM model
            timeout: Timeout in seconds for API calls
        """
        self.model_name = model_name
        self.timeout = timeout
        
        try:
            import anthropic
            
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning(
                    "anthropic_api_key_not_found",
                    message="ANTHROPIC_API_KEY environment variable not set. Vision LLM will fail.",
                )
            
            self.client = anthropic.Anthropic(api_key=api_key)
            self.anthropic = anthropic
            
            logger.info(
                "vision_llm_initialized",
                model_name=model_name,
                timeout=timeout,
            )
            
        except ImportError as e:
            logger.error(
                "anthropic_dependency_missing",
                error=str(e),
            )
            raise FigureDescriptionError(
                figure_id="init",
                cause=Exception(f"Failed to import anthropic: {e}. Install with: pip install anthropic")
            ) from e

    @retry_with_exponential_backoff(
        max_attempts=3,
        retryable_exceptions=(ConnectionError, TimeoutError),
    )
    def describe_figure(
        self,
        image: bytes,
        figure_type: FigureType,
        caption: str
    ) -> str:
        """
        Generate a structured description of a figure.
        
        Args:
            image: Raw image bytes (PNG or JPEG)
            figure_type: Type of figure (FLOWCHART, DIAGRAM, etc.)
            caption: Figure caption from PDF
        
        Returns:
            Structured textual description
        
        Raises:
            VisionLLMTimeoutError: If API call times out
            VisionLLMQuotaExceededError: If API quota exceeded
            FigureDescriptionError: For other failures
        """
        logger.info(
            "vision_llm_describe_started",
            figure_type=figure_type.value,
            caption=caption[:50],
        )
        
        try:
            image_b64 = base64.standard_b64encode(image).decode("utf-8")
            
            media_type = "image/png"
            
            prompt = self._build_clinical_figure_prompt(figure_type, caption)
            
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                timeout=float(self.timeout),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )
            
            description = message.content[0].text
            
            logger.info(
                "vision_llm_describe_complete",
                description_length=len(description),
            )
            
            return description
            
        except self.anthropic.APITimeoutError as e:
            logger.error(
                "vision_llm_timeout",
                caption=caption[:50],
                timeout=self.timeout,
            )
            raise VisionLLMTimeoutError(figure_id=caption, cause=e) from e
        
        except self.anthropic.RateLimitError as e:
            logger.error(
                "vision_llm_quota_exceeded",
                caption=caption[:50],
            )
            raise VisionLLMQuotaExceededError(figure_id=caption, cause=e) from e
        
        except Exception as e:
            logger.error(
                "vision_llm_describe_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise FigureDescriptionError(figure_id=caption, cause=e) from e

    def _build_clinical_figure_prompt(self, figure_type: FigureType, caption: str) -> str:
        """
        Build a specialized prompt for clinical figure description.
        
        Args:
            figure_type: Type of figure
            caption: Figure caption
        
        Returns:
            Prompt text
        """
        base_prompt = f"""You are analyzing a clinical guideline figure with the caption: "{caption}"

This is a {figure_type.value} from a medical guideline document.

Please provide a detailed, structured textual description of this figure that captures:

1. **Main Purpose**: What clinical decision, pathway, or information does this figure convey?
2. **Key Elements**: List the main components (boxes, nodes, decision points, data ranges, etc.)
3. **Flow/Structure**: If applicable, describe the logical flow or organization
4. **Clinical Relevance**: What specific clinical actions, thresholds, or recommendations are shown?

Your description should be comprehensive enough that a clinician could understand the figure's content and clinical guidance without seeing the image.

Format your response as clear, structured prose (not JSON). Be specific about medical terminology, values, thresholds, and decision criteria shown."""

        return base_prompt
