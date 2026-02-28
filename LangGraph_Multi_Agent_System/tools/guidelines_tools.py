"""
Clinical Guidelines Domain Tools
==================================
Tool functions scoped to the GUIDELINES AGENT domain.

What is a "Guidelines Agent"?
    A guidelines agent looks up evidence-based clinical practice guidelines
    (e.g., GOLD for COPD, KDIGO for CKD, AHA/ACC for Hypertension) and
    returns structured recommendations with evidence grades.

    This agent does NOT interpret the guidelines — it retrieves them.
    Interpretation is left to the specialist agent that requested the lookup.

Why is this a separate module?
    Even though there is only one tool here, it is isolated because:
    1. The data source (CLINICAL_GUIDELINE_DATABASE) is distinct from the
       drug and symptom data used by other tool modules.
    2. In a production system, this tool would be backed by a different
       service (e.g., a FHIR ClinicalUseDefinition API), so isolating
       it makes the swap straightforward.
    3. The Guidelines Agent in Script 03 Section 1 is shown as a separate
       agent with its own tool scope.

Data source:
    All static reference data is imported from _clinical_knowledge_base.py.
"""

import json
import logging

from langchain_core.tools import tool

from observability.decorators import observe_tool
from tools._clinical_knowledge_base import CLINICAL_GUIDELINE_DATABASE

logger = logging.getLogger(__name__)


# ============================================================
# Tool 1: Clinical Guideline Lookup
# ============================================================

@tool
@observe_tool(tool_name="lookup_clinical_guideline")
def lookup_clinical_guideline(condition: str, topic: str = "treatment") -> str:
    """
    Look up clinical guideline recommendations for a medical condition.

    Returns evidence-based recommendations from major clinical guidelines
    (GOLD for COPD, KDIGO for CKD, AHA/ACC for Hypertension, APA for
    Depression, etc.) with evidence grades.

    Args:
        condition: Medical condition (e.g., "COPD", "CKD", "Hypertension").
        topic: Aspect to look up (e.g., "treatment", "diagnosis", "monitoring").

    Returns:
        JSON string with guideline recommendations and evidence grades.
    """
    condition_lower = condition.lower().strip()
    guideline = CLINICAL_GUIDELINE_DATABASE.get(condition_lower)

    if not guideline:
        return json.dumps({
            "condition": condition,
            "found": False,
            "message": (
                f"No guideline found for '{condition}'. "
                f"Available conditions: {list(CLINICAL_GUIDELINE_DATABASE.keys())}"
            ),
        })

    topic_data = guideline.get(topic.lower(), {})
    if not topic_data:
        available_topics = [key for key in guideline.keys() if key != "source"]
        return json.dumps({
            "condition": condition,
            "source": guideline["source"],
            "found": False,
            "message": (
                f"Topic '{topic}' not found for {condition}. "
                f"Available topics: {available_topics}"
            ),
        })

    result = {
        "condition": condition,
        "source": guideline["source"],
        "topic": topic,
        "found": True,
        "recommendations": topic_data,
    }

    logger.info(f"Guideline lookup: {condition}/{topic} → {guideline['source']}")
    return json.dumps(result, indent=2)
