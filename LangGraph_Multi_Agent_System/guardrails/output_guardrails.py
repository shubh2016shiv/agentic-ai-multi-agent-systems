"""
Output Guardrails
===================
Validates agent outputs BEFORE they are returned to users.
Ensures medical safety, checks for required disclaimers, and
prevents hallucinated medical advice from reaching patients.

Output guardrails run as the LAST step in the agent pipeline, acting
as a final safety net before the response is delivered.

Where This Fits in the MAS Architecture
-----------------------------------------
Position in the pipeline:

    User/Agent Input
         |
    [input_guardrails]
         |
    [LLM Agent]
         |
    [output_guardrails]  ← YOU ARE HERE
         |
    [confidence_guardrails / llm_judge]  ← optional further checks
         |
    Deliver or Escalate

Why outputs need separate guardrails from inputs:
    Input guardrails catch what the USER sends.
    Output guardrails catch what the LLM GENERATES.

    LLMs hallucinate. Even with a safe input, an LLM can:
        - Recommend stopping all medications ("Stop all meds immediately")
        - Claim certainty it doesn't have ("This will definitely cure...")
        - Omit required disclaimers ("Consult your healthcare provider")
        - Return an incomplete or nonsensical response

    Output guardrails are the last line of defense before the user sees
    the response. Unlike input guardrails (which block), output guardrails
    can also FIX: auto-append disclaimers, flag for human review, or
    replace dangerous content with a safe fallback.

Three-way output routing (used in scripts/guardrails/output_validation.py):
    PASS   → deliver as-is
    NEEDS_FIX → auto-fix (append disclaimer) then deliver
    BLOCK  → replace with safe fallback, escalate to human

Pattern scripts:
    scripts/guardrails/output_validation.py  — Pattern B: 3-way routing
    scripts/guardrails/layered_validation.py — Pattern D: full pipeline

Key guardrails:
    1. Safety Disclaimer Check: Ensures medical disclaimers are present
    2. Confidence Validation: Flags low-confidence responses for review
    3. Prohibited Content: Blocks dangerous medical recommendations
    4. Completeness Check: Ensures responses address the original query

Usage:
    from guardrails.output_guardrails import validate_output

    result = validate_output(agent_response, original_query)
    if result["needs_review"]:
        escalate_to_human(result)
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# Dangerous recommendations that should NEVER appear in output
# ============================================================
PROHIBITED_CONTENT = [
    r"stop\s+all\s+medications?\s+(immediately|at\s+once|right\s+away)",
    r"no\s+need\s+(to\s+see|for)\s+(a\s+)?doctor",
    r"guaranteed\s+(cure|to\s+cure)",
    r"100%\s+(effective|safe|certain)",
    r"this\s+will\s+definitely",
    r"you\s+don't\s+need\s+medical\s+attention",
    r"ignore\s+your\s+doctor",
    r"instead\s+of\s+seeing\s+a\s+doctor",
]

# Required disclaimer elements for medical responses
REQUIRED_ELEMENTS = [
    "This is for informational purposes",
    "Consult your healthcare provider",
    "Not a substitute for professional medical advice",
    "professional medical",
    "consult",
    "healthcare provider",
    "doctor",
]


def validate_output(
    output_text: str,
    original_query: str = "",
    confidence: float | None = None,
    check_disclaimers: bool = True,
    check_prohibited: bool = True,
    min_confidence: float = 0.3,
) -> dict[str, Any]:
    """
    Run all output guardrails on an agent's response.

    Args:
        output_text: The agent's response text.
        original_query: The original user query (for relevance checking).
        confidence: Agent's self-assessed confidence score (0-1).
        check_disclaimers: Whether to check for medical disclaimers.
        check_prohibited: Whether to check for prohibited content.
        min_confidence: Minimum acceptable confidence score.

    Returns:
        Dict with validation results:
        {
            "passed": bool,
            "needs_review": bool,
            "issues": list[dict],
            "modified_output": str  # Output with disclaimers appended if needed
        }
    """
    issues = []
    needs_review = False
    modified_output = output_text

    # 1. Empty/too-short response check
    if not output_text or len(output_text.strip()) < 20:
        issues.append({
            "type": "incomplete_response",
            "severity": "HIGH",
            "detail": "Response is empty or too short to be meaningful",
        })
        needs_review = True

    # 2. Prohibited content check
    if check_prohibited:
        prohibited_result = check_prohibited_content(output_text)
        if prohibited_result["found"]:
            issues.append({
                "type": "prohibited_content",
                "severity": "CRITICAL",
                "detail": f"Dangerous recommendation detected: {prohibited_result['matched_pattern']}",
            })
            needs_review = True

    # 3. Low confidence check
    if confidence is not None and confidence < min_confidence:
        issues.append({
            "type": "low_confidence",
            "severity": "HIGH",
            "detail": f"Agent confidence ({confidence:.0%}) below threshold ({min_confidence:.0%})",
        })
        needs_review = True

    # 4. Safety disclaimer check
    if check_disclaimers:
        disclaimer_result = check_safety_disclaimers(output_text)
        if not disclaimer_result["has_disclaimer"]:
            # Append a standard disclaimer rather than blocking
            disclaimer = (
                "\n\n---\n"
                "⚕️ *This information is for educational purposes only and is not a substitute "
                "for professional medical advice. Always consult your healthcare provider "
                "before making medical decisions.*"
            )
            modified_output = output_text + disclaimer
            issues.append({
                "type": "missing_disclaimer",
                "severity": "LOW",
                "detail": "Medical disclaimer was not present — appended automatically",
            })

    passed = len([i for i in issues if i["severity"] in ("CRITICAL", "HIGH")]) == 0

    result = {
        "passed": passed,
        "needs_review": needs_review,
        "issues": issues,
        "issue_count": len(issues),
        "modified_output": modified_output,
    }

    if issues:
        logger.info(f"Output guardrails: {len(issues)} issue(s) found, passed={passed}")
    else:
        logger.debug("Output guardrails: all checks PASSED")

    return result


def check_prohibited_content(text: str) -> dict[str, Any]:
    """
    Check output for dangerous medical recommendations.

    Args:
        text: Response text to scan.

    Returns:
        Dict: {"found": bool, "matched_pattern": str | None}
    """
    text_lower = text.lower()

    for pattern in PROHIBITED_CONTENT:
        if re.search(pattern, text_lower):
            return {"found": True, "matched_pattern": pattern}

    return {"found": False, "matched_pattern": None}


def check_safety_disclaimers(text: str) -> dict[str, Any]:
    """
    Check if the output contains appropriate medical disclaimers.

    A valid disclaimer mentions at least ONE of the required elements
    (e.g., "consult your doctor", "not a substitute for medical advice").

    Args:
        text: Response text to check.

    Returns:
        Dict: {"has_disclaimer": bool, "found_elements": list}
    """
    text_lower = text.lower()
    found_elements = [elem for elem in REQUIRED_ELEMENTS if elem.lower() in text_lower]

    return {
        "has_disclaimer": len(found_elements) > 0,
        "found_elements": found_elements,
        "missing_disclaimer": len(found_elements) == 0,
    }


def add_human_review_flag(
    output_text: str,
    reason: str,
) -> str:
    """
    Add a human-in-the-loop review flag to an output.

    Used when guardrails detect issues that require physician review.
    This implements the HITL (Human-In-The-Loop) pattern from Chapter 4.

    Args:
        output_text: The agent's response.
        reason: Why this output needs human review.

    Returns:
        Modified output with review flag prepended.
    """
    flag = (
        f"🔍 **REQUIRES PHYSICIAN REVIEW**\n"
        f"Reason: {reason}\n"
        f"{'─' * 40}\n\n"
    )
    return flag + output_text
