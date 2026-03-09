"""
Input Guardrails
==================
Validates and filters user/agent inputs BEFORE they reach the LLM.
Prevents prompt injection, off-topic queries, and PII leakage.

Input guardrails run as the FIRST step in the agent pipeline, acting
as a gatekeeper that blocks harmful inputs before they consume tokens.

Where This Fits in the MAS Architecture
-----------------------------------------
Position in the pipeline:

    User/Agent Input
         |
    [input_guardrails]  ← YOU ARE HERE
         |
    [LLM Agent]
         |
    [output_guardrails]
         |
    Deliver or Escalate

Why inputs need guardrails:
    1. Token cost: Bad inputs still consume LLM tokens even if the output
       is useless. Block at the gate = zero token waste.
    2. Prompt injection: Malicious users can try to override the system
       prompt ("ignore previous instructions and..."). Catch it before
       the LLM sees it.
    3. PII compliance: Medical systems must not log or process raw PII
       without consent. Strip or block it at the input stage.
    4. Scope enforcement: A CDSS should only answer medical questions.
       Off-topic queries waste resources and confuse users.

Pattern scripts demonstrate how to wire validate_input() into a graph:
    scripts/guardrails/input_validation.py  — Pattern A: binary routing
    scripts/guardrails/layered_validation.py — Pattern D: full pipeline

Key guardrails:
    1. Medical Scope Check: Ensures queries are medical-related
    2. PII Filter: Detects and flags personally identifiable information
    3. Prompt Injection Detection: Blocks prompt manipulation attempts
    4. Input Length Validation: Prevents context overflow attacks

Usage:
    from guardrails.input_guardrails import validate_input

    result = validate_input(user_query)
    if not result["passed"]:
        return f"Blocked: {result['reason']}"
"""

import re
import logging
from typing import Any

from core.exceptions import GuardrailTripped

logger = logging.getLogger(__name__)


# ============================================================
# Pattern definitions
# ============================================================

# PII patterns (simplified — production systems use NER models)
PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "email": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "date_of_birth": r"\b(DOB|date of birth|born on)[:\s]*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b",
}

# Prompt injection indicators
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"system\s*prompt",
    r"forget\s+(everything|all|your)",
    r"jailbreak",
    r"pretend\s+to\s+be",
    r"act\s+as\s+if\s+you",
    r"override\s+(safety|guardrail|instruction)",
]

# Non-medical keywords that indicate off-topic queries
OFF_TOPIC_KEYWORDS = [
    "recipe", "cooking", "sports score", "stock price",
    "weather forecast", "movie review", "homework",
    "write a poem", "tell a joke", "code review",
]

# Medical keywords that confirm on-topic queries
MEDICAL_KEYWORDS = [
    "patient", "symptom", "diagnosis", "medication", "drug",
    "treatment", "vitals", "lab", "blood", "pain", "disease",
    "condition", "prescription", "dosage", "allergy", "history",
    "clinical", "medical", "health", "cough", "fever",
    "blood pressure", "heart rate", "surgery", "therapy",
]


def validate_input(
    text: str,
    check_pii: bool = True,
    check_injection: bool = True,
    check_scope: bool = True,
    max_length: int = 5000,
) -> dict[str, Any]:
    """
    Run all input guardrails on the provided text.

    Returns a result dict indicating whether the input passed or failed,
    and if failed, which guardrail tripped and why.

    Args:
        text: The input text to validate.
        check_pii: Whether to check for PII.
        check_injection: Whether to check for prompt injection.
        check_scope: Whether to check for medical relevance.
        max_length: Maximum allowed input length.

    Returns:
        Dict: {"passed": bool, "guardrail": str, "reason": str, "details": dict}
    """
    result = {"passed": True, "guardrail": None, "reason": None, "details": {}}

    # 1. Length check
    if len(text) > max_length:
        result.update({
            "passed": False,
            "guardrail": "input_length",
            "reason": f"Input exceeds maximum length ({len(text)} > {max_length})",
            "details": {"length": len(text), "max": max_length},
        })
        logger.warning(f"Input guardrail TRIPPED: length ({len(text)} > {max_length})")
        return result

    # 2. Empty check
    if not text.strip():
        result.update({
            "passed": False,
            "guardrail": "empty_input",
            "reason": "Input is empty or whitespace-only",
        })
        return result

    # 3. Prompt injection check
    if check_injection:
        injection_result = detect_prompt_injection(text)
        if injection_result["detected"]:
            result.update({
                "passed": False,
                "guardrail": "prompt_injection",
                "reason": "Potential prompt injection detected",
                "details": injection_result,
            })
            logger.warning(f"Input guardrail TRIPPED: prompt injection — {injection_result['pattern']}")
            return result

    # 4. PII check
    if check_pii:
        pii_result = detect_pii(text)
        if pii_result["found"]:
            result.update({
                "passed": False,
                "guardrail": "pii_detected",
                "reason": f"PII detected: {', '.join(pii_result['types'])}",
                "details": pii_result,
            })
            logger.warning(f"Input guardrail TRIPPED: PII — types: {pii_result['types']}")
            return result

    # 5. Medical scope check
    if check_scope:
        scope_result = check_medical_scope(text)
        if not scope_result["in_scope"]:
            result.update({
                "passed": False,
                "guardrail": "out_of_scope",
                "reason": "Query does not appear to be medically relevant",
                "details": scope_result,
            })
            logger.info(f"Input guardrail TRIPPED: out of medical scope")
            return result

    logger.debug("Input guardrails: all checks PASSED")
    return result


def detect_pii(text: str) -> dict[str, Any]:
    """
    Detect personally identifiable information in text.

    Args:
        text: Text to scan for PII patterns.

    Returns:
        Dict: {"found": bool, "types": list, "count": int}
    """
    found_types = []
    total_count = 0

    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            found_types.append(pii_type)
            total_count += len(matches)

    return {
        "found": len(found_types) > 0,
        "types": found_types,
        "count": total_count,
    }


def detect_prompt_injection(text: str) -> dict[str, Any]:
    """
    Detect prompt injection attempts in the input.

    Args:
        text: Text to scan for injection patterns.

    Returns:
        Dict: {"detected": bool, "pattern": str | None}
    """
    text_lower = text.lower()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return {"detected": True, "pattern": pattern}

    return {"detected": False, "pattern": None}


def check_medical_scope(text: str) -> dict[str, Any]:
    """
    Check if the query is within the medical domain.

    Uses keyword matching to determine if the input is medically relevant.
    A query that contains off-topic keywords AND no medical keywords
    is flagged as out of scope.

    Args:
        text: The query text to check.

    Returns:
        Dict: {"in_scope": bool, "medical_keywords_found": list, "off_topic_keywords_found": list}
    """
    text_lower = text.lower()

    medical_found = [kw for kw in MEDICAL_KEYWORDS if kw in text_lower]
    off_topic_found = [kw for kw in OFF_TOPIC_KEYWORDS if kw in text_lower]

    # In scope if ANY medical keyword is present, regardless of off-topic words
    in_scope = len(medical_found) > 0 or len(off_topic_found) == 0

    return {
        "in_scope": in_scope,
        "medical_keywords_found": medical_found,
        "off_topic_keywords_found": off_topic_found,
    }
