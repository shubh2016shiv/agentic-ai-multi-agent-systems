"""
Confidence Guardrails
======================
Threshold-based routing using an LLM agent's self-assessed confidence
score. This is a DIFFERENT kind of guardrail from input/output content
checks — it acts on CERTAINTY rather than content.

Where This Fits in the MAS Architecture
-----------------------------------------
In the guardrail layer of a multi-agent pipeline, there are two orthogonal
concerns:

    Content guardrails (input_guardrails.py, output_guardrails.py)
        → "Is the content safe, in-scope, complete?"

    Confidence guardrails (this file)
        → "Is the agent CERTAIN ENOUGH to deliver without review?"

Confidence gating is a post-processing step that sits AFTER the LLM call
but BEFORE delivery:

    [LLM agent] → generate response (includes confidence score)
         |
    [confidence_gate] → compare score to threshold
         |
    +---------+----------+
    |                    |
    deliver           escalate
    (>= threshold)  (< threshold)

Why self-assessed confidence?
    LLMs can report their own uncertainty. For ambiguous clinical cases
    (missing labs, unclear history) the model's confidence will be lower
    than for clear-cut presentations. This is a signal worth capturing.

    Limitation: LLMs can be overconfident ("hallucinate with confidence").
    Always combine confidence gating with deterministic content checks.

Stacking with content guardrails (recommended production pattern):
    1. validate_input()         ← content (injection, PII, scope)
    2. [LLM call]
    3. validate_output()        ← content (prohibited terms, disclaimers)
    4. extract_confidence()     ← certainty gating (this module)
    5. deliver or escalate

Usage:
    from guardrails.confidence_guardrails import extract_confidence, gate_on_confidence

    response_text = llm.invoke(...)
    confidence = extract_confidence(response_text)
    result = gate_on_confidence(confidence, threshold=0.75)
    if result["action"] == "escalate":
        send_to_human_review(response_text)
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# Confidence Extraction
# ============================================================

def extract_confidence(text: str, default: float = 0.5) -> float:
    """
    Extract a self-assessed confidence score from an LLM's response text.

    Concept — Why prompts embed confidence:
        When you include "state your confidence as a decimal 0.0–1.0"
        in the system prompt, the LLM produces a parseable confidence
        marker. This avoids a second LLM call just to score the first.

    The agent must be prompted to include a line like:
        "Confidence: 0.85"  or  "Confidence: 85%"

    If no confidence marker is found, a neutral default (0.5) is returned.
    This is a "fail-safe" default: if the agent didn't report confidence,
    we assume moderate certainty and let the threshold decide.

    Args:
        text: Full text of the agent's response.
        default: Confidence to assume when no score is found. Default 0.5.

    Returns:
        Float in [0.0, 1.0]. Values reported as percentages are normalised.

    Example:
        >>> extract_confidence("Assessment: COPD exacerbation.\\nConfidence: 0.87")
        0.87
        >>> extract_confidence("Assessment: unclear. Confidence: 43%")
        0.43
        >>> extract_confidence("No confidence reported.")
        0.5
    """
    patterns = [
        r"[Cc]onfidence[:\s]+(\d+\.?\d*)%",    # "Confidence: 85%"
        r"[Cc]onfidence[:\s]+(\d+\.?\d*)",       # "Confidence: 0.85"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            if value > 1.0:
                value = value / 100.0
            clamped = min(max(value, 0.0), 1.0)
            logger.debug(f"Extracted confidence: {clamped:.2f}")
            return clamped

    logger.debug(f"No confidence found in response — using default {default}")
    return default


# ============================================================
# Confidence Gating (Threshold Decision)
# ============================================================

def gate_on_confidence(
    confidence: float,
    threshold: float = 0.75,
    label_above: str = "deliver",
    label_below: str = "escalate",
) -> dict[str, Any]:
    """
    Apply a threshold gate to a confidence score and return a routing decision.

    Concept — Configurable thresholds per deployment context:
        Different medical contexts require different certainty levels:

            Emergency triage    → threshold=0.85 (escalate often, safety critical)
            Drug interaction    → threshold=0.90 (almost always review)
            Patient education   → threshold=0.60 (lower risk, less strict)
            Clinical research   → threshold=0.30 (tolerate uncertainty)

        The threshold is a deployment decision, not a code decision.
        This function keeps the threshold out of graph node logic — pass
        it as state at invocation time so the SAME graph works for all
        contexts.

    Args:
        confidence: Score from extract_confidence() in [0.0, 1.0].
        threshold: Minimum acceptable confidence. Default 0.75.
        label_above: Routing label when confidence >= threshold. Default "deliver".
        label_below: Routing label when confidence < threshold. Default "escalate".

    Returns:
        Dict with keys:
            "action"      — label_above or label_below
            "confidence"  — the raw confidence value
            "threshold"   — the threshold used
            "passed"      — bool, True if confidence >= threshold

    Example:
        >>> gate_on_confidence(0.87, threshold=0.75)
        {"action": "deliver", "confidence": 0.87, "threshold": 0.75, "passed": True}
        >>> gate_on_confidence(0.42, threshold=0.75)
        {"action": "escalate", "confidence": 0.42, "threshold": 0.75, "passed": False}
    """
    passed = confidence >= threshold
    action = label_above if passed else label_below

    logger.debug(
        f"Confidence gate: {confidence:.2f} vs threshold {threshold:.2f} → {action}"
    )

    return {
        "action": action,
        "confidence": confidence,
        "threshold": threshold,
        "passed": passed,
    }


# ============================================================
# Convenience: Extract + Gate in One Call
# ============================================================

def check_confidence(
    text: str,
    threshold: float = 0.75,
) -> dict[str, Any]:
    """
    Extract confidence from text AND apply threshold gate in one call.

    Convenience wrapper used when you have the raw LLM response and
    want a single function call that returns the routing decision.

    Args:
        text: Raw LLM response text.
        threshold: Confidence threshold. Default 0.75.

    Returns:
        Same dict as gate_on_confidence(), plus "raw_text_length" key.

    Example:
        from guardrails.confidence_guardrails import check_confidence

        result = check_confidence(llm_response, threshold=0.80)
        if result["action"] == "escalate":
            queue_for_human_review(llm_response)
    """
    confidence = extract_confidence(text)
    gate_result = gate_on_confidence(confidence, threshold=threshold)
    gate_result["raw_text_length"] = len(text)
    return gate_result
