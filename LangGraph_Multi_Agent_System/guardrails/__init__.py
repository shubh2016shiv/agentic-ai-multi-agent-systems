"""
Guardrails Package
===================
Safety layers that validate inputs and outputs at every stage of the
multi-agent pipeline. Prevents unsafe, off-topic, or low-quality content
from entering or leaving the agent system.

Where This Fits in the MAS Architecture
-----------------------------------------
Guardrails are the SAFETY BOUNDARY of the pipeline. They operate
independently of the graph topology and can be composed in layers:

    User/Agent Input
         |
    [input_guardrails]    ← content: PII, injection, scope checks
         |
    [LLM Agent]           ← core reasoning
         |
    [output_guardrails]   ← content: prohibited terms, disclaimers
         |
    [confidence_guardrails] ← certainty: is the agent confident enough?
         |
    [llm_judge_guardrails]  ← semantic: is the reasoning sound? (optional)
         |
    Deliver or Escalate

Guardrail Types (by module):
    input_guardrails.py      — validates inputs BEFORE the LLM
    output_guardrails.py     — validates outputs AFTER the LLM
    confidence_guardrails.py — routes based on self-assessed certainty
    llm_judge_guardrails.py  — semantic evaluation via a second LLM

Pattern scripts (scripts/guardrails/) show how to wire these into graphs:
    Pattern A — input_validation.py      : binary routing on input checks
    Pattern B — output_validation.py     : 3-way routing on output checks
    Pattern C — confidence_gating.py     : threshold-based routing
    Pattern D — layered_validation.py    : full input → agent → output pipeline
    Pattern E — llm_as_judge.py          : semantic evaluation with JudgeVerdict
"""

# Input guardrails — validate BEFORE the LLM call
from guardrails.input_guardrails import (
    validate_input,
    check_medical_scope,
    detect_pii,
    detect_prompt_injection,
)

# Output guardrails — validate AFTER the LLM call
from guardrails.output_guardrails import (
    validate_output,
    check_safety_disclaimers,
    check_prohibited_content,
    add_human_review_flag,
)

# Confidence guardrails — route on self-assessed certainty
from guardrails.confidence_guardrails import (
    extract_confidence,
    gate_on_confidence,
    check_confidence,
)

# LLM-as-judge guardrails — semantic evaluation via second LLM
from guardrails.llm_judge_guardrails import (
    JudgeVerdict,
    evaluate_with_judge,
    default_approve_verdict,
    JUDGE_SYSTEM_PROMPT,
)

__all__ = [
    # Input
    "validate_input",
    "check_medical_scope",
    "detect_pii",
    "detect_prompt_injection",
    # Output
    "validate_output",
    "check_safety_disclaimers",
    "check_prohibited_content",
    "add_human_review_flag",
    # Confidence
    "extract_confidence",
    "gate_on_confidence",
    "check_confidence",
    # LLM judge
    "JudgeVerdict",
    "evaluate_with_judge",
    "default_approve_verdict",
    "JUDGE_SYSTEM_PROMPT",
]
