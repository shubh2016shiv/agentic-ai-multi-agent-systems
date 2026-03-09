# Chapter 6 — Putting It All Together

---

## 6a — The Recommended Usage Pattern (The Golden Path)

The following shows the complete, annotated end-to-end flow for using this module in a production LangGraph pipeline. Every line has a comment explaining *why* it is written that way.

```python
import logging
from guardrails import (
    validate_input,
    validate_output,
    check_confidence,
    evaluate_with_judge,
    default_approve_verdict,
    add_human_review_flag,
)
from core.exceptions import GuardrailTripped

logger = logging.getLogger(__name__)

# ── Constants: define thresholds at the top, not buried in code ──────────
# Adjust per deployment context (emergency vs. education vs. research)
CONFIDENCE_THRESHOLD = 0.75
IS_HIGH_STAKES = True   # Set based on query category or route


def process_clinical_query(
    user_query: str,
    patient_case: dict,
    llm,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    run_judge: bool = IS_HIGH_STAKES,
) -> str:
    """
    Full guardrailed pipeline for a clinical query.
    Returns the final text to deliver to the user.
    Raises GuardrailTripped if a critical safety check fails.
    """

    # ── Layer 1: Input validation ──────────────────────────────────────────
    # Why first: blocks bad inputs BEFORE any LLM token is spent.
    # Why check_pii=True: patient queries from external users may contain PII.
    # Why check_scope=True: the CDSS should only answer medical questions.
    input_result = validate_input(
        user_query,
        check_pii=True,
        check_injection=True,
        check_scope=True,
        max_length=5000,
    )

    if not input_result["passed"]:
        # Raise a typed exception so the caller (graph node or API handler)
        # can return a user-friendly message without exposing internals.
        raise GuardrailTripped(
            f"Input guardrail tripped: {input_result['reason']}",
            details={"guardrail": input_result["guardrail"]},
        )

    # ── Layer 2: LLM call ──────────────────────────────────────────────────
    # Only reached if Layer 1 passed — no wasted tokens on blocked inputs.
    # The LLM's system prompt must include "Confidence: X.XX" instruction
    # for Layer 3 to work correctly.
    raw_response = llm.invoke(user_query)

    # ── Layer 3a: Output content validation ───────────────────────────────
    # Why pass confidence=None here: validate_output does a content check;
    # confidence gating is a separate step (Layer 3b) for clean separation
    # of concerns.
    output_result = validate_output(
        raw_response,
        original_query=user_query,
        check_disclaimers=True,
        check_prohibited=True,
    )

    if output_result["needs_review"]:
        # HIGH or CRITICAL issues: do not deliver, flag for human review.
        flagged = add_human_review_flag(
            output_result["modified_output"],
            reason="; ".join(i["detail"] for i in output_result["issues"]),
        )
        # In a real system: enqueue flagged to a human review queue.
        # Return a safe holding message to the user.
        logger.warning(f"Output flagged for review: {output_result['issues']}")
        return (
            "Your query is being reviewed by a clinical team member. "
            "You will receive a response within 2 hours."
        )

    # At this point: use modified_output (may have disclaimer appended).
    # NEVER use raw_response after this point — always use modified_output.
    validated_response = output_result["modified_output"]

    # ── Layer 3b: Confidence gating ────────────────────────────────────────
    # Why separate from output content: confidence measures certainty,
    # not content safety. They are orthogonal dimensions.
    conf_result = check_confidence(validated_response, threshold=confidence_threshold)

    if conf_result["action"] == "escalate":
        logger.info(
            f"Low confidence ({conf_result['confidence']:.2f}) → escalating"
        )
        flagged = add_human_review_flag(
            validated_response,
            reason=f"Agent confidence {conf_result['confidence']:.0%} below threshold {confidence_threshold:.0%}",
        )
        return (
            "Your query requires physician review due to clinical complexity. "
            "A team member will respond shortly."
        )

    # ── Layer 4: LLM judge (optional, high-stakes only) ───────────────────
    # Why optional: costs tokens, adds latency. Only worth it for high-stakes
    # decisions that passed all deterministic checks.
    if run_judge:
        try:
            verdict = evaluate_with_judge(
                llm=llm,
                patient_case=patient_case,
                user_query=user_query,
                agent_response=validated_response,
            )
        except RuntimeError as e:
            # Judge failed (API error, timeout, malformed output).
            # Fail open: deterministic checks already passed — do not block.
            logger.error(f"Judge LLM failed: {e}")
            verdict = default_approve_verdict(reason=str(e))

        if verdict.verdict == "reject":
            logger.warning(f"Judge REJECTED: {verdict.reasoning}")
            return (
                "I'm unable to provide a reliable assessment for this case. "
                "Please consult directly with a physician."
            )
        elif verdict.verdict == "revise" and verdict.suggested_fix:
            # The judge identified fixable issues but not a full rejection.
            # Log the suggestion; in production you might re-prompt the agent.
            logger.info(f"Judge suggested revision: {verdict.suggested_fix}")
            # For now, deliver with a note (production: re-invoke agent with fix)
            validated_response += f"\n\n*Note: This response may benefit from: {verdict.suggested_fix}*"

    # ── Delivery ───────────────────────────────────────────────────────────
    # All layers passed. Deliver the validated (and possibly modified) response.
    return validated_response
```

---

## 6b — When to Skip or Bypass This Module

Bypassing guardrails is sometimes the correct engineering choice. The following decision guide identifies when it is safe.

```
Is the content GENERATED by an untrusted LLM?
  YES → Do NOT skip output guardrails (validate_output is mandatory)
  NO  → May skip if content is static/hardcoded

Is the INPUT from an external or untrusted user?
  YES → Do NOT skip input guardrails (validate_input is mandatory)
  NO, it's internal agent-to-agent → May skip PII/scope checks
                                      (check_injection still recommended)

Is the deployment context HIGH STAKES (clinical decisions, patient-facing)?
  YES → Run all four layers; judge is mandatory
  NO  (internal tooling, dev/test) → May skip confidence + judge layers

Is there ANOTHER safety layer already enforced upstream?
  YES (e.g., the API gateway validates all inputs) → May skip redundant checks
      but document the assumption explicitly in code comments
  NO → Do not skip

Is this a UNIT TEST or LOCAL DEVELOPMENT environment?
  YES → Acceptable to mock or disable guardrails, but test the real guardrail
        logic in integration tests before production deployment
```

**The one case where ALL guardrails can be legitimately skipped:** Internal agent-to-agent messages where:
1. Both agents are part of the same controlled MAS
2. The sending agent's output has already been validated by its own output guardrails
3. The message is a structured data type (not free text), such as a Pydantic model

In this case, the receiving agent's input guardrails are redundant and add latency without safety benefit.

---

## 6c — How It Connects to the Rest of the System

```
ADJACENT COMPONENTS AND WHAT THIS MODULE EXCHANGES WITH EACH
─────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────┐
│  Upstream: User Interface / API Gateway                         │
│                                                                 │
│  SENDS TO guardrails:                                           │
│    - raw user_input (str)                                       │
│    - patient_case (dict) for judge                              │
│                                                                 │
│  RECEIVES FROM guardrails:                                      │
│    - GuardrailTripped exception (on block)                      │
│    - Human-review flag string (on escalation)                   │
│    - Safe delivery message (on pass)                            │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  This Module: guardrails/                                       │
│                                                                 │
│  READS from shared state:                                       │
│    - user_input (str)                                           │
│    - agent_response (str)                                       │
│    - confidence (float) embedded in response text               │
│                                                                 │
│  WRITES to shared state:                                        │
│    - validation_result (dict)  ← routing signal for graph edges │
│    - modified_output (str)     ← disclaimer-appended response   │
│    - needs_review (bool)       ← escalation flag                │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Downstream 1: LLM Agent Layer                                  │
│                                                                 │
│  RECEIVES FROM guardrails:                                      │
│    - Routing signal ("agent" vs "reject") from input validation │
│                                                                 │
│  SENDS TO guardrails (after agent runs):                        │
│    - Raw LLM response text (str), typically contains            │
│      "Confidence: X.XX" as last line                            │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Downstream 2: HITL / Human Review Queue                        │
│                                                                 │
│  RECEIVES FROM guardrails:                                      │
│    - add_human_review_flag() output (str) — flagged response    │
│    - Reason string identifying which guardrail escalated        │
│                                                                 │
│  This component is external to the guardrails module.           │
│  The module only prepares the flag; routing to the queue        │
│  is handled by the graph conditional edge.                      │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Downstream 3: Observability / Langfuse                         │
│                                                                 │
│  The module uses Python logging (WARNING, INFO, DEBUG).         │
│  It does NOT call Langfuse directly.                            │
│  The calling system is responsible for wrapping guardrail       │
│  node execution in Langfuse spans.                              │
└─────────────────────────────────────────────────────────────────┘
```

**Shared state protocol:** In LangGraph, the guardrail functions read from and write to the `TypedDict` state. The canonical state fields the guardrail layer expects:

```python
class ClinicalAgentState(TypedDict):
    user_input: str                  # read by input validation
    validation_result: dict          # written by input validation, read by router
    agent_response: str              # read by output validation
    output_validation_result: dict   # written by output validation, read by router
    final_response: str              # written at delivery, read by caller
```

---

## 6d — Scaling & Performance Considerations

### Bottlenecks Under Load

**Deterministic checks (Layers 1–3):** Near-zero CPU cost. `validate_input()` and `validate_output()` are pure Python regex — expect <1ms per call at any load. `extract_confidence()` and `gate_on_confidence()` are similarly trivial. These layers scale horizontally with no special consideration.

**LLM judge (Layer 4):** The bottleneck. A single `evaluate_with_judge()` call adds one full LLM API round trip — typically 1–5 seconds of latency and 500–2000 additional tokens. Under heavy load:
- LLM API rate limits will be hit first
- The judge doubles the token cost of every high-stakes request
- Latency becomes the user experience problem

### Caching Opportunities

**Do not cache guardrail decisions.** Each request has unique content; a cached "pass" for a previous request is not valid for a new one.

**Do cache frequently-used LLM objects.** The `llm.with_structured_output(JudgeVerdict)` call inside `evaluate_with_judge()` constructs a new wrapped LLM on every invocation. In a high-throughput system, cache this at the module level:

```python
# Module-level cache (safe: with_structured_output is stateless)
_judge_llm_cache: dict = {}

def evaluate_with_judge(llm, ...):
    cache_key = id(llm)
    if cache_key not in _judge_llm_cache:
        _judge_llm_cache[cache_key] = llm.with_structured_output(JudgeVerdict)
    judge_llm = _judge_llm_cache[cache_key]
    ...
```

### Async Considerations

The current implementation is synchronous. In a high-throughput API server (FastAPI, async LangGraph), all four `validate_*` calls should be `async def`. LangChain's `ainvoke()` supports async natively. The key change:

```python
# Synchronous (current)
verdict = judge_llm.invoke([SystemMessage(...), HumanMessage(...)])

# Async (production)
verdict = await judge_llm.ainvoke([SystemMessage(...), HumanMessage(...)])
```

For the judge specifically, consider a **fire-and-forget** async pattern:
1. Deliver the deterministically-validated response immediately (after Layers 1–3)
2. Fire the judge asynchronously in the background
3. If the judge returns `reject`, send a follow-up correction message to the user

This removes judge latency from the critical path entirely.

### Parallelisation

For responses that need multiple independent checks, you can run them in parallel using `asyncio.gather()`:

```python
import asyncio

async def validate_all(text, response):
    input_result, output_result = await asyncio.gather(
        asyncio.to_thread(validate_input, text),      # I/O-free, use thread for compatibility
        asyncio.to_thread(validate_output, response),
    )
    return input_result, output_result
```

However, note that `validate_input()` and `validate_output()` must run *sequentially* by nature (output validation only runs if input validation passes). The parallelisation opportunity applies to independent checks *within* a layer (e.g., running `detect_pii()` and `detect_prompt_injection()` in parallel in a future refactor of `validate_input()`).

### What Should Be Made Configurable in Production

| Currently hardcoded | Where | What to change it to |
|---------------------|-------|---------------------|
| `max_length=5000` | `validate_input()` | Per-model context window config |
| `min_confidence=0.3` | `validate_output()` | Per-deployment risk tier |
| `threshold=0.75` | `gate_on_confidence()` | Per-route configuration |
| `PROHIBITED_CONTENT` list | `output_guardrails.py` | Database-backed, hot-reloadable list |
| `MEDICAL_KEYWORDS` list | `input_guardrails.py` | Domain-specific keyword registry |
| `JUDGE_SYSTEM_PROMPT` | `llm_judge_guardrails.py` | Versioned prompt in a prompt management system |
