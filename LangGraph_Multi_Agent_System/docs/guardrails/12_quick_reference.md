# Chapter 7 — Quick Reference Card

---

## 7a — Function / Class Reference Table

### Input Guardrails (`guardrails.input_guardrails`)

| Name | Type | Input | Output | Use When |
|------|------|-------|--------|----------|
| `validate_input` | function | `text: str`, optional flags | `dict` — `{passed, guardrail, reason, details}` | Validate ALL input before any LLM call |
| `detect_pii` | function | `text: str` | `dict` — `{found, types, count}` | Standalone PII scan, audit logging |
| `detect_prompt_injection` | function | `text: str` | `dict` — `{detected, pattern}` | Standalone injection audit |
| `check_medical_scope` | function | `text: str` | `dict` — `{in_scope, medical_keywords_found, off_topic_keywords_found}` | Scope check outside full validation |

### Output Guardrails (`guardrails.output_guardrails`)

| Name | Type | Input | Output | Use When |
|------|------|-------|--------|----------|
| `validate_output` | function | `output_text: str`, `original_query: str`, `confidence: float \| None`, flags | `dict` — `{passed, needs_review, issues, issue_count, modified_output}` | After every LLM call, before delivery |
| `check_prohibited_content` | function | `text: str` | `dict` — `{found, matched_pattern}` | Standalone scan of specific content |
| `check_safety_disclaimers` | function | `text: str` | `dict` — `{has_disclaimer, found_elements, missing_disclaimer}` | Compliance audit of stored responses |
| `add_human_review_flag` | function | `output_text: str`, `reason: str` | `str` (modified text) | Prepend HITL marker before queuing for review |

### Confidence Guardrails (`guardrails.confidence_guardrails`)

| Name | Type | Input | Output | Use When |
|------|------|-------|--------|----------|
| `extract_confidence` | function | `text: str`, `default: float = 0.5` | `float` in `[0.0, 1.0]` | Need just the score, routing handled separately |
| `gate_on_confidence` | function | `confidence: float`, `threshold: float = 0.75`, labels | `dict` — `{action, confidence, threshold, passed}` | Have the score already, need routing decision |
| `check_confidence` | function | `text: str`, `threshold: float = 0.75` | same dict as `gate_on_confidence` + `raw_text_length` | Have raw LLM text, need score + routing in one call |

### LLM Judge Guardrails (`guardrails.llm_judge_guardrails`)

| Name | Type | Input | Output | Use When |
|------|------|-------|--------|----------|
| `JudgeVerdict` | Pydantic model | — | `.safety`, `.relevance`, `.completeness`, `.verdict`, `.reasoning`, `.suggested_fix` | Used as structured output schema for the judge |
| `evaluate_with_judge` | function | `llm`, `patient_case: dict`, `user_query: str`, `agent_response: str` | `JudgeVerdict` | High-stakes requests after deterministic checks pass |
| `default_approve_verdict` | function | `reason: str` | `JudgeVerdict` (verdict="approve") | In `except RuntimeError` block after `evaluate_with_judge` fails |
| `JUDGE_SYSTEM_PROMPT` | str constant | — | — | Override when customising judge rubric |

---

## 7b — Copy-Paste Usage Cheat Sheet

### Minimal: Input Validation Only

```python
from guardrails import validate_input

result = validate_input(user_query)
if not result["passed"]:
    return f"Request blocked: {result['reason']}"
# else: proceed to LLM
```

### Minimal: Output Validation Only

```python
from guardrails import validate_output

result = validate_output(agent_response, original_query=user_query)
# Always use modified_output — may have disclaimer appended
if result["needs_review"]:
    queue_for_human_review(result["modified_output"])
else:
    deliver(result["modified_output"])
```

### Minimal: Confidence Gating Only

```python
from guardrails import check_confidence

result = check_confidence(llm_response_text, threshold=0.75)
if result["action"] == "escalate":
    route_to_human_review(llm_response_text)
else:
    deliver(llm_response_text)
```

### Minimal: LLM Judge

```python
from guardrails import evaluate_with_judge, default_approve_verdict

try:
    verdict = evaluate_with_judge(llm, patient_case, user_query, agent_response)
except RuntimeError as e:
    verdict = default_approve_verdict(reason=str(e))

if verdict.verdict == "reject":
    return SAFE_FALLBACK_RESPONSE
elif verdict.verdict == "revise":
    print(f"Suggested fix: {verdict.suggested_fix}")
```

### Full Pipeline: All Four Layers

```python
from guardrails import (
    validate_input, validate_output, check_confidence,
    evaluate_with_judge, default_approve_verdict, add_human_review_flag,
)
from core.exceptions import GuardrailTripped

# Layer 1: Input
result = validate_input(user_query)
if not result["passed"]:
    raise GuardrailTripped(result["reason"], details={"guardrail": result["guardrail"]})

# Layer 2: LLM
response = llm.invoke(user_query)

# Layer 3a: Output content
out = validate_output(response, original_query=user_query)
if out["needs_review"]:
    return add_human_review_flag(out["modified_output"], reason=str(out["issues"]))

# Layer 3b: Confidence
conf = check_confidence(out["modified_output"], threshold=0.75)
if conf["action"] == "escalate":
    return add_human_review_flag(out["modified_output"], reason=f"Low confidence: {conf['confidence']:.0%}")

# Layer 4: LLM judge (high-stakes only)
try:
    verdict = evaluate_with_judge(llm, patient_case, user_query, out["modified_output"])
except RuntimeError as e:
    verdict = default_approve_verdict(reason=str(e))

if verdict.verdict == "reject":
    return "Clinical review required. A physician will respond shortly."

return out["modified_output"]  # Deliver
```

### Standalone PII Scan

```python
from guardrails import detect_pii

result = detect_pii("Patient DOB: 03/15/1968, SSN: 123-45-6789")
# result = {"found": True, "types": ["date_of_birth", "ssn"], "count": 2}
```

### Extracting Confidence Score Only

```python
from guardrails import extract_confidence

text = "Assessment: COPD exacerbation likely.\nConfidence: 0.87"
score = extract_confidence(text)     # → 0.87

text2 = "Assessment: unclear presentation."
score2 = extract_confidence(text2)   # → 0.5 (default, no marker found)
score3 = extract_confidence(text2, default=0.0)  # → 0.0 (for high-stakes contexts)
```

---

## 7c — Full Decision Flowchart

The complete module logic from entry to exit — every branch, every output path.

```
USER QUERY
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  validate_input(text)                                       │
│                                                             │
│  len(text) > 5000? ──YES──► BLOCK "input_length"           │
│       │                                                     │
│      NO                                                     │
│       │                                                     │
│  text.strip() == ""? ──YES──► BLOCK "empty_input"          │
│       │                                                     │
│      NO                                                     │
│       │                                                     │
│  detect_prompt_injection(text)?                             │
│    detected=True ──────────► BLOCK "prompt_injection"      │
│       │                                                     │
│   detected=False                                            │
│       │                                                     │
│  detect_pii(text)?                                          │
│    found=True ─────────────► BLOCK "pii_detected"          │
│       │                                                     │
│    found=False                                              │
│       │                                                     │
│  check_medical_scope(text)?                                 │
│    in_scope=False ─────────► BLOCK "out_of_scope"          │
│       │                                                     │
│    in_scope=True                                            │
│       │                                                     │
│  passed=True ──────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  LLM AGENT CALL                                             │
│  (core reasoning — not part of this module)                 │
│  Output: response_text (includes "Confidence: X.XX")        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  validate_output(response_text)                             │
│                                                             │
│  len(response) < 20? ──YES──► FLAG HIGH + needs_review=True │
│       │                                                     │
│      NO                                                     │
│       │                                                     │
│  check_prohibited_content()?                                │
│    found=True ─────────────► FLAG CRITICAL + needs_review   │
│       │                                                     │
│    found=False                                              │
│       │                                                     │
│  confidence < min_confidence (0.3)?                         │
│    YES ────────────────────► FLAG HIGH + needs_review       │
│       │                                                     │
│    NO (or confidence not provided)                          │
│       │                                                     │
│  check_safety_disclaimers()?                                │
│    has_disclaimer=False ──► APPEND disclaimer to output     │
│                              FLAG LOW (does NOT escalate)   │
│       │                                                     │
│  needs_review=True? ───────► ESCALATE: add_human_review_flag│
│       │                       → Human review queue          │
│    needs_review=False         │                             │
│       │                      STOP — do not deliver yet      │
│  passed=True                                                │
│  → use modified_output (may have disclaimer)                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  check_confidence(modified_output, threshold=0.75)          │
│                                                             │
│  extract_confidence():                                      │
│    "Confidence: X.XX" found? → parse to float               │
│    "Confidence: XX%" found?  → parse and normalize          │
│    not found?                → use default (0.5)            │
│       │                                                     │
│  gate_on_confidence():                                      │
│    confidence >= threshold? ──YES──► action="deliver"       │
│                              │                              │
│                             NO                              │
│                              │                              │
│                        action="escalate"                    │
│                              │                              │
│                    add_human_review_flag()                  │
│                    → Human review queue                     │
│                    STOP                                     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  evaluate_with_judge()  [OPTIONAL — high-stakes only]       │
│                                                             │
│  Judge LLM call succeeds?                                   │
│    NO (RuntimeError) ──► default_approve_verdict()          │
│                          verdict.verdict = "approve"        │
│    YES → JudgeVerdict                                       │
│           │                                                 │
│      verdict = ?                                            │
│       "reject" ────────► SAFE FALLBACK RESPONSE            │
│       "revise" ────────► log suggested_fix; optionally      │
│                           re-prompt agent or note in output │
│       "approve" ───────► continue to delivery               │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  DELIVER modified_output to user                            │
│  All four layers passed.                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary: Layer-by-Layer Cheat Sheet

| Layer | Module | Trigger to block/escalate | Cost | Optional? |
|-------|--------|--------------------------|------|-----------|
| Input content | `input_guardrails` | Length, empty, injection, PII, out-of-scope | ~0ms | No — always run |
| Output content | `output_guardrails` | Prohibited patterns, low confidence, missing disclaimer | ~0ms | No — always run |
| Confidence gate | `confidence_guardrails` | Score below threshold | ~0ms | Yes — skip for very low-stakes contexts |
| LLM judge | `llm_judge_guardrails` | Unsafe, irrelevant, or incomplete verdict | 1–5s + tokens | Yes — only for high-stakes decisions |
