# Chapter 2 — Core Concepts You Must Understand First

Five concepts underpin every design decision in this module. If any of these is unclear, the code will look like arbitrary choices. If all five are clear, the code reads as a natural consequence of the domain.

---

## Concept 1 — Guardrails vs. Routing

**What it is:**  
A *guardrail* is a check that evaluates whether content meets a policy. A *route* is the control-flow decision that follows. These are different things that happen to be co-located.

In `validate_input()`, the guardrail *detects* a prompt injection. The *routing* — whether to call the agent or the reject node — happens in the LangGraph conditional edge (`route_after_input()` in the pattern script). The guardrail returns a dict; the router reads the dict.

**Why it matters for this module:**  
The guardrail functions are pure Python — no LangGraph dependency, no state management. This lets you unit-test every check in isolation. The routing only exists in the graph scripts (`scripts/guardrails/`). Never conflate the two.

**Concrete example:**

```python
# The guardrail — pure logic, no routing
result = validate_input("What is the weather in New York?")
# result = {"passed": False, "guardrail": "out_of_scope", "reason": "..."}

# The router — reads the guardrail result, lives in the graph
def route_after_input(state):
    if state["validation_result"]["passed"]:
        return "agent"   # → proceed to LLM
    return "reject"      # → proceed to rejection handler
```

If you merge these, you cannot swap the routing logic (e.g., from binary pass/fail to three-way triage) without touching the guardrail code.

---

## Concept 2 — Fail-Open vs. Fail-Closed

**What it is:**  
When a safety mechanism itself fails (error, timeout, unexpected input), it must choose a default outcome:
- **Fail-closed** — default to *block*. Nothing passes unless the check explicitly succeeds.
- **Fail-open** — default to *allow*. Content passes unless the check explicitly fails.

Neither is universally correct. The right choice depends on what other safety layers exist and what the cost of each type of error is.

**Why it matters for this module:**  
The module uses *both* strategies in different places, and understanding why prevents you from accidentally making the system either too restrictive or too permissive.

**Where each strategy appears:**

| Location | Strategy | Reason |
|----------|----------|--------|
| `extract_confidence()` — default `0.5` | Fail-open | A 0.5 confidence triggers escalation only if threshold > 0.5; deterministic checks already ran. |
| `check_medical_scope()` — `in_scope = True` when no off-topic words | Fail-open | Unknown queries are treated as in-scope to avoid blocking legitimate edge cases. |
| `default_approve_verdict()` — verdict `"approve"` | Fail-open | Used only after Layer 1 deterministic checks passed; judge failure should not block a clean response. |
| `GuardrailTripped` exception pattern | Fail-closed | Explicit exception signals the orchestrator to stop; no partial delivery. |

**Concrete example:**

```python
# Fail-open: if confidence is missing, assume 0.5 (moderate — let threshold decide)
confidence = extract_confidence("Assessment: possible COPD. Treatment: bronchodilators.")
# → 0.5 (no "Confidence: X" marker found)

gate = gate_on_confidence(0.5, threshold=0.75)
# → {"action": "escalate", ...}
# The missing marker did NOT auto-approve — the threshold still caught it.
```

⚠️ **Why this matters:** A fail-open default is only safe when another check catches the gap. The 0.5 default is safe because threshold gating is still applied to it. If you removed threshold gating, a 0.5 default would silently deliver uncertain responses.

---

## Concept 3 — Content Checks vs. Certainty Checks

**What it is:**  
Content checks evaluate *what* the content says — are specific patterns present or absent?  
Certainty checks evaluate *how confident* the agent is about what it said.

These are orthogonal dimensions. A response can be:
- High quality content + high confidence → deliver
- High quality content + low confidence → escalate (agent is unsure, even if correct)
- Dangerous content + high confidence → block (agent is wrong and confidently so)
- Dangerous content + low confidence → block and escalate

**Why it matters for this module:**  
`input_guardrails.py` and `output_guardrails.py` are content checks.  
`confidence_guardrails.py` is a certainty check.  
They must run *in sequence*, not instead of each other:

```
Content check first → catch obvious problems cheaply
Certainty check second → catch subtle uncertainty that content checks miss
```

**Concrete example:**

```python
# This response passes content checks (no prohibited patterns, has disclaimer)
response = """
The patient likely has COPD exacerbation. Continue current bronchodilators.
Consult your healthcare provider.
Confidence: 0.42
"""

output_check = validate_output(response)
# → {"passed": True, "needs_review": False, ...}   ← content is fine

confidence_check = check_confidence(response, threshold=0.75)
# → {"action": "escalate", "confidence": 0.42, ...}  ← certainty is not fine
```

A content-only check would deliver this response. Adding certainty gating escalates it to human review.

---

## Concept 4 — Layered Defence (Defence in Depth)

**What it is:**  
No single guardrail catches everything. Layered defence is the principle that you stack multiple independent guardrails, each catching different failure modes. A response must pass all layers to be delivered. Any single layer failing causes escalation or blocking.

This is not redundancy (doing the same check twice). Each layer checks a *different dimension*:

```
Layer 1 — Input content    : Is the input safe to process?    (fast, free)
Layer 2 — Output content   : Is the output safe to deliver?   (fast, free)
Layer 3 — Confidence       : Is the agent certain enough?     (fast, free)
Layer 4 — LLM judge        : Is the reasoning sound?          (slow, costs tokens)
```

**Why it matters for this module:**  
Layer 4 is expensive. You never run it on every request. The correct pattern is:
- Run Layers 1–3 always (near-zero cost)
- Run Layer 4 only on high-stakes requests that passed Layers 1–3

This is why `layered_validation.py` stacks input + output guardrails, and `llm_as_judge.py` is a *separate* pattern script you add on top when needed.

**Concrete example:**

```python
# Layer 1 — block bad input immediately (zero LLM cost)
input_result = validate_input(user_query)
if not input_result["passed"]:
    return f"Blocked: {input_result['reason']}"

# Layer 2 — call LLM only if input passed
response = llm.invoke(user_query)

# Layer 3 — check output content (zero LLM cost)
output_result = validate_output(response)
if output_result["needs_review"]:
    return add_human_review_flag(response, "Content issues detected")

# Layer 4 — only for high-stakes clinical decisions
if is_high_stakes(user_query):
    verdict = evaluate_with_judge(llm, patient_case, user_query, response)
    if verdict.verdict == "reject":
        return SAFE_FALLBACK
```

⚠️ **Why this matters:** If you skip Layer 1 and rely only on Layer 4, you spend LLM tokens evaluating injected prompts and off-topic queries. Layer 1 exists specifically to prevent this cost.

---

## Concept 5 — The Two-LLM Pattern (LLM-as-Judge)

**What it is:**  
The two-LLM pattern uses one LLM to generate a response and a *second, independent LLM* to evaluate that response. The evaluator (judge) receives the original context, the original query, and the first LLM's response, and returns a structured verdict.

This solves a fundamental limitation of regex-based guardrails: they can only detect patterns they were programmed to look for. An LLM judge can reason about *semantic safety* — patterns that cannot be expressed as rules.

**Why it matters for this module:**  
Regex can catch `"stop all medications immediately"`. It cannot catch:
> "The patient's current ACE inhibitor should be discontinued given the potassium level of 4.2."  
> (This looks medically reasonable but misses that BNP=650 makes holding an ACEi dangerous in CHF.)

An LLM judge with the full patient context can reason: "This recommendation is unsafe given the CHF diagnosis." No regex could.

**The pattern in this module:**

```
┌─────────────────────────────────┐
│  LLM Call 1: Clinical Agent     │
│  Input:  patient_case + query   │
│  Output: clinical_assessment    │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│  LLM Call 2: Judge              │
│  Input:  patient_case           │
│           + query               │
│           + clinical_assessment │
│  Output: JudgeVerdict (Pydantic)│
│    .safety: safe/unsafe/borderline
│    .relevance: relevant/partial/irrelevant
│    .completeness: complete/partial/incomplete
│    .verdict: approve/revise/reject
└─────────────────────────────────┘
```

**Concrete example:**

```python
from guardrails import evaluate_with_judge, default_approve_verdict

try:
    verdict = evaluate_with_judge(
        llm=get_llm(),
        patient_case={"age": 72, "sex": "M", "chief_complaint": "SOB", ...},
        user_query="Assess this patient's medication regimen.",
        agent_response=clinical_response
    )
except RuntimeError:
    # Judge LLM failed — fall back to approve (deterministic checks already passed)
    verdict = default_approve_verdict("Judge unavailable — fail open.")

if verdict.verdict == "reject":
    return SAFE_FALLBACK
elif verdict.verdict == "revise":
    # verdict.suggested_fix contains the correction
    apply_fix(verdict.suggested_fix)
```

**One honest limitation:** The judge itself can hallucinate. It is a second LLM, not a ground-truth oracle. It reduces risk; it does not eliminate it. In production, judge verdicts should be logged and periodically audited by a human clinician.
