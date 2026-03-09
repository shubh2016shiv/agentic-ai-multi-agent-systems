# Chapter 4 — Design Decisions Worth Examining

Six decisions in this module look simple but encode meaningful architectural tradeoffs. Each is examined here from the perspective of: what was chosen, why, what it costs, and when the opposite would be correct.

---

## Decision 1 — Check Order in `validate_input()`: Injection Before PII

**What was chosen:**

```python
# Step 3: Prompt injection  ← runs BEFORE
if check_injection:
    injection_result = detect_prompt_injection(text)
    ...

# Step 4: PII check  ← runs AFTER
if check_pii:
    pii_result = detect_pii(text)
    ...
```

**Why this way:**  
Prompt injection is a *security* concern; PII is a *compliance* concern. Security violations are more immediately dangerous — an injected prompt could exfiltrate data, override safety rules, or cause the LLM to produce content that bypasses every downstream guardrail. PII in an input, by contrast, is a data handling issue that must be logged and reported but does not threaten the system's security posture in the same way.

Additionally, attackers sometimes *embed* PII in injection payloads to make them look like legitimate medical queries. By checking injection first, you block the attack before the PII extractor runs — preventing the PII scanner from "laundering" the intent of the request.

**The tradeoff:**  
A query with both PII and an injection will be reported as a `prompt_injection` failure, not a `pii_detected` failure. The caller gets one reason, not both. If your audit system needs to track PII-in-injection events separately, you lose that signal.

**When you'd change it:**  
If your compliance team requires that *every* PII detection event be logged regardless of whether injection was also detected, run the checks non-exclusively (collect all failures) and return the most-severe one as the primary `guardrail` field. See "Decision 6" in this chapter for the aggregation pattern.

⚠️ **Why this matters:** The order of security checks is rarely arbitrary. Changing the order changes which failure reason appears in logs, which can affect incident response procedures.

---

## Decision 2 — Default Confidence of `0.5` When No Marker Is Found

**What was chosen:**

```python
def extract_confidence(text: str, default: float = 0.5) -> float:
    ...
    logger.debug(f"No confidence found in response — using default {default}")
    return default
```

**Why this way:**  
`0.5` is the midpoint of the `[0.0, 1.0]` range — maximum uncertainty. This is intentional: when the LLM doesn't report its confidence, we don't know whether it's certain or uncertain, so we assume the worst middle-ground and let the threshold decide.

With a typical threshold of `0.75`:
- `0.5 < 0.75` → escalates to human review
- The response is not delivered automatically when confidence is unknown

If the default were `0.0`, every missing-confidence response would escalate. If the default were `1.0`, every missing-confidence response would be delivered. `0.5` is the conservative midpoint that defers to the threshold setting.

**The tradeoff:**  
If your threshold is below `0.5` (e.g., `0.30` for a patient education context), a missing-confidence response will silently auto-deliver rather than escalate. The default appears safe but is threshold-dependent.

**When you'd change it:**  
For high-stakes contexts (emergency triage, drug interaction checks), set `default=0.0` to ensure any response without a confidence marker is *always* escalated. Pass this as the `default` argument to `extract_confidence()`.

```python
# High-stakes: missing confidence = always escalate
confidence = extract_confidence(response_text, default=0.0)
```

---

## Decision 3 — Auto-Append Disclaimer Instead of Block

**What was chosen:**

```python
# In validate_output(), Step 4:
if not disclaimer_result["has_disclaimer"]:
    disclaimer = "\n\n---\n⚕️ *This information is for educational purposes only...*"
    modified_output = output_text + disclaimer   # ← FIX, not BLOCK
    issues.append({
        "type": "missing_disclaimer",
        "severity": "LOW",        # ← LOW, not HIGH
        ...
    })
    # needs_review is NOT set to True here
```

**Why this way:**  
A missing disclaimer is a *fixable* problem, not a fundamentally unsafe response. The LLM produced correct, safe clinical content — it just forgot to add the disclaimer. Auto-appending is better than blocking because:
1. It preserves the value of the LLM's clinical reasoning.
2. It applies the fix deterministically — the disclaimer text is always the same, always compliant.
3. It does not require a second LLM call or human intervention.

The `severity: "LOW"` designation means a missing disclaimer does *not* set `needs_review=True`, so the response flows through without escalation.

**The tradeoff:**  
If a disclaimer is appended silently without any audit trail, you lose visibility into how often LLM responses are missing disclaimers. Over time, a high missing-disclaimer rate might indicate a prompt engineering problem that should be fixed upstream. The current implementation logs at `INFO` level but does not emit a metric.

**When you'd change it:**  
In a highly regulated context (FDA-regulated software, ISO 13485 certified product), you may need to treat any missing required element as a `CRITICAL` failure that blocks delivery and triggers an audit event. In that case, change the severity to `"HIGH"` and set `needs_review = True`.

⚠️ **Why this matters:** The choice between "fix and continue" vs. "block and escalate" is a product decision, not just an engineering decision. It encodes your risk tolerance for each type of compliance gap.

---

## Decision 4 — Configurable Routing Labels in `gate_on_confidence()`

**What was chosen:**

```python
def gate_on_confidence(
    confidence: float,
    threshold: float = 0.75,
    label_above: str = "deliver",      # ← configurable
    label_below: str = "escalate",     # ← configurable
) -> dict[str, Any]:
    action = label_above if passed else label_below
    return {"action": action, ...}
```

**Why this way:**  
LangGraph conditional edges route on string values that match node names in the graph. Different graphs have different node names. If `gate_on_confidence()` returned hardcoded strings like `"deliver"` and `"escalate"`, every caller would need to name their nodes exactly those strings, or write a translation layer.

Configurable labels keep the guardrail function topology-agnostic. The graph controls the vocabulary:

```python
# Graph with standard names
result = gate_on_confidence(conf, threshold=0.75)
# action = "deliver" or "escalate"

# Graph with custom names
result = gate_on_confidence(conf, threshold=0.75,
                            label_above="publish",
                            label_below="human_review_queue")
# action = "publish" or "human_review_queue"
```

**The tradeoff:**  
More flexibility means more surface area for misconfiguration. If a caller passes `label_above="dliver"` (typo), the returned action will be `"dliver"`, which won't match any graph node, and LangGraph will raise an error at runtime rather than at configuration time.

**When you'd change it:**  
If your system has a fixed routing vocabulary (e.g., always `"pass"` / `"fail"`), remove the label parameters and hardcode the strings. Fewer parameters = fewer ways to misconfigure. Only use configurable labels when you have multiple graphs with different node naming conventions.

---

## Decision 5 — Fail-Open Default in `default_approve_verdict()`

**What was chosen:**

```python
def default_approve_verdict(reason: str = "Judge unavailable — fail open.") -> JudgeVerdict:
    return JudgeVerdict(
        safety="safe",
        relevance="relevant",
        completeness="complete",
        verdict="approve",         # ← approve, not reject
        ...
    )
```

**Why this way:**  
`default_approve_verdict()` is called only when the judge LLM *fails* (exception in `evaluate_with_judge()`). By the time the judge is invoked, the response has already passed three independent layers:
1. Input content checks (no injection, PII, or out-of-scope)
2. Output content checks (no prohibited patterns, disclaimer present or appended)
3. Confidence threshold (agent reported sufficient certainty)

Failing closed on a judge error would block responses that are already triple-validated. The marginal safety benefit of fail-closed is low; the cost (blocking valid clinical responses during a model outage) is high.

**The tradeoff:**  
If the judge is the *only* safety layer (no deterministic checks), failing open on judge errors delivers unvalidated content. This is unsafe. The fail-open choice is only correct when other layers have already run.

**When you'd change it:**  
Change to fail-closed (`verdict="reject"`) when:
1. The judge is the *only* or *primary* safety mechanism (no deterministic checks).
2. The clinical context has zero tolerance for unvalidated delivery (e.g., surgical planning, ICU dosing).
3. Your SLA prioritizes safety over availability.

```python
# Fail-closed version for high-stakes contexts
def default_reject_verdict(reason: str) -> JudgeVerdict:
    return JudgeVerdict(
        safety="borderline",
        relevance="partially_relevant",
        completeness="partial",
        verdict="reject",
        reasoning=reason,
    )
```

---

## Decision 6 — Structured Output for the Judge (`JudgeVerdict` Pydantic Model)

**What was chosen:**

```python
class JudgeVerdict(BaseModel):
    safety: Literal["safe", "unsafe", "borderline"]
    relevance: Literal["relevant", "partially_relevant", "irrelevant"]
    completeness: Literal["complete", "partial", "incomplete"]
    verdict: Literal["approve", "revise", "reject"]
    reasoning: str
    suggested_fix: str = ""

judge_llm = llm.with_structured_output(JudgeVerdict)
verdict = judge_llm.invoke([SystemMessage(...), HumanMessage(...)])
```

**Why this way:**  
Without structured output, the judge returns free text like:
> "The response appears safe. It is mostly relevant. I would approve it with minor reservations."

This requires:
1. Text parsing to extract the verdict
2. Handling natural language variations ("mostly safe", "generally relevant")
3. Validating that a verdict was actually expressed

With `with_structured_output(JudgeVerdict)`:
1. The LLM framework constructs a JSON-constrained generation
2. Pydantic validates the result
3. The routing code does `if verdict.verdict == "reject"` — no parsing

**The tradeoff:**  
`with_structured_output` adds one overhead: the LLM framework typically generates JSON by appending schema constraints to the prompt, which consumes additional tokens. For large patient cases + verbose reasoning, the evaluation prompt can approach the model's context limit.

**When you'd change it:**  
If you're using a model that doesn't support reliable JSON generation (older open-source models, small quantized models), `with_structured_output` may fail frequently. In that case, fall back to prompt-based extraction: ask the LLM to end its response with `VERDICT: approve/revise/reject` and parse the last line.

⚠️ **Why this matters:** Structured output is the cornerstone of reliable multi-agent orchestration. Without it, every LLM-to-LLM handoff becomes a fragile text parsing problem. The investment in `JudgeVerdict` pays dividends in observability, testability, and routing reliability.
