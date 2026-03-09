# Chapter 3 — Deep Dive: Every Major Component

This chapter walks through every significant function and class in the guardrails package. For each one: what it does, when it runs, how it works step by step, what goes in and comes out, why the implementation strategy was chosen, and where it falls short.

---

## Part A — Input Guardrails (`input_guardrails.py`)

### `validate_input()`

**Purpose:** Orchestrator that runs all input checks in a deterministic, fail-fast order and returns a single result dict.

**Position:** Runs as the *first* node in any guarded LangGraph graph, before any LLM call is made.

**Step-by-step walkthrough:**

```
validate_input(text, check_pii, check_injection, check_scope, max_length)
  │
  ├─ Step 1: Length check
  │     Is len(text) > max_length (default 5000)?
  │     YES → return {"passed": False, "guardrail": "input_length", ...}
  │     NO  → continue
  │
  ├─ Step 2: Empty check
  │     Is text.strip() empty?
  │     YES → return {"passed": False, "guardrail": "empty_input", ...}
  │     NO  → continue
  │
  ├─ Step 3: Prompt injection check  (if check_injection=True)
  │     Call detect_prompt_injection(text)
  │     DETECTED → return {"passed": False, "guardrail": "prompt_injection", ...}
  │     NOT DETECTED → continue
  │
  ├─ Step 4: PII check  (if check_pii=True)
  │     Call detect_pii(text)
  │     FOUND → return {"passed": False, "guardrail": "pii_detected", ...}
  │     NOT FOUND → continue
  │
  └─ Step 5: Medical scope check  (if check_scope=True)
        Call check_medical_scope(text)
        OUT OF SCOPE → return {"passed": False, "guardrail": "out_of_scope", ...}
        IN SCOPE → return {"passed": True, ...}
```

**Inputs & Outputs:**

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `text` | `str` | required | Raw input text |
| `check_pii` | `bool` | `True` | Disable for internal agent-to-agent messages |
| `check_injection` | `bool` | `True` | Disable only if text origin is fully trusted |
| `check_scope` | `bool` | `True` | Disable for general-purpose (non-medical) deployments |
| `max_length` | `int` | `5000` | Tune based on your context window size |

Returns: `dict[str, Any]` with keys `passed`, `guardrail`, `reason`, `details`.

**Under the hood:** The fail-fast return pattern (early return on first failure) is a deliberate performance choice. Once one guardrail fails, there is no value in running the remaining checks — the request will be rejected regardless. This also means the result always identifies *exactly one* tripped guardrail, making error messages clear.

**One real limitation:** The checks run sequentially and independently. A query can contain *both* PII and a prompt injection attempt, but `validate_input` will only report the injection (because injection is checked first, and it short-circuits). If your audit system needs to know all violations in a single request, you must run the sub-checks individually and aggregate.

**Production upgrade path:** Add an audit log event for *every* check run (not just failures), with a correlation ID linking to the session. This allows security teams to see patterns of failed attempts over time (e.g., repeated injection probes from the same user).

---

### `detect_pii()`

**Purpose:** Scan text for personally identifiable information using regex patterns.

**Position:** Called by `validate_input()` as Step 4. Can also be called standalone.

**Step-by-step walkthrough:**

```
detect_pii(text)
  │
  ├─ For each PII type in PII_PATTERNS:
  │     {"ssn": r"\b\d{3}-\d{2}-\d{4}\b",
  │      "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
  │      "email": r"\b[a-zA-Z0-9._%+-]+@...",
  │      "credit_card": r"\b\d{4}[-\s]?\d{4}...",
  │      "date_of_birth": r"\b(DOB|date of birth|born on)..."}
  │
  │     re.findall(pattern, text, re.IGNORECASE)
  │     If matches found → append type to found_types, add count
  │
  └─ Return {"found": bool, "types": list, "count": int}
```

**Inputs & Outputs:**
- In: `text: str`
- Out: `{"found": bool, "types": list[str], "count": int}`

**Under the hood:** Regex pattern matching is used because PII patterns for SSN, phone, and credit card numbers are highly regular. Regex is deterministic, free, and millisecond-fast. The tradeoff: it cannot detect PII that doesn't match the exact pattern (e.g., "my social is 123 45 6789" without hyphens in the right places).

**One real limitation:** The patterns are simplified. A real SSN is `\b\d{3}-\d{2}-\d{4}\b`, but phone numbers have dozens of valid formats internationally. The current patterns will miss `+44 7911 123456` (UK mobile), `(800) 555-1234`, and many others. They will also generate false positives: a lab value like `800-123-4567 mg/dL` could match the phone pattern.

**Production upgrade path:** Replace regex with a named entity recognition (NER) model such as spaCy with a medical NER pipeline, or Microsoft's Presidio library (purpose-built for PII detection). These handle format variation and context better than regex.

---

### `detect_prompt_injection()`

**Purpose:** Detect attempts to override the system prompt or jailbreak the LLM.

**Position:** Called by `validate_input()` as Step 3.

**Step-by-step walkthrough:**

```
detect_prompt_injection(text)
  │
  ├─ text_lower = text.lower()
  │
  ├─ For each pattern in INJECTION_PATTERNS:
  │     ["ignore (all) previous instructions",
  │      "you are now a/an ...",
  │      "system prompt",
  │      "forget everything/all/your",
  │      "jailbreak",
  │      "pretend to be",
  │      "act as if you",
  │      "override safety/guardrail/instruction"]
  │
  │     re.search(pattern, text_lower)
  │     MATCH → return {"detected": True, "pattern": pattern}
  │
  └─ No match → return {"detected": False, "pattern": None}
```

**Inputs & Outputs:**
- In: `text: str`
- Out: `{"detected": bool, "pattern": str | None}`

**Under the hood:** The list covers the most common jailbreak templates seen in the wild as of early 2024. Returning the matched pattern (not just `True`) is deliberate — it provides actionable information for audit logs and security alerts.

**One real limitation:** Adversarial users can encode injection attempts in ways that bypass the patterns: leetspeak (`1gnore pr3v1ous`), Unicode look-alikes, multi-turn injection (spread across multiple messages), or indirect injection via retrieved documents. This check catches naive attempts only.

**Production upgrade path:** Use a dedicated injection classifier (e.g., a fine-tuned BERT model or a commercial guardrails service like Guardrails AI or LlamaGuard). For multi-turn injection, maintain a session-level injection score that accumulates across messages.

---

### `check_medical_scope()`

**Purpose:** Determine whether the query is medically relevant before spending LLM tokens.

**Position:** Called by `validate_input()` as Step 5.

**Step-by-step walkthrough:**

```
check_medical_scope(text)
  │
  ├─ text_lower = text.lower()
  │
  ├─ medical_found = keywords from MEDICAL_KEYWORDS present in text_lower
  │     ["patient", "symptom", "diagnosis", "medication", "drug", ...]
  │
  ├─ off_topic_found = keywords from OFF_TOPIC_KEYWORDS present in text_lower
  │     ["recipe", "cooking", "sports score", "stock price", ...]
  │
  ├─ in_scope = (len(medical_found) > 0) OR (len(off_topic_found) == 0)
  │     Logic: in scope if ANY medical word present, regardless of off-topic words
  │             OR if no off-topic words and no medical words (benefit of the doubt)
  │             NOT in scope only if off-topic words present AND zero medical words
  │
  └─ Return {"in_scope": bool, "medical_keywords_found": list, "off_topic_keywords_found": list}
```

**Inputs & Outputs:**
- In: `text: str`
- Out: `{"in_scope": bool, "medical_keywords_found": list, "off_topic_keywords_found": list}`

**Under the hood:** The logic deliberately errs on the side of *inclusion*: "when in doubt, let it through to the LLM." This is a fail-open scope check. Blocking a legitimate medical query is worse (from a patient safety perspective) than processing an off-topic one.

**One real limitation:** Keyword matching is trivially fooled by queries that combine medical and non-medical content: "What recipe uses the same herbs as the medication amoxicillin?" — this would pass scope checking despite being off-topic. More fundamentally, it cannot detect *medical misinformation* queries ("What's the best way to overdose on aspirin?") that contain valid medical keywords.

**Production upgrade path:** Replace with a text classification model fine-tuned on medical vs. non-medical query datasets. A lightweight BERT classifier can achieve >95% accuracy on this task at ~5ms inference time.

---

## Part B — Output Guardrails (`output_guardrails.py`)

### `validate_output()`

**Purpose:** Orchestrator that runs all output checks and returns a three-state result: pass, needs-fix, or block.

**Position:** Runs as the node *immediately after* the LLM agent node, before any delivery or further routing.

**Step-by-step walkthrough:**

```
validate_output(output_text, original_query, confidence, check_disclaimers, check_prohibited, min_confidence)
  │
  ├─ issues = []
  │
  ├─ Step 1: Empty/short response check
  │     len(output_text.strip()) < 20?
  │     YES → append issue {severity: "HIGH"}, set needs_review = True
  │
  ├─ Step 2: Prohibited content check  (if check_prohibited=True)
  │     Call check_prohibited_content(output_text)
  │     FOUND → append issue {severity: "CRITICAL"}, set needs_review = True
  │
  ├─ Step 3: Low confidence check  (if confidence provided)
  │     confidence < min_confidence (default 0.3)?
  │     YES → append issue {severity: "HIGH"}, set needs_review = True
  │
  ├─ Step 4: Safety disclaimer check  (if check_disclaimers=True)
  │     Call check_safety_disclaimers(output_text)
  │     NO DISCLAIMER → AUTO-APPEND disclaimer to modified_output
  │                      append issue {severity: "LOW"}
  │                      (does NOT set needs_review)
  │
  └─ passed = no CRITICAL or HIGH severity issues found
     Return {
       "passed": bool,
       "needs_review": bool,
       "issues": list,
       "issue_count": int,
       "modified_output": str   ← may have disclaimer appended
     }
```

**Three-state semantics:**

| Condition | `passed` | `needs_review` | Action |
|-----------|----------|---------------|--------|
| No issues | `True` | `False` | Deliver `output_text` |
| Missing disclaimer only | `True` | `False` | Deliver `modified_output` (with appended disclaimer) |
| HIGH severity issue | `False` | `True` | Escalate; do not deliver |
| CRITICAL severity issue | `False` | `True` | Block + escalate; use safe fallback |

**Inputs & Outputs:**
- `output_text: str` — LLM's raw response
- `original_query: str` — The original user query (reserved for future relevance check)
- `confidence: float | None` — If provided, checked against `min_confidence`
- Returns: `dict` with `passed`, `needs_review`, `issues`, `issue_count`, `modified_output`

**Under the hood:** The `modified_output` field is always populated (even when no changes are made) so downstream code can always use `result["modified_output"]` without checking whether modifications occurred. This simplifies graph node code to a single reference.

**One real limitation:** The `original_query` parameter is accepted but not currently used in relevance checking. A common hallucination pattern is a response that passes all content checks but is completely unrelated to the query (e.g., the LLM answered a different question). This is not currently caught.

**Production upgrade path:** Implement relevance scoring using cosine similarity between the query and response embeddings. Responses with similarity below a threshold (e.g., 0.6) would be flagged for review.

---

### `check_prohibited_content()`

**Purpose:** Detect dangerous medical recommendations that should never appear in LLM output.

**Position:** Called by `validate_output()` as Step 2.

**Step-by-step walkthrough:**

```
check_prohibited_content(text)
  │
  ├─ text_lower = text.lower()
  │
  ├─ For each pattern in PROHIBITED_CONTENT:
  │     ["stop all medications? (immediately|at once|right away)",
  │      "no need (to see|for) a? doctor",
  │      "guaranteed (cure|to cure)",
  │      "100% (effective|safe|certain)",
  │      "this will definitely",
  │      "you don't need medical attention",
  │      "ignore your doctor",
  │      "instead of seeing a doctor"]
  │
  │     re.search(pattern, text_lower)
  │     MATCH → return {"found": True, "matched_pattern": pattern}
  │
  └─ No match → return {"found": False, "matched_pattern": None}
```

**Under the hood:** Returns the matched pattern, not just `True/False`. This is essential for audit trails — a security team needs to know *which* dangerous pattern was matched, not just that something was blocked.

**One real limitation:** Regex only catches the exact phrasing variants listed. A hallucinating LLM might produce `"It is entirely safe to cease your medications"` — semantically identical but not matched by any of the current patterns. Paraphrase and synonym variants are a known weakness of this approach.

---

### `check_safety_disclaimers()`

**Purpose:** Verify that the LLM's response includes appropriate medical disclaimers.

**Position:** Called by `validate_output()` as Step 4.

**Step-by-step walkthrough:**

```
check_safety_disclaimers(text)
  │
  ├─ text_lower = text.lower()
  │
  ├─ found_elements = all elements from REQUIRED_ELEMENTS present in text_lower
  │     ["This is for informational purposes",
  │      "Consult your healthcare provider",
  │      "Not a substitute for professional medical advice",
  │      "professional medical",
  │      "consult",
  │      "healthcare provider",
  │      "doctor"]
  │
  └─ has_disclaimer = len(found_elements) > 0
     Return {"has_disclaimer": bool, "found_elements": list, "missing_disclaimer": bool}
```

**One real limitation:** The check is very permissive — any single keyword from the list (including just "doctor") counts as a valid disclaimer. The response `"Ask your doctor about this."` passes. This may be appropriate for informal responses but is too weak for regulatory compliance contexts where specific disclaimer language is mandated.

---

### `add_human_review_flag()`

**Purpose:** Prepend a visible HITL (Human-In-The-Loop) review marker to an output that needs physician review.

**Position:** Called by graph node code when `validate_output()` returns `needs_review=True`.

**Inputs & Outputs:**
- In: `output_text: str`, `reason: str`
- Out: Modified string with review flag prepended

This function is intentionally simple — it formats a string. The HITL routing decision is made in the graph, not here. This function only produces the marked-up text.

---

## Part C — Confidence Guardrails (`confidence_guardrails.py`)

### `extract_confidence()`

**Purpose:** Parse a self-reported confidence score from the raw text of an LLM response.

**Position:** Runs after the LLM agent call, typically before `gate_on_confidence()`.

**Step-by-step walkthrough:**

```
extract_confidence(text, default=0.5)
  │
  ├─ Try pattern: r"[Cc]onfidence[:\s]+(\d+\.?\d*)%"
  │     Matches: "Confidence: 85%"  → 85 → normalise to 0.85
  │
  ├─ Try pattern: r"[Cc]onfidence[:\s]+(\d+\.?\d*)"
  │     Matches: "Confidence: 0.85" → 0.85
  │             "confidence: .7"    → 0.7
  │
  ├─ Normalise: if value > 1.0, divide by 100
  │     "Confidence: 87" → 87 > 1.0 → 87/100 = 0.87
  │
  ├─ Clamp: min(max(value, 0.0), 1.0)
  │     Prevents out-of-range scores from breaking downstream logic
  │
  └─ No match → return default (0.5)
```

**Under the hood:** Two patterns handle the two most natural ways an LLM reports confidence (percentage vs. decimal). The normalization step catches cases where the LLM reports a percentage without the `%` symbol (`"Confidence: 87"`). The clamping step prevents corrupt values like `"Confidence: 1.5"` from propagating.

**One real limitation:** This parser is fragile against rephrasing. `"I'm about 85% sure"`, `"My certainty is moderate"`, `"Confidence level: high"` all go undetected and return the default 0.5. The LLM must be explicitly prompted to use the exact format `"Confidence: X"` for this to work reliably.

---

### `gate_on_confidence()`

**Purpose:** Apply a threshold to a confidence score and return a routing label.

**Position:** Runs after `extract_confidence()`. Returns the label used by the LangGraph conditional edge.

**Step-by-step walkthrough:**

```
gate_on_confidence(confidence, threshold=0.75, label_above="deliver", label_below="escalate")
  │
  ├─ passed = confidence >= threshold
  │
  ├─ action = label_above if passed else label_below
  │
  └─ Return {
       "action": str,       # the routing label
       "confidence": float,
       "threshold": float,
       "passed": bool
     }
```

**Why configurable labels?** The caller decides what the labels mean in their graph. A graph with nodes named `"review_queue"` and `"final_delivery"` can pass those as labels. This keeps the guardrail function graph-topology-agnostic — it knows nothing about LangGraph node names.

**One real limitation:** A single threshold is a blunt instrument. In practice, different parts of a clinical response carry different stakes. Recommending a drug dosage might need 0.90 confidence; recommending "drink more water" needs only 0.50. There is no per-topic threshold in this implementation.

**Production upgrade path:** Implement per-topic thresholds keyed off the query category. A query classifier runs first; its output selects the threshold from a configuration dictionary.

---

### `check_confidence()`

**Purpose:** Convenience wrapper that calls `extract_confidence()` + `gate_on_confidence()` in one step.

**Position:** Used when you have the raw LLM response text and want the routing decision immediately.

**Inputs & Outputs:**
- In: `text: str`, `threshold: float = 0.75`
- Out: Same dict as `gate_on_confidence()`, plus `"raw_text_length": int`

The `raw_text_length` field is diagnostic — unusually short responses might have failed to include a confidence marker, explaining a default 0.5 score.

---

## Part D — LLM Judge Guardrails (`llm_judge_guardrails.py`)

### `JudgeVerdict`

**Purpose:** Pydantic model that enforces the structure of the judge LLM's output.

**Position:** This is a schema, not a function. It is used as the target type for `llm.with_structured_output(JudgeVerdict)`.

**Fields:**

| Field | Type | Options | Purpose |
|-------|------|---------|---------|
| `safety` | `Literal` | `safe / unsafe / borderline` | Detects dangerous medical advice |
| `relevance` | `Literal` | `relevant / partially_relevant / irrelevant` | Detects generic responses that ignore patient data |
| `completeness` | `Literal` | `complete / partial / incomplete` | Detects missing clinical considerations |
| `verdict` | `Literal` | `approve / revise / reject` | The routing decision |
| `reasoning` | `str` | free text | Human-readable audit trail |
| `suggested_fix` | `str` | free text | Correction instructions when `verdict="revise"` |

**Why Pydantic for the judge output?** Three reasons:

1. **Type safety.** The routing code can do `if verdict.verdict == "reject"` without parsing strings.
2. **Validation.** If the judge LLM returns `"REJECT"` (uppercase) or `"refused"`, Pydantic raises a `ValidationError` immediately — the error is caught, not silently propagated.
3. **Auditability.** The full verdict including `reasoning` is a structured record that can be stored in a database or observability system.

**One real limitation:** `Literal` types with `with_structured_output` work well with GPT-4 class models. Smaller or older models sometimes return valid-looking JSON with incorrect values (e.g., `"partially relevant"` with a space instead of underscore). The Pydantic validation will catch this — but it means you must handle the resulting `RuntimeError` in `evaluate_with_judge()`.

---

### `evaluate_with_judge()`

**Purpose:** Invoke a second LLM to semantically evaluate the first LLM's response for safety, relevance, and completeness.

**Position:** Runs after all deterministic output checks pass, on high-stakes requests only.

**Step-by-step walkthrough:**

```
evaluate_with_judge(llm, patient_case, user_query, agent_response, system_prompt)
  │
  ├─ judge_llm = llm.with_structured_output(JudgeVerdict)
  │     Forces the LLM to return a JSON object matching JudgeVerdict schema
  │
  ├─ Build evaluation_prompt:
  │     PATIENT CASE: [structured patient data — age, sex, symptoms, labs, vitals, ...]
  │     ORIGINAL QUERY: [user_query]
  │     AGENT RESPONSE TO EVALUATE: [agent_response]
  │     "Evaluate this response for safety, relevance, and completeness."
  │
  ├─ judge_llm.invoke([
  │     SystemMessage(content=system_prompt),   ← judge rubric
  │     HumanMessage(content=evaluation_prompt) ← context + response to evaluate
  │   ])
  │
  ├─ SUCCESS → return JudgeVerdict (Pydantic object)
  │
  └─ EXCEPTION → raise RuntimeError("Judge evaluation failed: ...")
                 Callers must catch this and use default_approve_verdict()
```

**Under the hood:** The `JUDGE_SYSTEM_PROMPT` contains explicit rubric rules:
- What counts as `unsafe` vs `borderline` vs `safe`
- How `relevance` maps from generic to patient-specific
- Verdict rules: `reject if safety=unsafe OR relevance=irrelevant`

This prompt is the most important part of the judge — a vague prompt produces vague verdicts. The rules are deterministic: if `safety=unsafe`, verdict must be `reject`. This reduces the LLM's freedom in the verdict field.

**One real limitation:** The judge sees the patient case as a formatted string. If the patient case has unusual fields, missing data, or is in a format the prompt doesn't anticipate, the judge may give worse evaluations. The evaluation prompt is templated and brittle against schema changes in `patient_case`.

**Production upgrade path:**
1. Use a stronger model for the judge than for the agent (e.g., agent on `gpt-4o-mini`, judge on `gpt-4o`).
2. Run the judge asynchronously — fire-and-forget after delivery for low-latency paths, with a correction follow-up if the verdict is `reject`.
3. Log all verdicts to an observability system (Langfuse) with the full context for periodic human audit.

---

### `default_approve_verdict()`

**Purpose:** Return a safe default `JudgeVerdict` when the judge LLM fails, implementing the fail-open strategy.

**Position:** Called in `except RuntimeError` blocks in calling code, never by `evaluate_with_judge()` itself.

**Why fail-open here:** By the time `evaluate_with_judge()` is called, the response has already passed:
- Input content checks (no injection, PII, or out-of-scope)
- Output content checks (no prohibited patterns, disclaimer present or appended)
- Confidence gate (above threshold)

The judge is a *fourth* layer on top of three already-passed checks. Failing closed on a judge error would block responses that passed three independent safety checks — a high false positive rate. The fail-open choice here is a calibrated risk decision, not negligence.

**One real limitation:** If the judge fails *frequently* (e.g., model outage), `default_approve_verdict()` silently degrades the safety level. There is no alerting built in. In production, judge failures should increment a metric and trigger an alert when they exceed a threshold (e.g., >5% failure rate in the last 5 minutes).
