# guardrails/

The `guardrails` package is the **safety boundary** of the LangGraph Multi-Agent System (MAS). It provides four composable layers that validate inputs before they reach an LLM, validate outputs before they reach a user, gate on self-assessed confidence, and semantically evaluate responses with a second LLM.

---

## Package Structure

```
guardrails/
├── __init__.py                 # Public API — imports from all four modules
├── input_guardrails.py         # Validate inputs BEFORE the LLM call
├── output_guardrails.py        # Validate outputs AFTER the LLM call
├── confidence_guardrails.py    # Route on self-assessed certainty
└── llm_judge_guardrails.py     # Semantic evaluation via a second LLM
```

---

## Where This Fits in the MAS

```
User / Agent Input
      |
[input_guardrails]      ← PII, injection, scope — blocks before any token cost
      |
[LLM Agent]             ← core reasoning
      |
[output_guardrails]     ← prohibited content, missing disclaimers, auto-fix
      |
[confidence_guardrails] ← is the agent certain enough to deliver?
      |
[llm_judge_guardrails]  ← optional: semantic safety / relevance / completeness
      |
Deliver to user  OR  Escalate to human review
```

---

## Quick Import

```python
# Input
from guardrails import validate_input, detect_pii, detect_prompt_injection, check_medical_scope

# Output
from guardrails import validate_output, check_prohibited_content, check_safety_disclaimers, add_human_review_flag

# Confidence
from guardrails import extract_confidence, gate_on_confidence, check_confidence

# LLM Judge
from guardrails import JudgeVerdict, evaluate_with_judge, default_approve_verdict
```

---

## Related Pattern Scripts

The `scripts/guardrails/` directory shows how each guardrail wires into a LangGraph graph:

| Script | Pattern | What it demonstrates |
|--------|---------|----------------------|
| `scripts/guardrails/input_validation.py` | A | Binary pass/fail routing before the LLM |
| `scripts/guardrails/output_validation.py` | B | Three-way deliver/fix/block routing after the LLM |
| `scripts/guardrails/confidence_gating.py` | C | Threshold-based escalation on self-assessed confidence |
| `scripts/guardrails/layered_validation.py` | D | Full input → agent → output pipeline (defence-in-depth) |
| `scripts/guardrails/llm_as_judge.py` | E | Semantic evaluation with structured `JudgeVerdict` output |

---

## Full Documentation

Pattern-focused reference (what each graph looks like):
- [`docs/guardrails/01_input_validation.md`](../docs/guardrails/01_input_validation.md)
- [`docs/guardrails/02_output_validation.md`](../docs/guardrails/02_output_validation.md)
- [`docs/guardrails/03_confidence_gating.md`](../docs/guardrails/03_confidence_gating.md)
- [`docs/guardrails/04_layered_validation.md`](../docs/guardrails/04_layered_validation.md)
- [`docs/guardrails/05_llm_as_judge.md`](../docs/guardrails/05_llm_as_judge.md)

Learning-oriented documentation (WHY the module is designed this way):

| Chapter | File | Content |
|---------|------|---------|
| 1 — The Big Picture | [`docs/guardrails/06_big_picture.md`](../docs/guardrails/06_big_picture.md) | Mental model, architecture diagram, design philosophy |
| 2 — Core Concepts | [`docs/guardrails/07_core_concepts.md`](../docs/guardrails/07_core_concepts.md) | 5 concepts you must understand before reading the code |
| 3 — Deep Dive | [`docs/guardrails/08_deep_dive.md`](../docs/guardrails/08_deep_dive.md) | Every function and class, step by step |
| 4 — Design Decisions | [`docs/guardrails/09_design_decisions.md`](../docs/guardrails/09_design_decisions.md) | Non-obvious choices, tradeoffs, when you'd change them |
| 5 — Gotchas & Anti-Patterns | [`docs/guardrails/10_gotchas.md`](../docs/guardrails/10_gotchas.md) | What the module gets wrong, integration mistakes, scope limits |
| 6 — Putting It Together | [`docs/guardrails/11_putting_it_together.md`](../docs/guardrails/11_putting_it_together.md) | Golden path, bypass guide, handoffs, scaling |
| 7 — Quick Reference | [`docs/guardrails/12_quick_reference.md`](../docs/guardrails/12_quick_reference.md) | Function table, copy-paste snippets, decision flowchart |

**Recommended reading order:** Ch 1 → Ch 2 → Ch 3 → then any of Ch 4–7 in the order relevant to your work.
