# Area 3 — Guardrail Patterns

> **Learning sequence position:** Area 3 of 9.
> Study this area **after** `scripts/handoff/` (Area 2 — conditional routing) and **before** `scripts/memory/` (Area 4).

This folder contains five LangGraph pattern scripts that demonstrate **how to wire guardrails into a graph** using conditional routing. The guardrail *logic itself* lives in the root `guardrails/` module — these scripts show the *architectural pattern*, not the validation rules.

---

## What Are These Scripts?

Each script builds a small but complete LangGraph `StateGraph` that encapsulates a specific safety strategy. All five patterns share a common spine:

```
User Input → [Guard Node] → Conditional Edge → Safe Path or Rejection Path
```

They differ in *what* they guard, *where* in the pipeline they sit, and *how many routing outcomes* they produce.

---

## Scripts — Recommended Reading Order

| # | Script | Pattern | Routing | Root Module |
|---|--------|---------|---------|-------------|
| 1 | [`input_validation.py`](input_validation.py) | Binary pass/fail before LLM runs | `agent` / `reject` | `guardrails/input_guardrails.py` |
| 2 | [`output_validation.py`](output_validation.py) | Three-way triage after LLM runs | `deliver` / `auto_fix` / `block` | `guardrails/output_guardrails.py` |
| 3 | [`confidence_gating.py`](confidence_gating.py) | Threshold routing on LLM certainty | `deliver` / `escalate` | `guardrails/confidence_guardrails.py` |
| 4 | [`layered_validation.py`](layered_validation.py) | Input + output stacked in one pipeline | `reject` / `deliver` / `auto_fix` / `block` | both above |
| 5 | [`llm_as_judge.py`](llm_as_judge.py) | Second LLM evaluates first LLM's output | `approve` / `revise` / `reject` | `guardrails/llm_judge_guardrails.py` |

Study them **in order**. Each script introduces one new LangGraph concept and builds on the previous one.

---

## Documentation Chapters

Full educational chapters for every pattern live in the [`docs/`](docs/) subfolder:

| Chapter | File | What it covers |
|---------|------|----------------|
| Overview | [`docs/00_overview.md`](docs/00_overview.md) | What guardrails are, why LangGraph, learning progression, defence-in-depth |
| Pattern A | [`docs/01_input_validation.md`](docs/01_input_validation.md) | Binary routing, `validate_input()`, PII / injection / scope checks |
| Pattern B | [`docs/02_output_validation.md`](docs/02_output_validation.md) | Three-way routing, auto-fix, severity-based decision table |
| Pattern C | [`docs/03_confidence_gating.md`](docs/03_confidence_gating.md) | Threshold gating, `add_messages` reducer, escalation path |
| Pattern D | [`docs/04_layered_validation.md`](docs/04_layered_validation.md) | Stacked input + output guardrails, four execution paths, token savings |
| Pattern E | [`docs/05_llm_as_judge.md`](docs/05_llm_as_judge.md) | Two-LLM pattern, `JudgeVerdict`, semantic vs deterministic checks |

Start with [`docs/00_overview.md`](docs/00_overview.md) for a map of the whole module.

---

## Root Module Connection

The scripts import their core logic from the project's `guardrails/` root package:

```
guardrails/
├── input_guardrails.py      ← validate_input(), detect_pii(), check_medical_scope()
├── output_guardrails.py     ← validate_output(), check_prohibited_content()
├── confidence_guardrails.py ← extract_confidence(), gate_on_confidence()
└── llm_judge_guardrails.py  ← JudgeVerdict, evaluate_with_judge()
```

The scripts in **this folder** (`scripts/guardrails/`) teach the **LangGraph wiring pattern**. The files above contain the **validation logic**. Keep that separation in mind as you read.

---

## Higher-Level Architecture Documentation

For the broader guardrails architecture — how these patterns fit the full multi-agent system design — see:

- [`LangGraph_Multi_Agent_System/docs/guardrails/README.md`](../../docs/guardrails/README.md)

---

## Prerequisites

Before studying this area, ensure you understand:

1. **LangGraph StateGraph basics** — how to define nodes and edges (`scripts/foundations/`).
2. **Conditional routing** — `add_conditional_edges()` (`scripts/handoff/conditional_routing.py`).
3. **Basic Python TypedDict** — used throughout as the graph state schema.

---

## How to Run Any Script

```bash
# From the project root:
cd "D:/Agentic AI/LangGraph_Multi_Agent_System"

# Run Pattern A (input validation):
python -m scripts.guardrails.input_validation

# Run Pattern B (output validation):
python -m scripts.guardrails.output_validation

# Run Pattern C (confidence gating — requires LLM API key):
python -m scripts.guardrails.confidence_gating

# Run Pattern D (layered validation — requires LLM API key):
python -m scripts.guardrails.layered_validation

# Run Pattern E (LLM-as-judge — requires LLM API key):
python -m scripts.guardrails.llm_as_judge
```

> **NOTE:** Patterns C, D, and E make real LLM calls. Set your `GOOGLE_API_KEY` or `OPENAI_API_KEY` in a `.env` file at the project root before running them.
