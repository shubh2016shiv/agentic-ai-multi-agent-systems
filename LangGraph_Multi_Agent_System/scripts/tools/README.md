# Area 1 — Tool Patterns

> **Learning sequence position:** Area 1 of 9.
> Study this area **first** — it is the prerequisite for every other area in the course.
> Followed by `scripts/handoff/` (Area 2 — conditional routing and agent handoffs).

This folder contains five LangGraph pattern scripts that demonstrate **how to bind tools to agents, execute them inside a graph, get structured output from LLMs, select tools at runtime, and recover from tool failures**. The clinical tool implementations themselves live in the root `tools/` package — these scripts demonstrate the architectural patterns for using them.

---

## What Are These Scripts?

Each script teaches one focused aspect of working with tools in LangGraph. They build on each other: start with `tool_binding.py` and work through them in order.

| # | Script | Pattern | New LangGraph Concept | Root Module Used |
|---|--------|---------|----------------------|-----------------|
| 1 | [`tool_binding.py`](tool_binding.py) | Scope tools per-agent via `bind_tools()` | `bind_tools()` immutability | `tools/`, `core/config` |
| 2 | [`toolnode_patterns.py`](toolnode_patterns.py) | `ToolNode` as graph node vs internal invocation | `StateGraph`, `ToolNode`, `add_messages`, `add_conditional_edges` | `tools/`, `core/config` |
| 3 | [`structured_output.py`](structured_output.py) | Validated Pydantic output from LLMs | `with_structured_output()`, response-format tool | `tools/`, `core/config` |
| 4 | [`dynamic_tool_selection.py`](dynamic_tool_selection.py) | Select tools at runtime from a registry | Pure-Python selector node, runtime `bind_tools()` | `tools/`, `core/config`, `core/models` |
| 5 | [`tool_error_handling.py`](tool_error_handling.py) | Recover from tool failures gracefully | `handle_tool_errors=True`, retry counter | `tools/`, `core/config` |

Study them **in order**. Each script introduces one new concept that the next script builds on.

---

## Documentation Chapters

Full educational chapters for every pattern live in the [`docs/`](docs/) subfolder:

| Chapter | File | What it covers |
|---------|------|----------------|
| Overview | [`docs/00_overview.md`](docs/00_overview.md) | What tools are, why LangGraph, 5-pattern progression, composition guide, vocabulary |
| Script 1 | [`docs/01_tool_binding.md`](docs/01_tool_binding.md) | `bind_tools()` immutability; context scoping; scoped vs over-scoped comparison |
| Script 2 | [`docs/02_toolnode_patterns.md`](docs/02_toolnode_patterns.md) | `ToolNode` as graph node vs `ToolNode.invoke()`; `add_messages` reducer; ReAct loop |
| Script 3 | [`docs/03_structured_output.md`](docs/03_structured_output.md) | `with_structured_output()`; response-format tool; Pydantic `ValidationError` |
| Script 4 | [`docs/04_dynamic_tool_selection.md`](docs/04_dynamic_tool_selection.md) | `TOOL_REGISTRY`; pure-Python selector node; runtime `bind_tools()` |
| Script 5 | [`docs/05_tool_error_handling.md`](docs/05_tool_error_handling.md) | `handle_tool_errors=True`; manual `try/except`; retry counter; forced fallback |

Start with [`docs/00_overview.md`](docs/00_overview.md) to understand the full picture before diving into individual patterns.

---

## Root Module Connections

All scripts import from shared project infrastructure:

```
tools/
├── __init__.py                  ← re-exports all 6 clinical tools
├── triage_tools.py              ← analyze_symptoms, assess_patient_risk
├── pharmacology_tools.py        ← check_drug_interactions, lookup_drug_info, calculate_dosage_adjustment
└── guidelines_tools.py          ← lookup_clinical_guideline

core/
├── config.py    ← get_llm()     — centralised LLM instantiation (OpenAI / Gemini / LM Studio)
└── models.py    ← PatientCase   — canonical domain model used in test scenarios

observability/
└── callbacks.py ← build_callback_config(trace_name=...) — Langfuse tracing on every LLM call
```

The `tools/` package uses a `@observe_tool` decorator from `observability/decorators.py` to trace individual tool calls. The scripts themselves use `build_callback_config()` to trace LLM calls.

---

## The 6 Clinical Tool Functions

The root `tools/` package exports these `@tool`-decorated functions, used across all 5 pattern scripts:

| Tool | Domain | Parameters |
|------|--------|------------|
| `analyze_symptoms` | Triage | `symptoms`, `patient_age`, `patient_sex` |
| `assess_patient_risk` | Triage | `age`, `conditions`, `medications`, `vitals` |
| `check_drug_interactions` | Pharmacology | `medications` (list) |
| `lookup_drug_info` | Pharmacology | `drug_name` |
| `calculate_dosage_adjustment` | Pharmacology | `drug_name`, `current_dose`, `egfr`, `weight_kg` |
| `lookup_clinical_guideline` | Guidelines | `condition`, `topic` |

---

## Prerequisites

None — this is Area 1. You need:
- Basic Python (functions, classes, `TypedDict`)
- A valid `OPENAI_API_KEY`, `GOOGLE_API_KEY`, or LM Studio running locally (see `core/config.py`)
- Project dependencies installed

---

## How to Run Any Script

```bash
# From the project root:
cd "D:/Agentic AI/LangGraph_Multi_Agent_System"

python -m scripts.tools.tool_binding
python -m scripts.tools.toolnode_patterns
python -m scripts.tools.structured_output
python -m scripts.tools.dynamic_tool_selection
python -m scripts.tools.tool_error_handling
```

> **NOTE:** All scripts make real LLM calls. Each uses `get_llm()` which reads from `core/config.py` — set your API key in `.env` before running. Scripts 1–3 use a COPD or CKD patient test case; Script 4 runs two test cases (renal + respiratory); Script 5 uses flaky demo tools that intentionally fail on first call.

---

## Architecture Positioning

```
Area 1 — tools/       ← What individual agents can do              ← YOU ARE HERE
Area 2 — handoff/     ← How agents pass work to each other
Area 3 — guardrails/  ← Automated safety constraints
Area 4 — HITL/        ← Human review gates
Area 5 — communication/   ← Message passing and coordination protocols
Area 6 — orchestration/   ← Supervisor loops and orchestration primitives
Area 7 — MAS_architectures/  ← Complete system-level patterns
Area 8+ — memory, observability, ...
```

Everything built in Areas 2–9 relies on the tool patterns learned here. Specifically:
- **Area 2 (Handoff)** — uses `ToolNode` from Script 2 in the linear pipeline pattern.
- **Area 3 (Guardrails)** — structured output (Script 3) is the mechanism guardrails use to get validated agent responses.
- **Area 4 (HITL)** — `ToolNode` as a graph node (Script 2 Pattern 1) enables HITL tool confirmation by making tool execution a separately addressable node.
- **Area 7 (MAS)** — `BaseAgent.bind_tools()` (Script 1) is called inside every specialist agent used by MAS architecture patterns.
