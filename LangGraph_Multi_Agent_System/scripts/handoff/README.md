# Area 2 — Handoff Patterns

> **Learning sequence position:** Area 2 of 9.
> Study this area **after** `scripts/tools/` (Area 1 — tool binding and ReAct loops) and **before** `scripts/guardrails/` (Area 3 — safety guardrails).

This folder contains six LangGraph pattern scripts that demonstrate **how agents hand off work to each other**. The question each pattern answers is: "After this agent finishes, who runs next — and how does that decision get made?"

---

## What Are These Scripts?

Each script builds a complete LangGraph `StateGraph` where multiple agents collaborate on a shared clinical case. They differ in *how the routing decision is made*:

| # | Script | Who Decides Next Agent | Routing Mechanism |
|---|--------|------------------------|-------------------|
| 1 | [`linear_pipeline.py`](linear_pipeline.py) | You (the developer) at build time | Fixed `add_edge()` |
| 2 | [`conditional_routing.py`](conditional_routing.py) | Python router function at runtime | `add_conditional_edges()` |
| 3 | [`command_handoff.py`](command_handoff.py) | The LLM — via transfer tool calls | `Command(goto=, update=)` |
| 4 | [`supervisor.py`](supervisor.py) | Supervisor LLM — centrally dispatches | `add_conditional_edges()` from coordinator |
| 5 | [`multihop_depth_guard.py`](multihop_depth_guard.py) | Python guard checks depth limit after each hop | `add_conditional_edges()` with counter |
| 6 | [`parallel_fanout.py`](parallel_fanout.py) | Coordinator fans out to all; runs concurrently | `Send` API with reducer |

---

## Scripts — Recommended Reading Order

Study them in the order shown. Each script introduces one new LangGraph concept and builds on the previous.

| # | Script | New Concept | Root Modules |
|---|--------|-------------|--------------|
| 1 | [`linear_pipeline.py`](linear_pipeline.py) | `add_edge()`, `HandoffContext`, manual ReAct loop | `core/models`, `tools/` |
| 2 | [`conditional_routing.py`](conditional_routing.py) | `add_conditional_edges()`, router functions | `core/models`, `tools/` |
| 3 | [`command_handoff.py`](command_handoff.py) | `Command(goto=, update=)`, transfer tools | `core/models`, `core/exceptions` |
| 4 | [`supervisor.py`](supervisor.py) | Supervisor node, worker-to-supervisor edges | `core/models`, `tools/` |
| 5 | [`multihop_depth_guard.py`](multihop_depth_guard.py) | Depth guard, circuit breaker, `HandoffLimitReached` | `core/exceptions` |
| 6 | [`parallel_fanout.py`](parallel_fanout.py) | `Send` API, `operator.add` reducer, parallel branches | `core/models`, `tools/` |

---

## Documentation Chapters

Full educational chapters for every pattern live in the [`docs/`](docs/) subfolder:

| Chapter | File | What it covers |
|---------|------|----------------|
| Overview | [`docs/00_overview.md`](docs/00_overview.md) | What handoffs are, why LangGraph, 6-pattern comparison, composition guide |
| Pattern 1 | [`docs/01_linear_pipeline.md`](docs/01_linear_pipeline.md) | Fixed edges, `HandoffContext` sender/receiver model, manual ReAct loop |
| Pattern 2 | [`docs/02_conditional_routing.md`](docs/02_conditional_routing.md) | Router functions, risk-based branching, zero-token routing |
| Pattern 3 | [`docs/03_command_handoff.md`](docs/03_command_handoff.md) | `Command` objects, transfer tools, LLM-driven routing |
| Pattern 4 | [`docs/04_supervisor.md`](docs/04_supervisor.md) | Supervisor LLM, worker return-to-coordinator pattern |
| Pattern 5 | [`docs/05_multihop_depth_guard.md`](docs/05_multihop_depth_guard.md) | Depth guard, circuit breaker, partial-data fallback |
| Pattern 6 | [`docs/06_parallel_fanout.md`](docs/06_parallel_fanout.md) | `Send` API, `operator.add` reducer, parallel agent execution |

Start with [`docs/00_overview.md`](docs/00_overview.md) for a map of the whole module.

---

## Root Module Connection

All scripts import shared infrastructure from the project root:

```
core/
├── config.py        ← get_llm() — centralised LLM client for all agents
├── models.py        ← PatientCase, HandoffContext — canonical domain models
└── exceptions.py    ← HandoffLimitReached — raised when depth guard trips

tools/               ← Clinical tool functions (analyze_symptoms, check_drug_interactions, ...)
observability/
└── callbacks.py     ← build_callback_config() — injects Langfuse tracing into every LLM call
```

The scripts in **this folder** teach the **routing pattern**. The files above provide the clinical domain logic and infrastructure they depend on.

---

## Prerequisites

Before studying this area, ensure you understand:

1. **Tool binding** — how `llm.bind_tools(tools)` works and how tool calls are processed in a ReAct loop (`scripts/tools/`).
2. **LangGraph StateGraph basics** — how to define nodes and wire a simple graph (`scripts/foundations/`).
3. **Python TypedDict** — used as the graph state schema throughout.

---

## How to Run Any Script

```bash
# From the project root:
cd "D:/Agentic AI/LangGraph_Multi_Agent_System"

# Pattern 1 — Fixed pipeline (LLM required):
python -m scripts.handoff.linear_pipeline

# Pattern 2 — Conditional routing (LLM required):
python -m scripts.handoff.conditional_routing

# Pattern 3 — Command-based handoff (LLM required):
python -m scripts.handoff.command_handoff

# Pattern 4 — Supervisor pattern (LLM required):
python -m scripts.handoff.supervisor

# Pattern 5 — Multihop depth guard (LLM required):
python -m scripts.handoff.multihop_depth_guard

# Pattern 6 — Parallel fan-out (LLM required):
python -m scripts.handoff.parallel_fanout
```

> **NOTE:** All six scripts make real LLM calls. Set your `GOOGLE_API_KEY` or `OPENAI_API_KEY` in a `.env` file at the project root before running them.

---

## Broader Context

For the higher-level architecture — how handoff patterns fit the full multi-agent system design — see:

- [`LangGraph_Multi_Agent_System/docs/handoff/`](../../docs/handoff/) (if present)
- After completing this area, continue to `scripts/guardrails/` (Area 3) — adding safety guardrails to the pipelines you learned to build here.
