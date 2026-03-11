# Area 7 — Multi-Agent System (MAS) Architecture Patterns

> **Learning sequence position:** Area 7 of 9.
> Study this area **after** `scripts/orchestration/` (Area 6 — orchestration primitives) and **before** `scripts/memory/` (Area 8).
> Prerequisites: tools (Area 1), handoff (Area 2), guardrails (Area 3), HITL (Area 4), communication (Area 5), orchestration (Area 6).

---

## What Are These Scripts?

Each script in this folder demonstrates a **system-level MAS architecture pattern** — a repeatable blueprint for how to compose multiple AI agents to solve a problem. These are not about what individual agents do internally (see `agents/` for that). They are about **how agents are coordinated**: who decides what to do next, in what order agents run, how their results are combined, and how the system handles disagreement, uncertainty, and failure.

The seven patterns span the design space from simple and deterministic (sequential pipeline) to dynamic and adversarial (debate + judge):

| # | Script | Pattern Name | Coordination Style | Concurrency | Primary Use Case |
|---|--------|-------------|-------------------|-------------|-----------------|
| 1 | [`supervisor_orchestration.py`](supervisor_orchestration.py) | Supervisor Orchestration | Centralised — LLM supervisor routes | Sequential (LLM-decided) | Dynamic task order; needs adaptability |
| 2 | [`sequential_pipeline.py`](sequential_pipeline.py) | Sequential Pipeline | Fixed — developer decides order at build time | Sequential | Predictable, deterministic workflows |
| 3 | [`parallel_voting.py`](parallel_voting.py) | Parallel Voting | Decentralised — aggregator forms consensus | Parallel (Send API) | High-stakes decisions; bias mitigation |
| 4 | [`adversarial_debate.py`](adversarial_debate.py) | Adversarial Debate | Adversarial — pro/con agents + judge | Sequential (rounds) | Complex dilemmas; documented rationale |
| 5 | [`hierarchical_delegation.py`](hierarchical_delegation.py) | Hierarchical Delegation | Hierarchical — L3 → L2 → L1 chain of command | Sequential by level | Large organisations; layered information filtering |
| 6 | [`map_reduce_fanout.py`](map_reduce_fanout.py) | Map-Reduce Fan-Out | Divide-and-conquer — map independent tasks, reduce results | Parallel (Send API) | Large tasks with independent sub-problems |
| 7 | [`reflection_self_critique.py`](reflection_self_critique.py) | Reflection / Self-Critique | Iterative — generator + critic loop | Sequential (iterative) | Safety-critical outputs; single-pass quality insufficient |

---

## Documentation Chapters

Full educational chapters for every pattern live in the [`docs/`](docs/) subfolder:

| Chapter | File | What it covers |
|---------|------|----------------|
| Overview | [`docs/00_overview.md`](docs/00_overview.md) | What MAS architectures are, pattern comparison table, 2D landscape map, when to choose each family |
| Pattern 1 | [`docs/01_supervisor_orchestration.md`](docs/01_supervisor_orchestration.md) | Centralised supervisor loop; `add_conditional_edges`; max-iteration guard; `AGENT_REGISTRY` |
| Pattern 2 | [`docs/02_sequential_pipeline.md`](docs/02_sequential_pipeline.md) | Fixed-order pipeline; context accumulation; `accumulated_context`; `synthesizer_stage` |
| Pattern 3 | [`docs/03_parallel_voting.md`](docs/03_parallel_voting.md) | `Send` API fan-out; `operator.add` reducer; independent assessment; agreement score |
| Pattern 4 | [`docs/04_adversarial_debate.md`](docs/04_adversarial_debate.md) | Pro/con agents; rebuttal round; judge ruling; anchoring bias mitigation |
| Pattern 5 | [`docs/05_hierarchical_delegation.md`](docs/05_hierarchical_delegation.md) | 3-level org hierarchy; information filtering; L3 specialists → L2 leads → L1 executive |
| Pattern 6 | [`docs/06_map_reduce_fanout.md`](docs/06_map_reduce_fanout.md) | Mapper → parallel workers → reducer → producer; different sub-tasks vs voting |
| Pattern 7 | [`docs/07_reflection_self_critique.md`](docs/07_reflection_self_critique.md) | Generate-critique-revise loop; `route_after_critique`; `revision_history`; severity scoring |

Start with [`docs/00_overview.md`](docs/00_overview.md) for a map of the entire MAS module.

---

## Root Module Connections

All scripts use shared infrastructure from the project root:

```
agents/
├── TriageAgent     ← Level 3 specialist used by patterns 1, 2, 3, 5, 6
├── DiagnosticAgent ← Level 3 specialist used by patterns 1, 2, 3, 5, 6
└── PharmacistAgent ← Level 3 specialist used by patterns 1, 2, 3, 5, 6

core/
├── config.py  ← get_llm()       — centralises LLM instantiation
└── models.py  ← PatientCase     — canonical domain model

observability/
└── callbacks.py ← build_callback_config() — attaches Langfuse tracing to every LLM call
```

Patterns 4 (debate), 5 (hierarchy L2/L1), and 7 (reflection) make direct LLM calls with custom system prompts rather than using `BaseAgent` subclasses, because their roles (judge, team lead, critic) are pattern-specific and not reusable across other patterns.

---

## Conceptual Background

For the high-level theory behind each pattern — without code — see the conceptual guide in:

- [`../../Principles of Multi-Agent Systems/07-MAS-Architectures.md`](../../Principles%20of%20Multi-Agent%20Systems/07-MAS-Architectures.md) — Descriptions of the 8 canonical MAS architecture patterns, trade-offs, and design criteria.

The scripts in this folder are the **runnable implementations** of those conceptual patterns. The `docs/` chapters bridge the two: they walk through the code line-by-line while grounding each implementation detail in the conceptual model.

---

## Prerequisites

Before studying this area, ensure you understand:

1. **LangGraph `StateGraph` basics** — defining nodes, `add_edge`, `add_conditional_edges` (`scripts/handoff/` Area 2).
2. **The `Send` API for parallel fan-out** — `scripts/handoff/docs/06_parallel_fanout.md` (Area 2, Pattern 6).
3. **`operator.add` as a state reducer** — used in both `parallel_voting.py` and `map_reduce_fanout.py`.
4. **Orchestration primitives** — supervisor loops, routing functions, state accumulation (`scripts/orchestration/` Area 6).

---

## How to Run Any Script

```bash
# From the project root:
cd "D:/Agentic AI/LangGraph_Multi_Agent_System"

python -m scripts.MAS_architectures.supervisor_orchestration
python -m scripts.MAS_architectures.sequential_pipeline
python -m scripts.MAS_architectures.parallel_voting
python -m scripts.MAS_architectures.adversarial_debate
python -m scripts.MAS_architectures.hierarchical_delegation
python -m scripts.MAS_architectures.map_reduce_fanout
python -m scripts.MAS_architectures.reflection_self_critique
```

> **NOTE:** All scripts make real LLM calls (except internal agent mock modes). Set `GOOGLE_API_KEY` or `OPENAI_API_KEY` in a `.env` file at the project root before running. Each script uses the same `PatientCase` domain — a 68-year-old male with chest pain and elevated troponin — enabling direct comparison of how each architecture processes the same clinical data.

---

## Architecture Positioning

```
Area 1 — tools/              ← What individual agents can do
Area 2 — handoff/            ← How two agents hand work to each other
Area 3 — guardrails/         ← Automated safety constraints
Area 4 — HITL/               ← Human review gates
Area 5 — communication/      ← Message passing and coordination protocols
Area 6 — orchestration/      ← Orchestration primitives (supervisor loops, routing)
Area 7 — MAS_architectures/  ← Complete system-level patterns  ← YOU ARE HERE
Area 8+ — memory, observability, ...
```
