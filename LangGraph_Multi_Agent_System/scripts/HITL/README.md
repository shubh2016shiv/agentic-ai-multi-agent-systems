# Area 4 — Human-in-the-Loop (HITL) Patterns

> **Learning sequence position:** Area 4 of 9.
> Study this area **after** `scripts/tools/` (Area 1), `scripts/handoff/` (Area 2), and `scripts/guardrails/` (Area 3).
> The prerequisite areas establish tool calling, multi-agent routing, and safety guardrails. HITL patterns add the human dimension: pausing the pipeline, receiving structured human input, and resuming based on that input.

---

## What Are These Scripts?

Each script in this folder demonstrates a different way a human can intervene in a running LangGraph pipeline. They all use the same two LangGraph primitives:

- **`interrupt(payload)`** — physically pauses graph execution, saves state to a checkpointer, and surfaces a structured payload to the caller.
- **`Command(resume=value)`** — resumes the paused graph, passing the human's decision back to the node that called `interrupt()`.

The patterns differ in **what the human can do** when the pipeline pauses:

| # | Script | Pattern | What the Human Can Do | HITL Interaction Style |
|---|--------|---------|----------------------|------------------------|
| A | [`basic_approval.py`](basic_approval.py) | Basic Approval | Approve or reject | Boolean resume (`True`/`False`) |
| B | [`tool_call_confirmation.py`](tool_call_confirmation.py) | Tool Call Confirmation | Execute or skip a tool call | Dict resume (`{"action": "execute"|"skip"}`) |
| C | [`edit_before_approve.py`](edit_before_approve.py) | Edit Before Approve | Approve, edit, or reject with content | Rich dict resume with `"content"` or `"reason"` |
| D | [`multi_step_approval.py`](multi_step_approval.py) | Multi-Step Approval | Approve or reject at two sequential checkpoints | Sequential boolean resumes, same `thread_id` |
| E | [`escalation_chain.py`](escalation_chain.py) | Escalation Chain | Junior: approve/escalate/reject; Senior: approve/reject | Tiered dict resumes, conditional second interrupt |

---

## Documentation Chapters

Full educational chapters for every pattern live in the [`docs/`](docs/) subfolder:

| Chapter | File | What it covers |
|---------|------|----------------|
| Overview | [`docs/00_overview.md`](docs/00_overview.md) | What HITL is, why LangGraph, 5-pattern comparison, how HITL composes with tools/guardrails/handoffs |
| Pattern A | [`docs/01_basic_approval.md`](docs/01_basic_approval.md) | `interrupt()` and `Command(resume=)` fundamentals; `MemorySaver`; `thread_id`; node restart behaviour |
| Pattern B | [`docs/02_tool_call_confirmation.md`](docs/02_tool_call_confirmation.md) | Intercepting tool calls before execution; structured dict resume; conditional routing after HITL |
| Pattern C | [`docs/03_edit_before_approve.md`](docs/03_edit_before_approve.md) | Rich resume payloads; human-injected content; approve/edit/reject from a single interrupt point |
| Pattern D | [`docs/04_multi_step_approval.md`](docs/04_multi_step_approval.md) | Multiple sequential interrupt points in one graph; `run_multi_interrupt_cycle`; early rejection saves tokens |
| Pattern E | [`docs/05_escalation_chain.md`](docs/05_escalation_chain.md) | Tiered authority; junior filters, senior decides; conditional interrupt (not every path hits every gate) |

Start with [`docs/00_overview.md`](docs/00_overview.md) for a map of the entire HITL module.

---

## Root Module Connection

All scripts import shared HITL infrastructure from the `hitl/` package at the project root:

```
hitl/
├── primitives.py   ← Payload builders + resume parser
│                     build_approval_payload()
│                     build_tool_payload()
│                     build_edit_payload()
│                     build_escalation_payload()
│                     parse_resume_action()
│
├── run_cycle.py    ← Invoke/pause/resume helpers
│                     run_hitl_cycle()           (single interrupt)
│                     run_multi_interrupt_cycle() (N sequential interrupts)
│                     display_interrupt_payload() (console output)
│
└── review_nodes.py ← Factory functions for reusable review nodes
                      create_approval_node()
                      create_tool_confirm_node()
                      create_edit_node()
                      create_escalation_node()
```

The scripts in **this folder** teach the **HITL patterns**. The `hitl/` package provides the standardised payload shapes, resume parsing, and run-cycle helpers they depend on.

For deeper conceptual background (why HITL, checkpointing theory, design decisions, gotchas), see the course-level docs at [`docs/hitl/`](../../docs/hitl/):

- [`docs/hitl/01_big_picture.md`](../../docs/hitl/01_big_picture.md) — Why HITL exists; analogy; where it fits in the MAS.
- [`docs/hitl/02_core_concepts.md`](../../docs/hitl/02_core_concepts.md) — Checkpointing, `thread_id`, `InterruptPayload`, `parse_resume_action`.
- [`docs/hitl/03_deep_dive.md`](../../docs/hitl/03_deep_dive.md) — Walkthrough of all three `hitl/` modules.
- [`docs/hitl/05_gotchas.md`](../../docs/hitl/05_gotchas.md) — Common bugs (double execution, missing checkpointer, thread ID collisions).

---

## Prerequisites

Before studying this area, ensure you understand:

1. **LangGraph StateGraph basics** — defining nodes and wiring a graph with `add_edge` and `add_conditional_edges` (`scripts/handoff/`).
2. **Tool binding and ToolNode** — how `llm.bind_tools()` and `ToolNode` execute tool calls (`scripts/tools/`).
3. **Python TypedDict** — used as the graph state schema throughout.

---

## How to Run Any Script

```bash
# From the project root:
cd "D:/Agentic AI/LangGraph_Multi_Agent_System"

# Pattern A — Basic Approval (simulated agent, no LLM required):
python -m scripts.HITL.basic_approval

# Pattern B — Tool Call Confirmation (LLM required):
python -m scripts.HITL.tool_call_confirmation

# Pattern C — Edit Before Approve (LLM required):
python -m scripts.HITL.edit_before_approve

# Pattern D — Multi-Step Approval (LLM required):
python -m scripts.HITL.multi_step_approval

# Pattern E — Escalation Chain (LLM required):
python -m scripts.HITL.escalation_chain
```

> **NOTE:** Patterns B–E make real LLM calls. Set `GOOGLE_API_KEY` or `OPENAI_API_KEY` in a `.env` file at the project root before running them. Pattern A uses a fixed simulated response and does not require an API key.

---

## Relationship to Other Areas

HITL is Layer 4 of the overall system. It sits after guardrails (which enforce safety automatically) and before orchestration (which coordinates many agents at once):

```
Area 1 — tools/          ← What agents can do
Area 2 — handoff/        ← How agents hand work to each other
Area 3 — guardrails/     ← Automated safety checks
Area 4 — HITL/           ← Human intervention when automation is not enough  ← YOU ARE HERE
Area 5+ — orchestration, memory, observability, ...
```
