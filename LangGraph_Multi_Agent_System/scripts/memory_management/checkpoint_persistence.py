#!/usr/bin/env python3
"""
============================================================
Checkpoint Persistence
============================================================
Pattern 2: MemorySaver for cross-invocation state persistence.
A pipeline can be stopped and resumed across multiple invoke()
calls using the same thread_id.
Prerequisite: working_memory_scratchpad.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
LangGraph checkpointers persist the graph's state after each
node execution. This enables:

    1. Resume after interrupt (HITL pattern)
    2. Multi-turn conversations (stateful agents)
    3. Fault recovery (restart from last checkpoint)
    4. State inspection (debugging, auditing)

MemorySaver is an IN-MEMORY checkpointer — fast, zero-config,
but state is lost when the process exits. For production:

    MemorySaver  → development/testing
    SqliteSaver  → single-machine persistence
    PostgresSaver → multi-machine persistence
    RedisSaver   → high-performance, distributed

All follow the SAME API. Swap the checkpointer, keep the code.

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [intake]        <-- records patient and sets initial state
       |
       v
    [assess]        <-- produces assessment, writes to state
       |
       v
    [plan]          <-- produces treatment plan, writes to state
       |
       v
    [END]

    After each node, MemorySaver saves the FULL state.
    You can inspect it with graph.get_state(config).

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    main()         graph              MemorySaver
      |               |                    |
      |-- invoke(state, thread_id="001") ->|
      |               |-- intake_node ---->|
      |               |                    |-- save checkpoint
      |               |-- assess_node ---->|
      |               |                    |-- save checkpoint
      |               |-- plan_node ------>|
      |               |                    |-- save checkpoint
      |<--- result ---|                    |
      |               |                    |
      |-- get_state(config) ------------->|
      |<--- snapshot (full state) --------|  
      |               |                    |
      |-- invoke(state, thread_id="002") ->|
      |               |                    |-- INDEPENDENT state
      |<--- result ---|                    |

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. MemorySaver and thread_id for state persistence
    2. graph.get_state() for checkpoint inspection
    3. Multi-step execution with state carried across invocations
    4. Thread isolation (different thread_ids are independent)
    5. Where to swap in RedisSaver / PostgresSaver

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.memory_management.checkpoint_persistence
============================================================
"""

# ── Standard library ────────────────────────────────────────────────────────
import sys
import json
from typing import TypedDict, Annotated

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── LangGraph ───────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# CONNECTION: Checkpoint management lives in the root memory module.
# build_checkpointed_graph() compiles a workflow with a checkpointer —
#   "memory" (MemorySaver), "sqlite", "postgres", or "redis" — same API.
# inspect_checkpoint() wraps graph.get_state() with pretty-printing.
# This script shows the MECHANICS of checkpointing; the root module
# shows the PRODUCTION SWAP patterns.
# See memory/checkpoint_helpers.py for the checkpointer type explanation.
from memory.checkpoint_helpers import build_checkpointed_graph, inspect_checkpoint


# ============================================================
# STAGE 2.1 — State Definition
# ============================================================

class CheckpointState(TypedDict):
    patient_summary: str            # Written by: intake_node
    messages: Annotated[list, add_messages]  # Conversation history
    assessment: str                  # Written by: assess_node
    treatment_plan: str              # Written by: plan_node
    steps_completed: list            # Tracks which nodes ran
    turn_count: int                  # Incremented per invoke()


# ============================================================
# STAGE 2.2 — Node Definitions
# ============================================================

def intake_node(state: CheckpointState) -> dict:
    """
    Record the patient and set initial state.

    This node runs on the FIRST invoke() only. On subsequent
    turns, the graph starts from the state saved at the last
    checkpoint, so this node's output is already in state.
    """
    summary = state.get("patient_summary", "")
    if not summary:
        summary = (
            "71F with CKD Stage 3a, Hypertension, CHF. "
            "Presenting with dizziness and elevated potassium (K+ 5.4 mEq/L). "
            "Current medications: Lisinopril 20mg, Spironolactone 25mg, "
            "Furosemide 40mg, Metformin 500mg."
        )

    steps = state.get("steps_completed", [])
    steps = list(steps) + ["intake"]
    turn = state.get("turn_count", 0) + 1

    print(f"    | [Intake] Patient recorded ({len(summary)} chars)")
    print(f"    | [Intake] Turn: {turn}, Steps so far: {steps}")

    return {
        "patient_summary": summary,
        "steps_completed": steps,
        "turn_count": turn,
    }


def assess_node(state: CheckpointState) -> dict:
    """
    Produce a clinical assessment (simulated — no LLM).

    Reads the patient summary from state. In a real pipeline,
    this would call an LLM.
    """
    summary = state["patient_summary"]

    assessment = (
        "Assessment: CKD Stage 3a with hyperkalemia (K+ 5.4 mEq/L). "
        "Likely cause: ACEi + MRA combination (Lisinopril + Spironolactone). "
        "Risk of cardiac arrhythmia if K+ exceeds 5.5. "
        "Recommend holding Spironolactone and rechecking K+ in 48 hours."
    )

    steps = list(state.get("steps_completed", [])) + ["assess"]
    print(f"    | [Assess] Assessment produced ({len(assessment)} chars)")
    print(f"    | [Assess] Steps: {steps}")

    return {
        "assessment": assessment,
        "steps_completed": steps,
    }


def plan_node(state: CheckpointState) -> dict:
    """
    Produce a treatment plan (simulated — no LLM).

    Reads both patient summary and assessment from state.
    """
    plan = (
        "Treatment Plan:\n"
        "  1. HOLD Spironolactone until K+ < 5.0 mEq/L\n"
        "  2. REDUCE Lisinopril from 20mg to 10mg\n"
        "  3. Continue Furosemide 40mg daily\n"
        "  4. Recheck electrolytes (K+, Na+, Cr) in 48 hours\n"
        "  5. If K+ remains elevated: consider Sodium Polystyrene\n"
        "  6. Consult nephrology if eGFR continues to decline"
    )

    steps = list(state.get("steps_completed", [])) + ["plan"]
    print(f"    | [Plan] Treatment plan produced ({len(plan)} chars)")
    print(f"    | [Plan] Steps: {steps}")

    return {
        "treatment_plan": plan,
        "steps_completed": steps,
    }


# ============================================================
# STAGE 2.3 — Graph Construction
# ============================================================

def _build_workflow() -> StateGraph:
    """Build the shared workflow (without checkpointer) for reuse."""
    workflow = StateGraph(CheckpointState)
    workflow.add_node("intake", intake_node)
    workflow.add_node("assess", assess_node)
    workflow.add_node("plan", plan_node)
    workflow.add_edge(START, "intake")
    workflow.add_edge("intake", "assess")
    workflow.add_edge("assess", "plan")
    workflow.add_edge("plan", END)
    return workflow


def build_checkpoint_graph():
    """
    Build the graph WITH MemorySaver checkpointer.

    CONNECTION: build_checkpointed_graph() from memory.checkpoint_helpers
    compiles the workflow with a MemorySaver checkpointer (default).
    In production, swap to "sqlite", "postgres", or "redis" by passing
    checkpointer_type to build_checkpointed_graph().

    Same graph topology, different persistence — that is the key concept:
    The node code and edges are IDENTICAL with or without a checkpointer.
    The checkpointer is an operational concern, not a business logic concern.

    See memory/checkpoint_helpers.py for the production swap examples.
    """
    # CONNECTION: build_checkpointed_graph() handles the compile(checkpointer=...)
    # call and provides a single place to swap checkpointer types.
    return build_checkpointed_graph(_build_workflow(), checkpointer_type="memory")


def build_graph_without_checkpointer():
    """
    Build the SAME graph WITHOUT a checkpointer (for comparison).

    This shows that the workflow definition is IDENTICAL. Only the
    compile() step differs. Without a checkpointer:
        - graph.get_state() raises an error
        - interrupt() raises an error
        - State is NOT persisted between invoke() calls
    """
    return _build_workflow().compile()


# ============================================================
# STAGE 2.4 — Test Cases
# ============================================================

def make_initial_state() -> CheckpointState:
    return {
        "patient_summary": "",
        "messages": [],
        "assessment": "",
        "treatment_plan": "",
        "steps_completed": [],
        "turn_count": 0,
    }


def main() -> None:
    print("\n" + "=" * 70)
    print("  CHECKPOINT PERSISTENCE")
    print("  Pattern: MemorySaver for cross-invocation state")
    print("=" * 70)

    print("""
    MemorySaver saves state after EVERY node execution.
    State is keyed by thread_id — different threads are independent.

      graph.invoke(state, {"configurable": {"thread_id": "t1"}})
      graph.invoke(state, {"configurable": {"thread_id": "t2"}})
                                              ^^^^^^^^^^^^^^^^^^
                                    Different thread = different state

    After invoke(), you can inspect the saved state:
      snapshot = graph.get_state(config)
      snapshot.values["assessment"]  # → the saved assessment

    Production swap:
      MemorySaver     → in-memory (dev/test)
      SqliteSaver     → single-machine persistence
      PostgresSaver   → multi-machine persistence
      RedisSaver      → high-performance, distributed
    """)

    # ── Test 1: With checkpointer ─────────────────────────────────────
    print("=" * 70)
    print("  TEST 1: WITH MemorySaver (state persists)")
    print("=" * 70)

    graph = build_checkpoint_graph()
    config = {"configurable": {"thread_id": "checkpoint-demo-001"}}

    print(f"\n    Invoke with thread_id='checkpoint-demo-001'")
    result = graph.invoke(make_initial_state(), config=config)

    print(f"\n    Result:")
    print(f"      Steps completed: {result['steps_completed']}")
    print(f"      Turn count: {result['turn_count']}")
    print(f"      Assessment: {result['assessment'][:80]}...")

    # ── Inspect checkpoint ────────────────────────────────────────────
    print(f"\n    CHECKPOINT INSPECTION:")
    # CONNECTION: inspect_checkpoint() from memory.checkpoint_helpers wraps
    # graph.get_state(config) with pretty-printing. It returns the values dict.
    saved_state = inspect_checkpoint(graph, thread_id="checkpoint-demo-001")

    # ── Test 2: Thread isolation ──────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  TEST 2: THREAD ISOLATION")
    print("=" * 70)

    config_thread_a = {"configurable": {"thread_id": "thread-A"}}
    config_thread_b = {"configurable": {"thread_id": "thread-B"}}

    print(f"\n    Invoke with thread_id='thread-A'")
    result_thread_a = graph.invoke(make_initial_state(), config=config_thread_a)

    print(f"\n    Invoke with thread_id='thread-B'")
    result_thread_b = graph.invoke(make_initial_state(), config=config_thread_b)

    # CONNECTION: inspect_checkpoint() retrieves saved state for each thread.
    saved_a = inspect_checkpoint(graph, thread_id="thread-A", verbose=False)
    saved_b = inspect_checkpoint(graph, thread_id="thread-B", verbose=False)

    print(f"\n    Thread A steps: {saved_a.get('steps_completed')}")
    print(f"    Thread B steps: {saved_b.get('steps_completed')}")
    print(f"    Thread A and B are INDEPENDENT — modifying one does not affect the other.")

    # ── Test 3: Without checkpointer ──────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  TEST 3: WITHOUT checkpointer (state NOT persisted)")
    print("=" * 70)

    graph_no_checkpointer = build_graph_without_checkpointer()

    print(f"\n    Invoke WITHOUT checkpointer:")
    result_no_checkpointer = graph_no_checkpointer.invoke(make_initial_state())

    print(f"\n    Result:")
    print(f"      Steps completed: {result_no_checkpointer['steps_completed']}")

    print(f"\n    Can we inspect state? NO.")
    print(f"    graph.get_state() requires a checkpointer.")
    print(f"    Without one, state exists only during invoke().")
    # ── Summary ───────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  CHECKPOINT PERSISTENCE COMPLETE")
    print("=" * 70)
    print("""
    What you saw:
      1. MemorySaver saves state after every node execution
      2. graph.get_state(config) retrieves the saved state
      3. Different thread_ids are completely independent
      4. Without a checkpointer, state is not accessible after invoke()

    Production checkpointers (same API, drop-in replacement):
      SqliteSaver     — file-based, single machine
      PostgresSaver   — database-backed, multi-machine
      RedisSaver      — in-memory, sub-millisecond, distributed

    Why this matters for multi-agent systems:
      - HITL: checkpoint state before interrupt, resume from it
      - Multi-turn: carry conversation state across requests
      - Fault recovery: restart pipeline from last good checkpoint
      - Auditing: inspect what each agent produced at each step

    Next: semantic_retrieval.py — ChromaDB RAG in graph nodes.
    """)


if __name__ == "__main__":
    main()
