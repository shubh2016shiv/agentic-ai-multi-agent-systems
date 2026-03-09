"""
HITL Run Cycle Helpers
=======================
Utilities for executing the complete interrupt/resume lifecycle in
educational demos and test harnesses.

Where This Fits in the MAS Architecture
-----------------------------------------
Every HITL pattern follows the same three-step cycle:

    1. INVOKE: graph.invoke(initial_state, config)
       ─ Graph runs until it hits interrupt()
       ─ Returns result with result["__interrupt__"] payload
       ─ Graph is FROZEN at the interrupt point

    2. PAUSE: Human reads the interrupt payload and makes a decision
       ─ In production: the payload is displayed in a UI, ticket, or CLI
       ─ In tests/demos: the decision is passed programmatically

    3. RESUME: graph.invoke(Command(resume=decision), config)
       ─ Same thread_id — MUST match the paused execution
       ─ Node restarts from line 1; interrupt() returns decision immediately
       ─ Pipeline continues to completion

This module provides:
    run_hitl_cycle()         — runs the full invoke → pause → resume cycle
    run_multi_interrupt_cycle() — runs N interrupts sequentially (same thread)
    display_interrupt_payload() — pretty-prints the interrupt payload

These helpers eliminate repetitive boilerplate from pattern scripts.
Scripts import them to keep the demo code focused on the pattern, not
the mechanics of checking result["__interrupt__"].

Usage:
    from hitl.run_cycle import run_hitl_cycle, display_interrupt_payload
    from langgraph.types import Command

    final = run_hitl_cycle(
        graph=graph,
        thread_id="demo-001",
        initial_state=make_state(),
        resume_value=True,           # or {"action": "approve"}, "escalate", etc.
        verbose=True,
    )
"""

from typing import Any
from langgraph.types import Command


# ============================================================
# Single-Interrupt Cycle
# ============================================================

def run_hitl_cycle(
    graph: Any,
    thread_id: str,
    initial_state: dict,
    resume_value: Any,
    verbose: bool = True,
) -> dict:
    """
    Run a complete single-interrupt HITL cycle.

    Concept — The mechanics in detail:
        CALL 1: graph.invoke(initial_state, config)
            → Graph executes nodes until interrupt() is called
            → interrupt() serialises state to checkpointer
            → invoke() returns {"__interrupt__": [payload], ...other state...}
            → Graph is FROZEN

        CALL 2: graph.invoke(Command(resume=resume_value), config)
            → LangGraph loads the frozen state from the checkpointer
            → The interrupted node RESTARTS from line 1
            → interrupt() detects the waiting Command and returns resume_value
            → Code after interrupt() executes
            → Graph completes normally

        The SAME config (thread_id) MUST be used for both calls.
        Different thread_ids = completely independent executions.

    Args:
        graph: Compiled LangGraph graph (must have a checkpointer).
        thread_id: Unique ID for this execution. Used as the state key.
        initial_state: Initial state dict to pass to the first invoke().
        resume_value: The human's decision. Can be bool, str, or dict.
            bool   → Pattern A (approve=True, reject=False)
            str    → "approve" | "reject" | "escalate" | "execute" | "skip"
            dict   → {"action": "edit", "content": "..."} or similar
        verbose: If True, print progress and interrupt payloads.

    Returns:
        Final state dict after the graph completes.

    Raises:
        RuntimeError: If the graph does not interrupt as expected.
    """
    config = {"configurable": {"thread_id": thread_id}}

    if verbose:
        print(f"\n    CALL 1: graph.invoke(initial_state, thread_id='{thread_id}')")

    result = graph.invoke(initial_state, config=config)

    if "__interrupt__" in result and result["__interrupt__"]:
        if verbose:
            print(f"\n    Graph PAUSED at interrupt point.")
            display_interrupt_payload(result["__interrupt__"])
            _print_resume_decision(resume_value)
            print(f"\n    CALL 2: graph.invoke(Command(resume=...), config)")

        final = graph.invoke(Command(resume=resume_value), config=config)
        return final

    if verbose:
        print(f"    Graph completed without interrupt (no interrupt node reached).")
    return result


# ============================================================
# Multi-Interrupt Cycle
# ============================================================

def run_multi_interrupt_cycle(
    graph: Any,
    thread_id: str,
    initial_state: dict,
    resume_sequence: list[Any],
    verbose: bool = True,
) -> dict:
    """
    Run a HITL cycle with multiple sequential interrupt points.

    Concept — Sequential interrupts on the same thread:
        Patterns D (multi_step_approval) and E (escalation_chain) have
        multiple interrupt points in one graph. Each resume call targets
        the NEXT interrupt in sequence.

        The thread_id stays the SAME across all calls. LangGraph uses the
        checkpointed state to know where the graph is paused.

        Example (Pattern D — 2 interrupts):
            Call 1: invoke(state)          → PAUSED at interrupt #1
            Call 2: invoke(Command(True))  → PAUSED at interrupt #2
            Call 3: invoke(Command(True))  → COMPLETED

        This function handles that sequence automatically given a list
        of resume values.

    Args:
        graph: Compiled LangGraph graph with checkpointer.
        thread_id: Unique execution ID (shared across all calls).
        initial_state: Initial state dict.
        resume_sequence: List of resume values, one per interrupt point.
            e.g., [True, True]          — approve both interrupts
            e.g., [True, False]         — approve first, reject second
            e.g., [{"action": "approve"}, {"action": "reject", "reason": "x"}]
        verbose: If True, print progress at each step.

    Returns:
        Final state dict after the last resume (or early exit if graph
        completes before all resume values are consumed).
    """
    config = {"configurable": {"thread_id": thread_id}}
    total = len(resume_sequence)

    if verbose:
        print(f"\n    CALL 1: Start pipeline (thread_id='{thread_id}')")

    result = graph.invoke(initial_state, config=config)

    for i, resume_value in enumerate(resume_sequence, start=1):
        if "__interrupt__" not in result or not result["__interrupt__"]:
            if verbose:
                print(f"\n    Graph completed after {i - 1} interrupt(s) (before all resume values consumed).")
            return result

        if verbose:
            print(f"\n    PAUSED at interrupt #{i} of {total}")
            display_interrupt_payload(result["__interrupt__"])
            _print_resume_decision(resume_value)
            print(f"\n    CALL {i + 1}: graph.invoke(Command(resume=...), config)")

        result = graph.invoke(Command(resume=resume_value), config=config)

    return result


# ============================================================
# Display Helpers
# ============================================================

def display_interrupt_payload(interrupts: list) -> None:
    """
    Pretty-print the interrupt payload(s) from result["__interrupt__"].

    Concept — What __interrupt__ contains:
        result["__interrupt__"] is a list of Interrupt objects (one per
        interrupt() call that fired). Each has a .value attribute with
        the payload dict passed to interrupt(payload).

        In practice, most patterns have a single interrupt per invoke(),
        so the list has one item. Multi-interrupt patterns (Patterns D/E)
        fire one interrupt per invoke() call.

    Args:
        interrupts: The list from result["__interrupt__"].
    """
    for item in interrupts:
        value = item.value if hasattr(item, "value") else item
        if isinstance(value, dict):
            print(f"\n    Interrupt payload:")
            for key, val in value.items():
                val_str = str(val)
                if len(val_str) > 120:
                    val_str = val_str[:120] + "..."
                print(f"      {key}: {val_str}")
        else:
            print(f"\n    Interrupt value: {str(value)[:200]}")


def _print_resume_decision(resume_value: Any) -> None:
    """Print the human's decision in a readable format."""
    if isinstance(resume_value, bool):
        decision = "APPROVE" if resume_value else "REJECT"
    elif isinstance(resume_value, str):
        decision = resume_value.upper()
    elif isinstance(resume_value, dict):
        action = resume_value.get("action", "unknown").upper()
        extras = {k: v for k, v in resume_value.items() if k != "action"}
        decision = f"{action} ({extras})" if extras else action
    else:
        decision = str(resume_value)

    print(f"\n    -- Human decides: {decision} --")
