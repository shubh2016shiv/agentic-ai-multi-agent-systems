#!/usr/bin/env python3
"""
============================================================
Basic Approval
============================================================
Pattern A: The fundamental interrupt/resume cycle.
Agent produces output -> interrupt pauses -> human approves
or rejects -> pipeline completes.

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Understand the core mechanics of LangGraph's Human-in-the-Loop:

    interrupt(payload)    — pauses the graph, saves state
    Command(resume=value) — resumes the graph, provides input

These two primitives enable real pipeline pauses. The graph
physically stops executing. State is saved to a checkpointer
(MemorySaver). Nothing runs until the human provides input.

This is NOT the same as appending a "REQUIRES REVIEW" text
banner (which is cosmetic — the pipeline continues running).

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [agent]            <-- simulated agent (focus on HITL mechanics)
       |
       v
    [review]           <-- interrupt(payload) PAUSES HERE
       |
       |   <- graph is frozen, state saved to MemorySaver ->
       |
       |   <- Command(resume=True/False) resumes ->
       |
       +-- approved=True  -> status="approved", deliver
       +-- approved=False -> status="rejected", safe fallback
       |
       v
    [END]

------------------------------------------------------------
KEY CONCEPTS
------------------------------------------------------------
    1. MemorySaver — required for interrupt(). Without a
       checkpointer, there is nowhere to save state.
    2. thread_id — identifies the paused execution. Different
       thread_ids are independent runs.
    3. Node restart — the review node RESTARTS from line 1
       on resume. Put idempotent code before interrupt(),
       decision code after it.
    4. __interrupt__ — the invoke() return contains the
       interrupt payload in result["__interrupt__"].

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.HITL.basic_approval
============================================================
"""

# ── Standard library ────────────────────────────────────────────────────────
import sys
from typing import TypedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── LangGraph ───────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

# CONNECTION: The HITL root module (hitl/) provides reusable primitives for
# all interrupt/resume patterns. This script imports:
#   build_approval_payload() — standardised interrupt payload builder
#   parse_resume_action()    — normalise bool/str/dict resume values to a dict
#   run_hitl_cycle()         — encapsulates the invoke → pause → resume cycle
# See hitl/primitives.py and hitl/run_cycle.py for concept explanations.
from hitl.primitives import build_approval_payload, parse_resume_action
from hitl.run_cycle import run_hitl_cycle, display_interrupt_payload


# ============================================================
# STAGE 1.1 — State Definition
# ============================================================

class ApprovalState(TypedDict):
    query: str                 # Set at invocation
    agent_response: str        # Written by: agent_node
    decision: str              # Written by: review_node ("approved" | "rejected")
    final_output: str          # Written by: review_node


# ============================================================
# STAGE 1.2 — Node Definitions
# ============================================================

def agent_node(state: ApprovalState) -> dict:
    """
    Simulated clinical agent.

    This demo uses a fixed response to keep the focus on
    interrupt() mechanics, not LLM behaviour. In production,
    this would call llm.invoke().
    """
    response = (
        "Based on elevated BNP (650 pg/mL) and progressive dyspnea, "
        "this patient likely has early-stage heart failure (NYHA Class II). "
        "Recommend initiating ACEi + beta-blocker + SGLT2i per 2024 guidelines. "
        "Echocardiogram needed to confirm ejection fraction."
    )
    print(f"    | [Agent] Produced assessment ({len(response)} chars)")
    return {"agent_response": response}


def review_node(state: ApprovalState) -> dict:
    """
    Pause execution for human review using interrupt().

    EXECUTION TRACE:
      1st call (graph.invoke):
        - Code runs until interrupt()
        - interrupt() saves state to MemorySaver
        - interrupt() surfaces payload to caller
        - GRAPH STOPS HERE. Nothing below runs.
        - Caller gets result["__interrupt__"] with the payload

      2nd call (graph.invoke with Command(resume=...)):
        - Node RESTARTS from line 1
        - interrupt() is reached again
        - Command(resume=True/False) is waiting
        - interrupt() returns True or False IMMEDIATELY
        - Code below interrupt() now runs
        - Graph continues to END
    """
    # Idempotent setup code — runs both on first call and resume
    response = state["agent_response"]
    print(f"    | [Review] Response preview: {response[:80]}...")

    # ── interrupt() PAUSES HERE on first call ────────────────────────
    # CONNECTION: build_approval_payload() from hitl.primitives creates the
    # standardised InterruptPayload dict. All HITL patterns use this shape
    # so that review UIs can render any interrupt without special-casing.
    approved = interrupt(build_approval_payload(
        response=response,
        question="Do you approve this clinical recommendation?",
        note="Respond with Command(resume=True) to approve or Command(resume=False) to reject.",
    ))

    # ── This code runs ONLY after Command(resume=...) ────────────────
    if approved:
        print("    | [Review] Human APPROVED")
        return {
            "decision": "approved",
            "final_output": response,
        }
    else:
        print("    | [Review] Human REJECTED")
        return {
            "decision": "rejected",
            "final_output": (
                "This recommendation was REJECTED by the reviewer.\n"
                "The patient should be seen for direct clinical evaluation."
            ),
        }


# ============================================================
# STAGE 1.3 — Graph Construction
# ============================================================

def build_approval_graph():
    """
    Build the basic approval graph with MemorySaver.

    MemorySaver is REQUIRED for interrupt(). Without it,
    there is nowhere to save state when the graph pauses.

    Returns the compiled graph.
    """
    workflow = StateGraph(ApprovalState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("review", review_node)

    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", "review")
    workflow.add_edge("review", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ============================================================
# STAGE 1.4 — Helper: Run a Complete HITL Cycle
# ============================================================
# CONNECTION: run_hitl_cycle() from hitl.run_cycle encapsulates the full
# invoke → check __interrupt__ → resume flow. This replaces the inline
# run_approval_cycle() function that would otherwise repeat across all
# pattern scripts.
#
# run_hitl_cycle() accepts any resume_value type (bool, str, dict) so it
# works for Pattern A through E. See hitl/run_cycle.py for the
# implementation and concept explanation of the two-call cycle.

def run_approval_cycle(graph, thread_id: str, human_approves: bool) -> dict:
    """
    Run a complete HITL approval cycle using the root module helper.

    This is a thin wrapper that provides the Pattern A-specific initial
    state and delegates the invoke/pause/resume mechanics to run_hitl_cycle().

    Args:
        graph: The compiled graph with MemorySaver.
        thread_id: Unique ID for this execution (state key).
        human_approves: The human's decision (True=approve, False=reject).

    Returns:
        The final state dict after resume.
    """
    initial_state = {
        "query": "67M with BNP 650, progressive dyspnea, ankle edema",
        "agent_response": "",
        "decision": "pending",
        "final_output": "",
    }

    # CONNECTION: run_hitl_cycle() handles the two-call sequence:
    #   Call 1 → graph.invoke(initial_state, config)  → PAUSED
    #   Call 2 → graph.invoke(Command(resume=bool), config) → DONE
    # It also calls display_interrupt_payload() to print the payload.
    final = run_hitl_cycle(
        graph=graph,
        thread_id=thread_id,
        initial_state=initial_state,
        resume_value=human_approves,
        verbose=True,
    )

    print(f"\n    Decision : {final['decision'].upper()}")
    print(f"    Output   : {final['final_output'][:120]}...")
    return final


# ============================================================
# STAGE 1.5 — Main
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  BASIC APPROVAL")
    print("  Pattern: interrupt() / Command(resume=bool)")
    print("=" * 70)

    print("""
    How interrupt() works:

      graph.invoke(state, config)     <- CALL 1: starts pipeline
          |
          v
       [agent] -> [review]
                     |
                  interrupt(payload)  <- PAUSES here
                     |
                  state saved to MemorySaver
                     |
                  returns __interrupt__ to caller
                     |
                     ...time passes...
                     |
      graph.invoke(Command(resume=True), config)  <- CALL 2
                     |
                  review node RESTARTS from line 1
                  interrupt() returns True immediately
                  decision code runs
                     |
                     v
                   [END]

    CRITICAL RULE: Nodes restart from line 1 on resume.
      Put idempotent setup code BEFORE interrupt().
      Put decision-dependent code AFTER interrupt().
    """)

    graph = build_approval_graph()
    print("    Graph compiled with MemorySaver.\n")

    # ── Test 1: Approval ──────────────────────────────────────────────
    print("=" * 70)
    print("  TEST 1: Human APPROVES")
    print("=" * 70)
    r1 = run_approval_cycle(graph, thread_id="approval-001", human_approves=True)

    # ── Test 2: Rejection ─────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  TEST 2: Human REJECTS")
    print("=" * 70)
    r2 = run_approval_cycle(graph, thread_id="approval-002", human_approves=False)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  BASIC APPROVAL COMPLETE")
    print("=" * 70)
    print(f"""
    Results:
      Test 1 (approve): {r1['decision'].upper()}
      Test 2 (reject):  {r2['decision'].upper()}

    What you saw:
      - interrupt() STOPS the graph. Not a text banner.
      - MemorySaver holds frozen state (keyed by thread_id).
      - Command(resume=True/False) resumes from the pause point.
      - Different thread_ids = completely independent runs.

    Key rule: Nodes RESTART from line 1 on resume.
      Idempotent code before interrupt().
      Decision code after interrupt().

    Next: tool_call_confirmation.py — approve/reject tool calls.
    """)


if __name__ == "__main__":
    main()
