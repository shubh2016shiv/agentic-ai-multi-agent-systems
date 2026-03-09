"""
HITL (Human-in-the-Loop) Package
===================================
Reusable primitives and utilities for introducing human decision points
into LangGraph multi-agent pipelines.

Where This Fits in the MAS Architecture
-----------------------------------------
HITL is the mechanism that bridges automated agent execution and human
judgment. It is used when:

    1. The agent's output is HIGH-STAKES (prescriptions, financial decisions)
       and must be reviewed before delivery.

    2. The agent's CONFIDENCE is low (ambiguous case, missing data).

    3. The pipeline requires MULTI-LEVEL APPROVAL (junior → senior review).

    4. TOOLS WITH SIDE EFFECTS need human authorisation before execution.

Core LangGraph HITL primitives (from langgraph.types):
    interrupt(payload)       — pauses the graph, saves state, returns payload
    Command(resume=value)    — resumes the paused graph with human decision

This package provides REUSABLE wrappers around those primitives:

Modules:
    primitives.py    — InterruptPayload TypedDict, ResumeAction Literal,
                       payload builders, parse_resume_action()
    run_cycle.py     — run_hitl_cycle(), run_multi_interrupt_cycle(),
                       display_interrupt_payload()
    review_nodes.py  — Factory functions: create_approval_node(),
                       create_edit_node(), create_tool_confirmation_node(),
                       create_escalation_node()

Pattern scripts (scripts/HITL/) demonstrate progressively complex patterns:
    Pattern A — basic_approval.py         : interrupt/resume fundamentals
    Pattern B — tool_call_confirmation.py : approve/reject tool calls
    Pattern C — edit_before_approve.py    : approve/edit/reject with rich resume
    Pattern D — multi_step_approval.py    : multiple interrupt points
    Pattern E — escalation_chain.py       : tiered reviewer escalation

Key requirement — checkpointer:
    ALL HITL patterns require a checkpointer. Without one, LangGraph
    cannot save state when interrupt() is called. Always compile with:
        workflow.compile(checkpointer=MemorySaver())

Usage:
    from hitl.primitives import build_approval_payload, parse_resume_action
    from hitl.run_cycle import run_hitl_cycle
    from hitl.review_nodes import create_approval_node
"""

# Primitives — types, builders, parsers
from hitl.primitives import (
    ResumeAction,
    InterruptPayload,
    build_approval_payload,
    build_edit_payload,
    build_tool_payload,
    build_escalation_payload,
    parse_resume_action,
)

# Run cycle helpers — for test harnesses and demos
from hitl.run_cycle import (
    run_hitl_cycle,
    run_multi_interrupt_cycle,
    display_interrupt_payload,
)

# Review node factories — for production reuse
from hitl.review_nodes import (
    create_approval_node,
    create_edit_node,
    create_tool_confirmation_node,
    create_escalation_node,
)

__all__ = [
    # Primitives
    "ResumeAction",
    "InterruptPayload",
    "build_approval_payload",
    "build_edit_payload",
    "build_tool_payload",
    "build_escalation_payload",
    "parse_resume_action",
    # Run cycle
    "run_hitl_cycle",
    "run_multi_interrupt_cycle",
    "display_interrupt_payload",
    # Factories
    "create_approval_node",
    "create_edit_node",
    "create_tool_confirmation_node",
    "create_escalation_node",
]
