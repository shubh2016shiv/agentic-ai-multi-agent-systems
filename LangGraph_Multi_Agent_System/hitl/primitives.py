"""
HITL Primitives
================
Core building blocks for Human-in-the-Loop (HITL) patterns in LangGraph.
Provides standardised types, payload builders, and resume value parsers
that are shared across all HITL pattern scripts.

Where This Fits in the MAS Architecture
-----------------------------------------
HITL is the mechanism that introduces a human decision point into an
otherwise automated agent pipeline. It is NOT just a "flag for review"
banner — it physically pauses graph execution and waits for human input.

The two LangGraph primitives that make HITL work:

    interrupt(payload)
        ─────────────
        • Pauses the graph at the CURRENT node
        • Serialises full graph state to the checkpointer (MemorySaver,
          SqliteSaver, etc.)
        • Surfaces the payload to the caller via result["__interrupt__"]
        • The graph is now FROZEN — no further nodes run

    Command(resume=value)
        ─────────────────
        • Resumes the frozen graph when passed to graph.invoke()
        • The node that called interrupt() RESTARTS from line 1
        • interrupt() returns the resume value immediately on restart
        • Code AFTER interrupt() now executes

Critical rule — Node restart:
    When a node is resumed, it RESTARTS from its first line.
    Place idempotent setup code (reading state, printing previews)
    BEFORE interrupt(). Place decision-dependent code AFTER interrupt().

    ❌ Wrong:
        def review(state):
            side_effect()     # Runs TWICE: on pause and on resume
            approved = interrupt(...)

    ✓ Correct:
        def review(state):
            preview = state["response"][:100]  # Idempotent read — OK twice
            approved = interrupt({"preview": preview})
            if approved:                        # Runs only after resume
                ...

Checkpointer requirement:
    interrupt() REQUIRES a checkpointer to save state. Without one,
    LangGraph raises an error. Always compile with:
        workflow.compile(checkpointer=MemorySaver())

Pattern scripts (scripts/HITL/) show progressively complex patterns:
    Pattern A — basic_approval.py         : bool resume value
    Pattern B — tool_call_confirmation.py : dict resume value
    Pattern C — edit_before_approve.py    : rich dict with content
    Pattern D — multi_step_approval.py    : multiple interrupt points
    Pattern E — escalation_chain.py       : tiered reviewer roles
"""

from typing import Any, Literal, TypedDict


# ============================================================
# Type Definitions
# ============================================================

# All possible actions a human reviewer can take across any HITL pattern.
# Different patterns use different subsets of these actions.
ResumeAction = Literal[
    "approve",   # Accept the response as-is (basic_approval, edit_before_approve)
    "reject",    # Block the response entirely (all patterns)
    "edit",      # Modify the response before delivery (edit_before_approve)
    "escalate",  # Pass to a higher authority (escalation_chain)
    "execute",   # Execute a proposed tool call (tool_call_confirmation)
    "skip",      # Skip a proposed tool call (tool_call_confirmation)
]


class InterruptPayload(TypedDict, total=False):
    """
    Standardised shape for interrupt() payloads.

    Concept — Why standardise payloads:
        All HITL patterns call interrupt() with a dict payload. Without
        a standard shape, the payload structure varies per script, making
        it hard to build generic reviewers (e.g., a UI that renders any
        HITL interrupt without special-casing each pattern).

        This TypedDict defines the common fields. Scripts can include any
        subset — all fields are optional (total=False).

    Fields:
        question    : The human-readable question to display to the reviewer.
        content     : The agent's output being reviewed (response, tool call, etc.).
        options     : What actions the reviewer can take, with instructions.
        step        : Name of the current review step (for multi-step patterns).
        reviewer_role : Who is expected to review (e.g., "junior", "senior").
        note        : Additional context or instructions for the reviewer.
        tool_name   : Tool name (tool_call_confirmation pattern only).
        tool_args   : Tool arguments (tool_call_confirmation pattern only).
        tool_id     : Tool call ID (tool_call_confirmation pattern only).
    """
    question: str
    content: str
    options: dict | list
    step: str
    reviewer_role: str
    note: str
    tool_name: str
    tool_args: dict
    tool_id: str


# ============================================================
# Payload Builders
# ============================================================

def build_approval_payload(
    response: str,
    question: str = "Do you approve this recommendation?",
    options: list[str] | None = None,
    note: str = "",
) -> InterruptPayload:
    """
    Build a standardised interrupt payload for an approval gate.

    Use this for basic approve/reject patterns (Pattern A and simpler
    gates in multi-step and escalation patterns).

    Args:
        response: The agent's response text to display to the reviewer.
        question: The question to ask the reviewer.
        options: List of available actions. Default: ["approve", "reject"]
        note: Additional context for the reviewer.

    Returns:
        InterruptPayload dict with question, content, options, and note.

    Example:
        approved = interrupt(build_approval_payload(
            response=state["agent_response"],
            question="Approve this clinical recommendation?",
        ))
    """
    if options is None:
        options = ["approve", "reject"]

    payload: InterruptPayload = {
        "question": question,
        "content": response,
        "options": options,
    }
    if note:
        payload["note"] = note
    return payload


def build_edit_payload(
    response: str,
    question: str = "Review this response. You can approve, edit, or reject.",
) -> InterruptPayload:
    """
    Build a standardised interrupt payload for approve/edit/reject patterns.

    Use this for Pattern C (edit_before_approve) and similar patterns where
    the reviewer can inject modified content.

    The options dict shows the resume value format for each action, so the
    reviewer (or test harness) knows exactly what Command(resume=...) to send.

    Args:
        response: The agent's response text to display.
        question: The question to ask the reviewer.

    Returns:
        InterruptPayload with options as a dict mapping action → resume format.

    Example:
        human_input = interrupt(build_edit_payload(response=state["response"]))
        action = parse_resume_action(human_input)["action"]
    """
    return {
        "question": question,
        "content": response,
        "options": {
            "approve": '{"action": "approve"}',
            "edit":    '{"action": "edit", "content": "your edited text here"}',
            "reject":  '{"action": "reject", "reason": "why you are rejecting"}',
        },
    }


def build_tool_payload(
    tool_name: str,
    tool_args: dict,
    tool_id: str = "unknown",
) -> InterruptPayload:
    """
    Build a standardised interrupt payload for tool-call confirmation.

    Use this for Pattern B (tool_call_confirmation). Shows the reviewer
    which tool the agent wants to call and with what arguments.

    Args:
        tool_name: Name of the proposed tool.
        tool_args: Arguments the agent wants to pass to the tool.
        tool_id: LangChain tool call ID (from response.tool_calls[0]["id"]).

    Returns:
        InterruptPayload with tool details and execute/skip options.

    Example:
        decision = interrupt(build_tool_payload(
            tool_name=tc["name"],
            tool_args=tc["args"],
            tool_id=tc["id"],
        ))
    """
    return {
        "question": "Approve this tool call?",
        "tool_name": tool_name,
        "tool_args": tool_args,
        "tool_id": tool_id,
        "options": ["execute", "skip"],
    }


def build_escalation_payload(
    response: str,
    reviewer_role: str,
    options: list[str],
    note: str = "",
) -> InterruptPayload:
    """
    Build a standardised interrupt payload for tiered escalation patterns.

    Use this for Pattern E (escalation_chain) where different reviewer roles
    have different available actions.

    Args:
        response: The agent's response text.
        reviewer_role: Human-readable role label (e.g., "junior (resident)").
        options: Actions available to this reviewer tier.
        note: Additional instructions for this reviewer role.

    Returns:
        InterruptPayload with reviewer_role, options, and note.

    Example:
        decision = interrupt(build_escalation_payload(
            response=state["agent_response"],
            reviewer_role="junior (resident/nurse)",
            options=["approve", "escalate", "reject"],
            note="Escalate if uncertain about dosage.",
        ))
    """
    payload: InterruptPayload = {
        "reviewer_role": reviewer_role,
        "content": response,
        "options": options,
    }
    if note:
        payload["note"] = note
    return payload


# ============================================================
# Resume Value Parsers
# ============================================================

def parse_resume_action(
    resume_value: Any,
    default_action: str = "reject",
) -> dict[str, Any]:
    """
    Parse the resume value from Command(resume=...) into a structured dict.

    Concept — Resume value types:
        Different patterns use different resume value types:
            bool   — Pattern A (basic_approval): True=approve, False=reject
            str    — Simple action string: "approve", "reject", "escalate"
            dict   — Rich resume: {"action": "edit", "content": "...", "reason": "..."}

        This function normalises all three into a standard dict with at least
        an "action" key, making downstream routing code simpler.

    Args:
        resume_value: The value returned by interrupt() after Command(resume=...).
        default_action: Action to use if the resume value is None or unrecognised.

    Returns:
        Dict with keys:
            "action"  — str: the primary action (approve/reject/edit/escalate/etc.)
            "content" — str: edited content if action="edit" (empty otherwise)
            "reason"  — str: rejection reason if action="reject" (empty otherwise)
            "note"    — str: reviewer note for audit trail (empty if not provided)
            "raw"     — the original resume_value (for debugging)

    Example:
        # Bool resume (Pattern A)
        parse_resume_action(True)
        → {"action": "approve", "content": "", "reason": "", "note": "", "raw": True}

        # Dict resume (Pattern C)
        parse_resume_action({"action": "edit", "content": "revised text"})
        → {"action": "edit", "content": "revised text", "reason": "", "note": "", ...}

        # Str resume (Pattern E junior)
        parse_resume_action("escalate")
        → {"action": "escalate", "content": "", "reason": "", "note": "", ...}
    """
    if resume_value is None:
        return {"action": default_action, "content": "", "reason": "", "note": "", "raw": None}

    # bool resume (Pattern A: basic approval)
    if isinstance(resume_value, bool):
        action = "approve" if resume_value else "reject"
        return {"action": action, "content": "", "reason": "", "note": "", "raw": resume_value}

    # str resume (simple action string)
    if isinstance(resume_value, str):
        return {"action": resume_value, "content": "", "reason": "", "note": "", "raw": resume_value}

    # dict resume (rich resume with optional fields)
    if isinstance(resume_value, dict):
        return {
            "action":  resume_value.get("action", default_action),
            "content": resume_value.get("content", ""),
            "reason":  resume_value.get("reason", ""),
            "note":    resume_value.get("note", ""),
            "raw":     resume_value,
        }

    # Fallback for unexpected types
    return {"action": default_action, "content": "", "reason": "", "note": "", "raw": resume_value}
