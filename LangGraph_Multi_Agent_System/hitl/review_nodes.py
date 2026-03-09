"""
HITL Review Node Factories
============================
Factory functions that create reusable LangGraph review nodes for common
HITL patterns. Reduces the boilerplate of writing interrupt() + resume
handling in every script.

Where This Fits in the MAS Architecture
-----------------------------------------
In LangGraph, a "review node" is a node function that:
    1. Reads the agent's output from state
    2. Calls interrupt(payload) to pause and present the output to a human
    3. Receives the human's decision via Command(resume=value)
    4. Writes the decision outcome back to state

This pattern repeats with minor variations across all HITL scripts:
    basic_approval      : bool resume → approve/reject
    edit_before_approve : dict resume → approve/edit/reject
    escalation_chain    : dict resume → approve/escalate/reject (junior)
                          dict resume → approve/reject (senior)

The factories here generate these nodes with configurable parameters so
that scripts don't need to duplicate the interrupt/resume boilerplate.

NOTE on factories vs inline nodes:
    For EDUCATIONAL use, Pattern scripts keep their nodes INLINE so you
    can see exactly what each node does. Factories are for PRODUCTION use
    where the same pattern is reused across many workflows.

Usage:
    from hitl.review_nodes import create_approval_node, create_edit_node

    # Create a simple approve/reject node that reads "agent_response" from state
    review = create_approval_node(
        state_key="agent_response",
        question="Do you approve this clinical recommendation?",
    )

    workflow.add_node("review", review)
"""

from typing import Any, Callable
from hitl.primitives import (
    build_approval_payload,
    build_edit_payload,
    build_escalation_payload,
    parse_resume_action,
)

try:
    from langgraph.types import interrupt
except ImportError:
    interrupt = None  # type: ignore


# ============================================================
# Approval Node Factory (Pattern A / D / E style)
# ============================================================

def create_approval_node(
    state_key: str = "agent_response",
    question: str = "Do you approve this recommendation?",
    approve_key: str = "decision",
    options: list[str] | None = None,
    note: str = "",
) -> Callable[[dict], dict]:
    """
    Create a graph node that pauses for boolean approve/reject.

    Concept — Factory pattern for nodes:
        Instead of writing a new function for every approval gate, use
        this factory to generate one. The returned function is a valid
        LangGraph node: it accepts state dict and returns update dict.

        The node:
            1. Reads state[state_key] to get the content to review
            2. Calls interrupt(build_approval_payload(...)) to pause
            3. Parses the resume value via parse_resume_action()
            4. Returns {approve_key: bool} for conditional routing

    Args:
        state_key: Key in state dict to read as the reviewable content.
        question: Question to display to the reviewer.
        approve_key: State key to write the boolean approval result to.
        options: Available actions. Default: ["approve", "reject"]
        note: Additional instructions for the reviewer.

    Returns:
        A node function compatible with workflow.add_node().

    Example:
        assess_review = create_approval_node(
            state_key="assessment",
            question="Approve the diagnostic assessment?",
            approve_key="assessment_approved",
        )
        workflow.add_node("review_assessment", assess_review)
        workflow.add_conditional_edges(
            "review_assessment",
            lambda s: "plan" if s["assessment_approved"] else "reject",
        )
    """
    if options is None:
        options = ["approve", "reject"]

    def approval_node(state: dict) -> dict:
        content = state.get(state_key, "")
        payload = build_approval_payload(
            response=str(content),
            question=question,
            options=options,
            note=note,
        )

        resume_value = interrupt(payload)
        parsed = parse_resume_action(resume_value, default_action="reject")
        approved = parsed["action"] == "approve"

        return {approve_key: approved}

    approval_node.__name__ = f"approval_node_for_{state_key}"
    return approval_node


# ============================================================
# Edit-Before-Approve Node Factory (Pattern C style)
# ============================================================

def create_edit_node(
    state_key: str = "agent_response",
    question: str = "Review this response. You can approve, edit, or reject.",
    output_key: str = "final_output",
    status_key: str = "status",
    action_key: str = "human_action",
) -> Callable[[dict], dict]:
    """
    Create a graph node that pauses for approve/edit/reject.

    Concept — Rich resume values:
        Pattern C extends basic approval by allowing the human to INJECT
        new content. The resume value is a dict:
            {"action": "approve"}
            {"action": "edit", "content": "modified text"}
            {"action": "reject", "reason": "why rejecting"}

        parse_resume_action() normalises all three into a standard dict
        so this factory handles all cases uniformly.

    Args:
        state_key: Key in state dict to read as the reviewable content.
        question: Question/instruction for the reviewer.
        output_key: State key to write the final output to.
        status_key: State key to write "approved"/"edited"/"rejected" to.
        action_key: State key to write the action string to.

    Returns:
        A node function compatible with workflow.add_node().
    """
    def edit_node(state: dict) -> dict:
        content = state.get(state_key, "")
        payload = build_edit_payload(response=str(content), question=question)

        resume_value = interrupt(payload)
        parsed = parse_resume_action(resume_value, default_action="approve")
        action = parsed["action"]

        if action == "approve":
            return {
                action_key: "approve",
                output_key: str(content),
                status_key: "approved",
            }
        elif action == "edit":
            edited = parsed["content"] or str(content)
            return {
                action_key: "edit",
                output_key: edited,
                status_key: "edited",
            }
        elif action == "reject":
            reason = parsed["reason"] or "No reason provided"
            return {
                action_key: "reject",
                output_key: (
                    f"This recommendation was REJECTED by the reviewer.\n"
                    f"Reason: {reason}\n"
                    "The patient should be seen for direct clinical evaluation."
                ),
                status_key: "rejected",
            }

        return {action_key: action, output_key: str(content), status_key: "approved"}

    edit_node.__name__ = f"edit_node_for_{state_key}"
    return edit_node


# ============================================================
# Tool Confirmation Node Factory (Pattern B style)
# ============================================================

def create_tool_confirmation_node(
    proposed_key: str = "proposed_tool_call",
    decision_key: str = "human_decision",
) -> Callable[[dict], dict]:
    """
    Create a graph node that pauses for tool-call approve/skip.

    Concept — Why gate tool calls:
        Tools with SIDE EFFECTS (database writes, API calls, prescriptions)
        should never execute without human approval. The agent proposes a
        tool call; the human reviews name + arguments before execution.

        The interrupt payload shows:
            tool_name — what the agent wants to call
            tool_args — arguments it wants to pass
            tool_id   — LangChain call ID

        The human responds with "execute" or "skip".

    Args:
        proposed_key: State key containing the proposed tool call dict
            with keys: "name", "args", "id".
        decision_key: State key to write the human's decision to.

    Returns:
        A node function compatible with workflow.add_node().
    """
    from hitl.primitives import build_tool_payload

    def tool_confirmation_node(state: dict) -> dict:
        proposed = state.get(proposed_key, {})

        if not proposed:
            return {decision_key: "no_tool_call"}

        payload = build_tool_payload(
            tool_name=proposed.get("name", "unknown"),
            tool_args=proposed.get("args", {}),
            tool_id=proposed.get("id", "unknown"),
        )

        resume_value = interrupt(payload)
        parsed = parse_resume_action(resume_value, default_action="skip")
        action = parsed["action"]

        return {decision_key: action}

    tool_confirmation_node.__name__ = "tool_confirmation_node"
    return tool_confirmation_node


# ============================================================
# Escalation Node Factory (Pattern E style)
# ============================================================

def create_escalation_node(
    state_key: str = "agent_response",
    reviewer_role: str = "reviewer",
    options: list[str] | None = None,
    decision_key: str = "reviewer_decision",
    note_key: str = "reviewer_note",
    instructions: str = "",
) -> Callable[[dict], dict]:
    """
    Create a graph node for a specific tier in an escalation chain.

    Concept — Tiered reviewer authority:
        Escalation chains have multiple reviewer tiers with different
        authority levels and different available actions:

        Junior (nurse/resident):  ["approve", "escalate", "reject"]
        Senior (attending):       ["approve", "reject"]

        Each tier uses a different factory call with different options.
        The "escalate" action is what drives the conditional edge to the
        senior node — only escalated cases reach the attending.

    Args:
        state_key: Key in state containing the content to review.
        reviewer_role: Human-readable role (displayed in interrupt payload).
        options: Actions available to this tier. Default: ["approve", "reject"]
        decision_key: State key to write the decision action to.
        note_key: State key to write the reviewer's note to.
        instructions: Additional instructions in the interrupt payload.

    Returns:
        A node function compatible with workflow.add_node().

    Example:
        junior_review = create_escalation_node(
            reviewer_role="junior (resident/nurse)",
            options=["approve", "escalate", "reject"],
            decision_key="junior_decision",
            note_key="junior_note",
            instructions="Escalate if uncertain about dosage or drug interactions.",
        )
        senior_review = create_escalation_node(
            reviewer_role="senior (attending physician)",
            options=["approve", "reject"],
            decision_key="senior_decision",
            note_key="senior_note",
            instructions="You are the terminal authority. No further escalation.",
        )
    """
    if options is None:
        options = ["approve", "reject"]

    def escalation_node(state: dict) -> dict:
        content = state.get(state_key, "")
        payload = build_escalation_payload(
            response=str(content),
            reviewer_role=reviewer_role,
            options=options,
            note=instructions,
        )

        resume_value = interrupt(payload)
        parsed = parse_resume_action(resume_value, default_action="reject")

        return {
            decision_key: parsed["action"],
            note_key: parsed["note"],
        }

    escalation_node.__name__ = f"escalation_node_{reviewer_role.split('(')[0].strip()}"
    return escalation_node
