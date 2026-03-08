#!/usr/bin/env python3
"""
============================================================
Edit Before Approve
============================================================
Pattern C: Human can MODIFY the agent's response before
delivery. The resume value carries edited content.
Prerequisite: tool_call_confirmation.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
In basic_approval.py, the human can only approve or reject.
In many real scenarios, the human wants to EDIT the response:

    "The diagnosis is correct, but change the dosage from
     20mg to 10mg because of the patient's renal function."

The resume value is NOT just True/False — it's a dict:

    Command(resume={"action": "approve"})
    Command(resume={"action": "edit", "content": "edited text"})
    Command(resume={"action": "reject", "reason": "..."})

This demonstrates that interrupt() returns WHATEVER value
is passed to Command(resume=value). The resume value can be
any JSON-serialisable type: bool, str, dict, list.

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [agent]              <-- real LLM call
       |
       v
    [review]             <-- interrupt(payload)
       |
       | <- PAUSED, human reviews response ->
       |
       | Command(resume={"action": "approve"})
       |    -> deliver response unchanged
       |
       | Command(resume={"action": "edit", "content": "..."})
       |    -> deliver the edited version
       |
       | Command(resume={"action": "reject", "reason": "..."})
       |    -> replace with rejection message
       |
       v
    [END]

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. Rich resume values (dict, not just bool)
    2. Three-way action from a single interrupt point
    3. Human can inject content into the pipeline
    4. interrupt() returns EXACTLY what Command(resume=...) passes

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.HITL.edit_before_approve
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
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

# ── LangChain ───────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage

# ── Project imports ─────────────────────────────────────────────────────────
from core.config import get_llm
from core.models import PatientCase
from observability.callbacks import build_callback_config

# CONNECTION: edit_before_approve uses hitl.primitives for the rich payload and
# resume parsing. build_edit_payload() creates the interrupt payload showing
# approve/edit/reject options with their resume value formats.
# parse_resume_action() normalises the dict resume to {"action", "content", "reason"}.
# See hitl/primitives.py — especially the "Rich resume values" concept note.
from hitl.primitives import build_edit_payload, parse_resume_action
from hitl.run_cycle import run_hitl_cycle


# ============================================================
# STAGE 3.1 — State Definition
# ============================================================

class EditState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_case: dict
    agent_response: str        # Written by: agent_node
    human_action: str          # Written by: review_node ("approve" | "edit" | "reject")
    human_edit: str             # Written by: review_node (edited content if action="edit")
    reject_reason: str          # Written by: review_node (reason if action="reject")
    final_output: str           # Written by: review_node
    status: str                 # Written by: review_node


# ============================================================
# STAGE 3.2 — Node Definitions
# ============================================================

def agent_node(state: EditState) -> dict:
    """
    Clinical agent — real LLM call.

    Produces a clinical assessment that the human can
    approve, edit, or reject.
    """
    llm = get_llm()
    patient = state["patient_case"]

    system = SystemMessage(content=(
        "You are a clinical triage specialist. Provide a concise "
        "clinical assessment and treatment recommendation. "
        "Include specific medication names and dosages."
    ))
    prompt = HumanMessage(content=f"""Patient: {patient.get('age')}y {patient.get('sex')}
Complaint: {patient.get('chief_complaint')}
Symptoms: {', '.join(patient.get('symptoms', []))}
History: {', '.join(patient.get('medical_history', []))}
Medications: {', '.join(patient.get('current_medications', []))}
Labs: {json.dumps(patient.get('lab_results', {}))}
Vitals: {json.dumps(patient.get('vitals', {}))}""")

    config = build_callback_config(trace_name="edit_before_approve_agent")
    response = llm.invoke([system, prompt], config=config)

    print(f"    | [Agent] Assessment: {len(response.content)} chars")
    return {
        "messages": [response],
        "agent_response": response.content,
    }


def review_node(state: EditState) -> dict:
    """
    Pause for human review with three options.

    The resume value determines the action:
        {"action": "approve"}                    -> deliver as-is
        {"action": "edit", "content": "..."}     -> deliver edited version
        {"action": "reject", "reason": "..."}    -> reject with reason
    """
    response = state["agent_response"]
    print(f"    | [Review] Response preview: {response[:100]}...")

    # ── interrupt() PAUSES HERE ──────────────────────────────────────
    # CONNECTION: build_edit_payload() from hitl.primitives creates the
    # standardised interrupt payload with approve/edit/reject options.
    # Each option shows the resume value format so the reviewer knows
    # exactly what Command(resume=...) to send.
    human_input = interrupt(build_edit_payload(
        response=response,
        question="Review this clinical recommendation. You can approve, edit, or reject.",
    ))

    # ── Process the human's decision ─────────────────────────────────
    # CONNECTION: parse_resume_action() normalises all resume value types:
    #   bool   → {"action": "approve"/"reject", ...}
    #   str    → {"action": str, ...}
    #   dict   → {"action": ..., "content": ..., "reason": ..., "note": ...}
    # interrupt() returns EXACTLY what Command(resume=value) passes.
    parsed = parse_resume_action(human_input, default_action="approve")
    action = parsed["action"]

    if action == "approve":
        print("    | [Review] Human APPROVED (no edits)")
        return {
            "human_action": "approve",
            "final_output": response,
            "status": "approved",
        }

    elif action == "edit":
        edited = parsed["content"] or response
        print(f"    | [Review] Human EDITED ({len(edited)} chars)")
        print(f"    |   Edit preview: {edited[:80]}...")
        return {
            "human_action": "edit",
            "human_edit": edited,
            "final_output": edited,
            "status": "edited",
        }

    elif action == "reject":
        reason = parsed["reason"] or "No reason provided"
        print(f"    | [Review] Human REJECTED: {reason}")
        return {
            "human_action": "reject",
            "reject_reason": reason,
            "final_output": (
                f"This recommendation was REJECTED by the reviewer.\n"
                f"Reason: {reason}\n"
                "The patient should be seen for direct clinical evaluation."
            ),
            "status": "rejected",
        }

    # Fallback
    return {
        "human_action": action,
        "final_output": response,
        "status": "approved",
    }


# ============================================================
# STAGE 3.3 — Graph Construction
# ============================================================

def build_edit_graph():
    """Build the edit-before-approve graph with MemorySaver."""
    workflow = StateGraph(EditState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("review", review_node)

    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", "review")
    workflow.add_edge("review", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ============================================================
# STAGE 3.4 — Test Cases
# ============================================================

TEST_PATIENT = PatientCase(
    patient_id="PT-EBA-001",
    age=71, sex="F",
    chief_complaint="Dizziness with elevated potassium",
    symptoms=["dizziness", "fatigue", "ankle edema"],
    medical_history=["CKD Stage 3a", "Hypertension", "CHF"],
    current_medications=["Lisinopril 20mg", "Spironolactone 25mg", "Furosemide 40mg"],
    allergies=["Sulfa drugs"],
    lab_results={"K+": "5.4 mEq/L", "eGFR": "42 mL/min", "BNP": "450 pg/mL"},
    vitals={"BP": "105/65", "HR": "58"},
)


def make_state() -> EditState:
    return {
        "messages": [],
        "patient_case": TEST_PATIENT.model_dump(),
        "agent_response": "",
        "human_action": "",
        "human_edit": "",
        "reject_reason": "",
        "final_output": "",
        "status": "pending",
    }


def run_edit_cycle(graph, thread_id: str, resume_value: dict) -> dict:
    """
    Run a complete edit-before-approve cycle.

    CONNECTION: Delegates to run_hitl_cycle() from hitl.run_cycle.
    The resume value for Pattern C is a dict with an "action" key.
    """
    # CONNECTION: run_hitl_cycle() handles the two-call invoke/resume.
    # Pattern C resume values are dicts, e.g.:
    #   {"action": "approve"}
    #   {"action": "edit", "content": "revised text"}
    #   {"action": "reject", "reason": "reason text"}
    final = run_hitl_cycle(
        graph=graph,
        thread_id=thread_id,
        initial_state=make_state(),
        resume_value=resume_value,
        verbose=True,
    )

    print(f"\n    STATUS: {final['status'].upper()}")
    print(f"    Output: {final['final_output'][:150]}...")
    return final


def main() -> None:
    print("\n" + "=" * 70)
    print("  EDIT BEFORE APPROVE")
    print("  Pattern: rich resume values (approve / edit / reject)")
    print("=" * 70)

    print("""
    The resume value is NOT just True/False.
    interrupt() returns EXACTLY what Command(resume=...) passes:

      Command(resume={"action": "approve"})
        -> interrupt() returns {"action": "approve"}
        -> deliver response unchanged

      Command(resume={"action": "edit", "content": "..."})
        -> interrupt() returns {"action": "edit", "content": "..."}
        -> deliver the EDITED version

      Command(resume={"action": "reject", "reason": "..."})
        -> interrupt() returns {"action": "reject", "reason": "..."}
        -> replace with rejection message
    """)

    graph = build_edit_graph()
    print("    Graph compiled with MemorySaver.\n")

    # ── Test 1: Approve as-is ─────────────────────────────────────────
    print("=" * 70)
    print("  TEST 1: Human APPROVES (no changes)")
    print("=" * 70)
    r1 = run_edit_cycle(
        graph, thread_id="edit-approve-001",
        resume_value={"action": "approve"},
    )

    # ── Test 2: Edit then deliver ─────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  TEST 2: Human EDITS the response")
    print("=" * 70)
    r2 = run_edit_cycle(
        graph, thread_id="edit-modify-001",
        resume_value={
            "action": "edit",
            "content": (
                "Assessment: CKD Stage 3a with hyperkalemia (K+ 5.4). "
                "REDUCE Lisinopril to 10mg due to declining renal function. "
                "HOLD Spironolactone until K+ < 5.0 mEq/L. "
                "Recheck electrolytes in 48 hours. "
                "Consult nephrology if eGFR continues to decline."
            ),
        },
    )

    # ── Test 3: Reject ────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  TEST 3: Human REJECTS the response")
    print("=" * 70)
    r3 = run_edit_cycle(
        graph, thread_id="edit-reject-001",
        resume_value={
            "action": "reject",
            "reason": "Assessment does not account for drug interaction between Lisinopril and Spironolactone causing hyperkalemia.",
        },
    )

    # ── Summary ───────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  EDIT BEFORE APPROVE COMPLETE")
    print("=" * 70)
    print(f"""
    Results:
      Test 1 (approve): {r1['status'].upper()}
      Test 2 (edit):    {r2['status'].upper()}
      Test 3 (reject):  {r3['status'].upper()}

    Key differences from basic_approval:
      1. Resume value is a dict with "action" + optional fields.
      2. Human can inject new content into the pipeline.
      3. Three actions from a single interrupt point.
      4. interrupt() returns EXACTLY what Command(resume=...) passes.

    When to use this pattern:
      - Clinical recommendations that need dosage adjustments
      - Legal documents that need clause modifications
      - Customer communications that need tone edits
      - Any output where "close but needs tweaking" is common

    Next: multi_step_approval.py — multiple interrupt points.
    """)


if __name__ == "__main__":
    main()
