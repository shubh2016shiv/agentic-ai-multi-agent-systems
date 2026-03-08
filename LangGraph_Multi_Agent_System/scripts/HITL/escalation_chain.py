#!/usr/bin/env python3
"""
============================================================
Escalation Chain
============================================================
Pattern E: Tiered escalation — if a junior reviewer is
uncertain, the case is escalated to a senior reviewer.
Multiple interrupt points with different reviewer roles.
Prerequisite: multi_step_approval.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
In multi_step_approval.py, both review steps are at the
same authority level. In an escalation chain, the reviewers
have DIFFERENT authority levels:

    Junior reviewer (nurse):
        - Can APPROVE (deliver)
        - Can REJECT (block)
        - Can ESCALATE (pass to senior)

    Senior reviewer (attending physician):
        - Can APPROVE (deliver with authority)
        - Can REJECT (block with authority)
        - Cannot escalate further (terminal)

The junior reviewer acts as a filter:
    - Clear-cut cases are handled without bothering seniors
    - Complex or uncertain cases get escalated
    - This reduces the load on senior reviewers

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [agent]              <-- real LLM call
       |
       v
    [junior_review]      <-- interrupt #1: approve / escalate / reject
       |
    route_after_junior()
       |
    +--+----------+----------+
    |              |          |
    | "deliver"    | "senior" | "reject"
    v              v          v
    [deliver]   [senior_review] [reject]
    |              |          |
    |         interrupt #2    |
    |              |          |
    |    +----+----+----+     |
    |    |              |     |
    |  "deliver"     "reject" |
    |    |              |     |
    v    v              v     v
    [END][END]        [END] [END]

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. Tiered reviewer roles with different permissions
    2. Escalation as a routing decision (not just pass/fail)
    3. Reducing senior reviewer load via junior filter
    4. Conditional interrupt — not every path hits every interrupt

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.HITL.escalation_chain
============================================================
"""

# ── Standard library ────────────────────────────────────────────────────────
import sys
import json
from typing import TypedDict, Literal, Annotated

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

# CONNECTION: escalation_chain uses hitl.primitives for the tiered escalation
# payloads and resume parsing. build_escalation_payload() creates payloads
# with a reviewer_role field so that review UIs can show who is being asked.
# parse_resume_action() normalises the dict resume {"action": ..., "note": ...}.
# See hitl/primitives.py for the escalation payload concept explanation.
from hitl.primitives import build_escalation_payload, parse_resume_action
from hitl.run_cycle import run_multi_interrupt_cycle


# ============================================================
# STAGE 5.1 — State Definition
# ============================================================

class EscalationState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_case: dict
    agent_response: str        # Written by: agent_node
    junior_decision: str       # Written by: junior_review ("approve" | "escalate" | "reject")
    junior_note: str           # Written by: junior_review (reason for decision)
    senior_decision: str       # Written by: senior_review ("approve" | "reject")
    senior_note: str           # Written by: senior_review (reason for decision)
    final_output: str          # Written by: terminal nodes
    status: str                # "delivered" | "delivered_senior" | "rejected" | "rejected_senior"


# ============================================================
# STAGE 5.2 — Node Definitions
# ============================================================

def agent_node(state: EscalationState) -> dict:
    """
    Clinical agent — real LLM call.

    Produces a clinical assessment that will be reviewed
    by the junior reviewer first.
    """
    llm = get_llm()
    patient = state["patient_case"]

    system = SystemMessage(content=(
        "You are a clinical triage specialist. Provide a clinical "
        "assessment with treatment recommendations. Include specific "
        "medication names and dosages. End with a disclaimer: "
        "'Consult your healthcare provider for personalised advice.'"
    ))
    prompt = HumanMessage(content=f"""Patient: {patient.get('age')}y {patient.get('sex')}
Complaint: {patient.get('chief_complaint')}
Symptoms: {', '.join(patient.get('symptoms', []))}
History: {', '.join(patient.get('medical_history', []))}
Medications: {', '.join(patient.get('current_medications', []))}
Allergies: {', '.join(patient.get('allergies', []))}
Labs: {json.dumps(patient.get('lab_results', {}))}
Vitals: {json.dumps(patient.get('vitals', {}))}""")

    config = build_callback_config(trace_name="escalation_agent")
    response = llm.invoke([system, prompt], config=config)

    print(f"    | [Agent] Assessment: {len(response.content)} chars")
    return {
        "messages": [response],
        "agent_response": response.content,
    }


def junior_review_node(state: EscalationState) -> dict:
    """
    Interrupt #1: Junior reviewer (e.g., nurse, resident).

    The junior reviewer has THREE options:
        - approve:  Clear-cut case, deliver
        - escalate: Uncertain, send to attending physician
        - reject:   Clearly incorrect, block

    The escalate option is what differentiates this from
    basic_approval. It creates a conditional path to a
    SECOND interrupt point at a higher authority level.
    """
    response = state["agent_response"]
    print(f"    | [Junior] Reviewing response ({len(response)} chars)...")

    # ── INTERRUPT #1: Junior review ──────────────────────────────────
    # CONNECTION: build_escalation_payload() from hitl.primitives creates
    # a payload with reviewer_role and the three-option set specific to
    # the junior tier. Each tier has a different options list.
    decision = interrupt(build_escalation_payload(
        response=response,
        reviewer_role="junior (resident/nurse)",
        options=["approve", "escalate", "reject"],
        note="Escalate if you are uncertain about the diagnosis or dosage.",
    ))

    # CONNECTION: parse_resume_action() normalises the dict resume value
    # {"action": "escalate", "note": "drug interaction concern"} into a
    # standard dict with "action" and "note" keys.
    parsed = parse_resume_action(decision, default_action="escalate")
    action = parsed["action"]
    note = parsed["note"]

    print(f"    | [Junior] Decision: {action.upper()}")
    if note:
        print(f"    | [Junior] Note: {note}")

    return {
        "junior_decision": action,
        "junior_note": note,
    }


def route_after_junior(state: EscalationState) -> Literal["deliver", "senior_review", "reject"]:
    """
    Route based on junior reviewer's decision.

    approve  -> deliver directly (no senior needed)
    escalate -> senior_review (needs attending approval)
    reject   -> reject (blocked at junior level)
    """
    decision = state["junior_decision"]
    if decision == "approve":
        return "deliver"
    if decision == "escalate":
        return "senior_review"
    return "reject"


def senior_review_node(state: EscalationState) -> dict:
    """
    Interrupt #2: Senior reviewer (attending physician).

    Only reached when the junior reviewer escalated.
    The senior sees:
        - The agent's original response
        - The junior's escalation note
        - The patient case details

    The senior can APPROVE or REJECT (no further escalation).
    """
    response = state["agent_response"]
    junior_note = state.get("junior_note", "No note provided")

    print(f"    | [Senior] Escalated from junior reviewer")
    print(f"    | [Senior] Junior's note: {junior_note}")
    print(f"    | [Senior] Reviewing response ({len(response)} chars)...")

    # ── INTERRUPT #2: Senior review ──────────────────────────────────
    # CONNECTION: same build_escalation_payload() pattern but with senior
    # tier options — no "escalate" option for the terminal authority.
    decision = interrupt(build_escalation_payload(
        response=response,
        reviewer_role="senior (attending physician)",
        options=["approve", "reject"],
        note="You are the terminal authority. No further escalation possible.",
    ))

    parsed = parse_resume_action(decision, default_action="reject")
    action = parsed["action"]
    note = parsed["note"]

    print(f"    | [Senior] Decision: {action.upper()}")
    if note:
        print(f"    | [Senior] Note: {note}")

    return {
        "senior_decision": action,
        "senior_note": note,
    }


def route_after_senior(state: EscalationState) -> Literal["deliver_senior", "reject_senior"]:
    """Route based on senior reviewer's decision."""
    if state["senior_decision"] == "approve":
        return "deliver_senior"
    return "reject_senior"


def deliver_node(state: EscalationState) -> dict:
    """Deliver — approved by junior reviewer (no escalation)."""
    print("    | [Deliver] Approved by junior reviewer")
    return {
        "final_output": state["agent_response"],
        "status": "delivered",
    }


def deliver_senior_node(state: EscalationState) -> dict:
    """Deliver — approved by senior reviewer after escalation."""
    junior_note = state.get("junior_note", "")
    senior_note = state.get("senior_note", "")
    output = (
        f"{state['agent_response']}\n\n"
        f"--- Review Chain ---\n"
        f"Junior reviewer: ESCALATED"
    )
    if junior_note:
        output += f" ({junior_note})"
    output += f"\nSenior reviewer: APPROVED"
    if senior_note:
        output += f" ({senior_note})"

    print("    | [Deliver] Approved by senior reviewer (after escalation)")
    return {"final_output": output, "status": "delivered_senior"}


def reject_node(state: EscalationState) -> dict:
    """Reject — blocked by junior reviewer."""
    note = state.get("junior_note", "No reason provided")
    print(f"    | [Reject] Blocked by junior reviewer: {note}")
    return {
        "final_output": (
            "This recommendation was REJECTED by the junior reviewer.\n"
            f"Reason: {note}\n"
            "The patient should be reassessed."
        ),
        "status": "rejected",
    }


def reject_senior_node(state: EscalationState) -> dict:
    """Reject — blocked by senior reviewer after escalation."""
    junior_note = state.get("junior_note", "")
    senior_note = state.get("senior_note", "No reason provided")
    print(f"    | [Reject] Blocked by senior reviewer: {senior_note}")
    return {
        "final_output": (
            "This recommendation was REJECTED by the senior reviewer.\n"
            f"Junior escalation reason: {junior_note}\n"
            f"Senior rejection reason: {senior_note}\n"
            "The patient should be reassessed with additional clinical input."
        ),
        "status": "rejected_senior",
    }


# ============================================================
# STAGE 5.3 — Graph Construction
# ============================================================

def build_escalation_graph():
    """
    Build the escalation chain graph.

    Junior reviewer: approve / escalate / reject
    Senior reviewer: approve / reject (only on escalation)
    """
    workflow = StateGraph(EscalationState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("junior_review", junior_review_node)
    workflow.add_node("senior_review", senior_review_node)
    workflow.add_node("deliver", deliver_node)
    workflow.add_node("deliver_senior", deliver_senior_node)
    workflow.add_node("reject", reject_node)
    workflow.add_node("reject_senior", reject_senior_node)

    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", "junior_review")
    workflow.add_conditional_edges(
        "junior_review",
        route_after_junior,
        {"deliver": "deliver", "senior_review": "senior_review", "reject": "reject"},
    )
    workflow.add_conditional_edges(
        "senior_review",
        route_after_senior,
        {"deliver_senior": "deliver_senior", "reject_senior": "reject_senior"},
    )
    workflow.add_edge("deliver", END)
    workflow.add_edge("deliver_senior", END)
    workflow.add_edge("reject", END)
    workflow.add_edge("reject_senior", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ============================================================
# STAGE 5.4 — Test Cases
# ============================================================

TEST_PATIENT = PatientCase(
    patient_id="PT-EC-001",
    age=71, sex="F",
    chief_complaint="Dizziness with elevated potassium",
    symptoms=["dizziness", "fatigue", "ankle edema"],
    medical_history=["CKD Stage 3a", "Hypertension", "CHF"],
    current_medications=["Lisinopril 20mg", "Spironolactone 25mg", "Furosemide 40mg"],
    allergies=["Sulfa drugs"],
    lab_results={"K+": "5.4 mEq/L", "eGFR": "42 mL/min", "BNP": "450 pg/mL"},
    vitals={"BP": "105/65", "HR": "58"},
)


def make_state() -> EscalationState:
    return {
        "messages": [],
        "patient_case": TEST_PATIENT.model_dump(),
        "agent_response": "",
        "junior_decision": "",
        "junior_note": "",
        "senior_decision": "",
        "senior_note": "",
        "final_output": "",
        "status": "pending",
    }


def main() -> None:
    print("\n" + "=" * 70)
    print("  ESCALATION CHAIN")
    print("  Pattern: tiered reviewer escalation")
    print("=" * 70)

    print("""
    Two reviewer tiers:

      [agent] -> [junior_review]
                      |
            +---------+-----------+
            |         |           |
         approve   escalate    reject
            |         |           |
        [deliver]  [senior_review] [reject]
                      |                |
              +-------+-------+        |
              |               |        |
           approve          reject     |
              |               |        |
        [deliver_senior] [reject_senior]|
              |               |        |
            [END]           [END]    [END]

    Junior reviewer acts as a filter:
      - Clear cases: handled without senior involvement
      - Uncertain: escalated to attending physician
      - Wrong: rejected immediately
    """)

    graph = build_escalation_graph()
    print("    Graph compiled with MemorySaver.\n")

    # ── Test 1: Junior approves (no escalation) ───────────────────────
    print("=" * 70)
    print("  TEST 1: Junior APPROVES (no escalation needed)")
    print("=" * 70)

    # CONNECTION: run_multi_interrupt_cycle() handles sequential interrupts.
    # Test 1 has 1 interrupt (junior approves → no senior needed).
    # The sequence list has 1 item; the cycle exits cleanly after junior.
    r1 = run_multi_interrupt_cycle(
        graph=graph,
        thread_id="esc-junior-approve-001",
        initial_state=make_state(),
        resume_sequence=[{"action": "approve", "note": "Assessment looks thorough."}],
        verbose=True,
    )
    print(f"\n    STATUS: {r1['status'].upper()}")
    print(f"    Interrupt count: 1 (junior only)")

    # ── Test 2: Junior escalates, senior approves ─────────────────────
    print("\n\n" + "=" * 70)
    print("  TEST 2: Junior ESCALATES -> Senior APPROVES")
    print("=" * 70)

    # Test 2 has 2 interrupts (junior escalates → senior approves).
    # The sequence list has 2 items — one per interrupt point.
    r2 = run_multi_interrupt_cycle(
        graph=graph,
        thread_id="esc-escalate-approve-001",
        initial_state=make_state(),
        resume_sequence=[
            {"action": "escalate", "note": "Drug interaction between Lisinopril and Spironolactone needs attending review."},
            {"action": "approve", "note": "Agree with assessment. Monitor K+ closely."},
        ],
        verbose=True,
    )
    print(f"\n    STATUS: {r2['status'].upper()}")
    print(f"    Interrupt count: 2 (junior + senior)")

    # ── Test 3: Junior rejects (no escalation) ────────────────────────
    print("\n\n" + "=" * 70)
    print("  TEST 3: Junior REJECTS (no escalation)")
    print("=" * 70)

    r3 = run_multi_interrupt_cycle(
        graph=graph,
        thread_id="esc-junior-reject-001",
        initial_state=make_state(),
        resume_sequence=[{"action": "reject", "note": "Assessment missing critical drug interaction warning."}],
        verbose=True,
    )
    print(f"\n    STATUS: {r3['status'].upper()}")
    print(f"    Interrupt count: 1 (junior only)")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  ESCALATION CHAIN COMPLETE")
    print("=" * 70)
    print(f"""
    Results:
      Test 1 (junior approves):    {r1['status'].upper()}
      Test 2 (escalate -> senior): {r2['status'].upper()}
      Test 3 (junior rejects):     {r3['status'].upper()}

    Escalation benefits:
      - Junior handles clear cases (reduces senior load)
      - Complex cases reach the right authority level
      - Rejection at any tier stops the pipeline immediately
      - The review chain is recorded in state (audit trail)

    How this differs from multi_step_approval:
      Multi-step   = sequential checkpoints at same authority
      Escalation   = conditional routing to higher authority
      Key delta    = the "escalate" option and tiered roles

    This completes the HITL pattern series:
      A. basic_approval          — interrupt/resume fundamentals
      B. tool_call_confirmation  — approve/reject tool calls
      C. edit_before_approve     — modify response before delivery
      D. multi_step_approval     — multiple interrupt points
      E. escalation_chain        — tiered reviewer escalation
    """)


if __name__ == "__main__":
    main()
