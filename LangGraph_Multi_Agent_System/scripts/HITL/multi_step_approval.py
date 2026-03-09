#!/usr/bin/env python3
"""
============================================================
Multi-Step Approval
============================================================
Pattern D: Multiple interrupt points in the same graph.
The pipeline pauses at two different nodes, each requiring
independent human approval.
Prerequisite: edit_before_approve.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Real-world pipelines often need MULTIPLE review checkpoints,
not just one. A clinical pipeline might require:

    Step 1: Review the diagnostic assessment
    Step 2: Review the treatment plan

Each step has its own interrupt point. Each resume targets
the NEXT interrupt in the sequence. The SAME thread_id is
used across all resume calls.

From the human's perspective:
    Call 1: Start pipeline           -> PAUSED at step 1
    Call 2: Resume with approval 1   -> PAUSED at step 2
    Call 3: Resume with approval 2   -> Pipeline completes

If the human rejects at step 1, step 2 never runs.
This saves LLM tokens and reviewer time.

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [assess]             <-- produces diagnostic assessment
       |
       v
    [review_assessment]  <-- interrupt #1: approve/reject assessment
       |
    route_after_assessment()
       |
    +--+-------------------+
    |                      |
    | "plan"               | "reject"
    v                      v
    [plan]                [reject] -> [END]
       |
       v
    [review_plan]        <-- interrupt #2: approve/reject treatment plan
       |
    route_after_plan()
       |
    +--+-------------------+
    |                      |
    | "deliver"            | "reject"
    v                      v
    [deliver]             [reject] -> [END]
       |
       v
    [END]

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. Multiple interrupt points in one graph
    2. Same thread_id across all resume calls
    3. Sequential interrupts: each resume targets the NEXT one
    4. Early rejection skips downstream nodes

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.HITL.multi_step_approval
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

# CONNECTION: multi_step_approval uses hitl.primitives for payload building
# and hitl.run_cycle for the multi-interrupt cycle helper.
# run_multi_interrupt_cycle() handles N sequential interrupts on the same
# thread_id — exactly the pattern used here (2 interrupt points).
# See hitl/run_cycle.py for the concept explanation of sequential interrupts.
from hitl.primitives import build_approval_payload
from hitl.run_cycle import run_multi_interrupt_cycle


# ============================================================
# STAGE 4.1 — State Definition
# ============================================================

class MultiStepState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_case: dict
    assessment: str             # Written by: assess_node
    assessment_approved: bool   # Written by: review_assessment_node
    treatment_plan: str         # Written by: plan_node
    plan_approved: bool         # Written by: review_plan_node
    final_output: str           # Written by: deliver_node or reject_node
    status: str                 # "delivered" | "assessment_rejected" | "plan_rejected"


# ============================================================
# STAGE 4.2 — Node Definitions
# ============================================================

def assess_node(state: MultiStepState) -> dict:
    """
    Step 1: Produce a diagnostic assessment using the LLM.

    The assessment is the FIRST deliverable that needs review.
    """
    llm = get_llm()
    patient = state["patient_case"]

    system = SystemMessage(content=(
        "You are a clinical specialist. Provide ONLY a diagnostic "
        "assessment (2-3 paragraphs). Do NOT include treatment "
        "recommendations yet. Focus on: differential diagnosis, "
        "severity assessment, and key findings."
    ))
    prompt = HumanMessage(content=f"""Patient: {patient.get('age')}y {patient.get('sex')}
Complaint: {patient.get('chief_complaint')}
Symptoms: {', '.join(patient.get('symptoms', []))}
History: {', '.join(patient.get('medical_history', []))}
Medications: {', '.join(patient.get('current_medications', []))}
Labs: {json.dumps(patient.get('lab_results', {}))}
Vitals: {json.dumps(patient.get('vitals', {}))}""")

    config = build_callback_config(trace_name="multi_step_assessment")
    response = llm.invoke([system, prompt], config=config)

    print(f"    | [Assess] Diagnostic assessment: {len(response.content)} chars")
    return {
        "messages": [response],
        "assessment": response.content,
    }


def review_assessment_node(state: MultiStepState) -> dict:
    """
    Interrupt #1: Human reviews the diagnostic assessment.

    If approved, the pipeline continues to generate a
    treatment plan. If rejected, the pipeline stops.
    """
    assessment = state["assessment"]
    print(f"    | [Review Assessment] Preview: {assessment[:80]}...")

    # ── INTERRUPT #1 ─────────────────────────────────────────────────
    # CONNECTION: build_approval_payload() from hitl.primitives creates the
    # standardised payload. The "note" field carries the context message.
    approved = interrupt(build_approval_payload(
        response=assessment,
        question="Do you approve this diagnostic assessment?",
        note="If rejected, no treatment plan will be generated.",
        options=["approve", "reject"],
    ))

    if approved:
        print("    | [Review Assessment] APPROVED")
    else:
        print("    | [Review Assessment] REJECTED")

    return {"assessment_approved": approved}


def route_after_assessment(state: MultiStepState) -> Literal["plan", "reject"]:
    """Route: approved -> generate treatment plan, rejected -> stop."""
    if state["assessment_approved"]:
        return "plan"
    return "reject"


def plan_node(state: MultiStepState) -> dict:
    """
    Step 2: Generate treatment plan based on approved assessment.

    This node ONLY runs if the assessment was approved.
    If rejected, the pipeline stops — no LLM call wasted.
    """
    llm = get_llm()
    patient = state["patient_case"]

    system = SystemMessage(content=(
        "You are a clinical specialist. Based on the approved "
        "diagnostic assessment below, provide a specific treatment "
        "plan with: medications (name, dose, frequency), monitoring "
        "schedule, and follow-up timeline."
    ))
    prompt = HumanMessage(content=f"""APPROVED ASSESSMENT:
{state['assessment']}

PATIENT:
Age: {patient.get('age')}y {patient.get('sex')}
Medications: {', '.join(patient.get('current_medications', []))}
Allergies: {', '.join(patient.get('allergies', []))}
Labs: {json.dumps(patient.get('lab_results', {}))}

Generate a specific treatment plan.""")

    config = build_callback_config(trace_name="multi_step_plan")
    response = llm.invoke([system, prompt], config=config)

    print(f"    | [Plan] Treatment plan: {len(response.content)} chars")
    return {
        "messages": [response],
        "treatment_plan": response.content,
    }


def review_plan_node(state: MultiStepState) -> dict:
    """
    Interrupt #2: Human reviews the treatment plan.

    This is the SECOND interrupt point in the same graph.
    The resume call targets this interrupt specifically.
    """
    plan = state["treatment_plan"]
    print(f"    | [Review Plan] Preview: {plan[:80]}...")

    # ── INTERRUPT #2 ─────────────────────────────────────────────────
    # CONNECTION: same build_approval_payload() pattern — consistent shape
    # across all interrupt points in this graph.
    approved = interrupt(build_approval_payload(
        response=plan,
        question="Do you approve this treatment plan?",
        note="The diagnostic assessment was already approved.",
        options=["approve", "reject"],
    ))

    if approved:
        print("    | [Review Plan] APPROVED")
    else:
        print("    | [Review Plan] REJECTED")

    return {"plan_approved": approved}


def route_after_plan(state: MultiStepState) -> Literal["deliver", "reject_plan"]:
    """Route: approved -> deliver, rejected -> reject."""
    if state["plan_approved"]:
        return "deliver"
    return "reject_plan"


def deliver_node(state: MultiStepState) -> dict:
    """Both assessment and plan approved — deliver complete recommendation."""
    full_output = (
        f"DIAGNOSTIC ASSESSMENT (approved):\n"
        f"{state['assessment']}\n\n"
        f"TREATMENT PLAN (approved):\n"
        f"{state['treatment_plan']}"
    )
    print("    | [Deliver] Full recommendation delivered")
    return {"final_output": full_output, "status": "delivered"}


def reject_node(state: MultiStepState) -> dict:
    """Assessment rejected — treatment plan was never generated."""
    print("    | [Reject] Assessment rejected — pipeline stopped early")
    return {
        "final_output": (
            "Diagnostic assessment was REJECTED by the reviewer.\n"
            "No treatment plan was generated.\n"
            "The patient should be reassessed."
        ),
        "status": "assessment_rejected",
    }


def reject_plan_node(state: MultiStepState) -> dict:
    """Assessment approved but treatment plan rejected."""
    print("    | [Reject Plan] Treatment plan rejected")
    return {
        "final_output": (
            "Diagnostic assessment was APPROVED.\n"
            "Treatment plan was REJECTED by the reviewer.\n"
            "A revised treatment plan is needed."
        ),
        "status": "plan_rejected",
    }


# ============================================================
# STAGE 4.3 — Graph Construction
# ============================================================

def build_multi_step_graph():
    """
    Build the multi-step approval graph.

    Two interrupt points:
        1. review_assessment — approve/reject diagnosis
        2. review_plan — approve/reject treatment (only if diagnosis approved)
    """
    workflow = StateGraph(MultiStepState)

    workflow.add_node("assess", assess_node)
    workflow.add_node("review_assessment", review_assessment_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("review_plan", review_plan_node)
    workflow.add_node("deliver", deliver_node)
    workflow.add_node("reject", reject_node)
    workflow.add_node("reject_plan", reject_plan_node)

    workflow.add_edge(START, "assess")
    workflow.add_edge("assess", "review_assessment")
    workflow.add_conditional_edges(
        "review_assessment",
        route_after_assessment,
        {"plan": "plan", "reject": "reject"},
    )
    workflow.add_edge("plan", "review_plan")
    workflow.add_conditional_edges(
        "review_plan",
        route_after_plan,
        {"deliver": "deliver", "reject_plan": "reject_plan"},
    )
    workflow.add_edge("deliver", END)
    workflow.add_edge("reject", END)
    workflow.add_edge("reject_plan", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ============================================================
# STAGE 4.4 — Test Cases
# ============================================================

TEST_PATIENT = PatientCase(
    patient_id="PT-MS-001",
    age=67, sex="M",
    chief_complaint="Progressive dyspnea and ankle edema",
    symptoms=["dyspnea", "ankle edema", "fatigue", "orthopnea"],
    medical_history=["Hypertension", "Type 2 Diabetes"],
    current_medications=["Lisinopril 10mg", "Metformin 500mg"],
    allergies=[],
    lab_results={"BNP": "650 pg/mL", "eGFR": "55 mL/min", "HbA1c": "7.2%"},
    vitals={"BP": "145/90", "HR": "95", "SpO2": "93%"},
)


def make_state() -> MultiStepState:
    return {
        "messages": [],
        "patient_case": TEST_PATIENT.model_dump(),
        "assessment": "",
        "assessment_approved": False,
        "treatment_plan": "",
        "plan_approved": False,
        "final_output": "",
        "status": "pending",
    }


def main() -> None:
    print("\n" + "=" * 70)
    print("  MULTI-STEP APPROVAL")
    print("  Pattern: multiple interrupt points in one graph")
    print("=" * 70)

    print("""
    Two review checkpoints in one graph:

      [assess] -> [review_assessment] -> interrupt #1
                           |
                   PAUSED (resume=True/False)
                           |
                  +--------+--------+
                  |                 |
              (approved)        (rejected)
                  |                 |
              [plan]           [reject] -> [END]
                  |
          [review_plan] -> interrupt #2
                  |
          PAUSED (resume=True/False)
                  |
         +--------+--------+
         |                  |
     (approved)        (rejected)
         |                  |
     [deliver]        [reject_plan]
         |                  |
       [END]              [END]

    Same thread_id across all resume calls.
    Each resume targets the NEXT interrupt in sequence.
    """)

    graph = build_multi_step_graph()
    print("    Graph compiled with MemorySaver.\n")

    # ── Test 1: Both approved ─────────────────────────────────────────
    print("=" * 70)
    print("  TEST 1: Both APPROVED (assessment + treatment plan)")
    print("=" * 70)

    # CONNECTION: run_multi_interrupt_cycle() from hitl.run_cycle handles
    # multiple sequential interrupts on the same thread_id automatically.
    # Resume sequence: [True, True] = approve interrupt #1 then #2.
    # Compare with basic_approval.py which uses run_hitl_cycle() for 1 interrupt.
    r1 = run_multi_interrupt_cycle(
        graph=graph,
        thread_id="multi-both-001",
        initial_state=make_state(),
        resume_sequence=[True, True],  # Approve assessment, approve plan
        verbose=True,
    )
    print(f"\n    STATUS: {r1['status'].upper()}")

    # ── Test 2: Assessment approved, plan rejected ────────────────────
    print("\n\n" + "=" * 70)
    print("  TEST 2: Assessment APPROVED, Plan REJECTED")
    print("=" * 70)

    r2 = run_multi_interrupt_cycle(
        graph=graph,
        thread_id="multi-plan-reject-001",
        initial_state=make_state(),
        resume_sequence=[True, False],  # Approve assessment, reject plan
        verbose=True,
    )
    print(f"\n    STATUS: {r2['status'].upper()}")

    # ── Test 3: Assessment rejected (plan never generated) ────────────
    print("\n\n" + "=" * 70)
    print("  TEST 3: Assessment REJECTED (plan never generated)")
    print("=" * 70)

    # Only 1 item in resume_sequence — if assessment rejected, plan never runs.
    # run_multi_interrupt_cycle() exits early when graph completes without
    # consuming all resume values.
    r3 = run_multi_interrupt_cycle(
        graph=graph,
        thread_id="multi-assess-reject-001",
        initial_state=make_state(),
        resume_sequence=[False],  # Reject assessment — plan node never runs
        verbose=True,
    )
    print(f"\n    STATUS: {r3['status'].upper()}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  MULTI-STEP APPROVAL COMPLETE")
    print("=" * 70)
    print(f"""
    Results:
      Test 1 (both approved):         {r1['status'].upper()}
      Test 2 (assessment ok, plan no): {r2['status'].upper()}
      Test 3 (assessment rejected):    {r3['status'].upper()}

    Call sequence for full approval (3 calls total):
      Call 1: graph.invoke(state, config)         -> PAUSED at #1
      Call 2: graph.invoke(Command(resume=True))  -> PAUSED at #2
      Call 3: graph.invoke(Command(resume=True))  -> DELIVERED

    Token savings from early rejection:
      Test 3 rejected at step 1 -> no treatment plan LLM call.
      Only 1 LLM call instead of 2.

    Next: escalation_chain.py — tiered reviewer escalation.
    """)


if __name__ == "__main__":
    main()
