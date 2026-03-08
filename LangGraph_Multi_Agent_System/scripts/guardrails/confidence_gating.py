#!/usr/bin/env python3
"""
============================================================
Confidence Gating
============================================================
Pattern C: Route agent responses based on self-assessed
confidence scores. Responses below the threshold are
escalated instead of delivered.
Prerequisite: output_validation.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
When an LLM agent includes a confidence score in its response,
use that score as a routing signal:

    High confidence (>= threshold) -> deliver to user
    Low confidence  (< threshold)  -> escalate for review

This pattern is distinct from output validation (Pattern B)
because:
    - Output validation checks CONTENT quality (prohibited terms,
      missing disclaimers).
    - Confidence gating checks the agent's SELF-ASSESSED CERTAINTY.
    - They can be stacked: content check first, then confidence gate.

This script uses a real LLM call to generate the assessment,
then extracts the confidence score for routing.

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [agent]                <-- real LLM call, extracts confidence
       |
       v
    [confidence_gate]      <-- compares confidence to threshold
       |
    route_after_gate()
       |
    +--+---------+
    |             |
    | "deliver"   | "escalate"
    v             v
    [deliver]   [escalate]
    |             |
    v             v
    [END]        [END]

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. Extracting confidence from LLM responses
    2. Threshold-based conditional routing
    3. Configurable thresholds for different deployment contexts
    4. Separation between content checks and confidence checks

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.guardrails.confidence_gating
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

# ── LangChain ───────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage

# ── Project imports ─────────────────────────────────────────────────────────
from core.config import get_llm
from core.models import PatientCase
from observability.callbacks import build_callback_config

# CONNECTION: confidence extraction logic lives in the root guardrails module.
# extract_confidence() parses "Confidence: 0.85" or "Confidence: 85%" from
# LLM response text. See guardrails/confidence_guardrails.py for the full
# implementation including the concept explanation and alternative patterns.
from guardrails.confidence_guardrails import extract_confidence


# ============================================================
# STAGE 3.1 — State Definition
# ============================================================

class ConfidenceGateState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_case: dict
    agent_response: str    # Written by: agent_node
    confidence: float      # Written by: agent_node
    threshold: float       # Set at invocation time
    gate_result: str       # Written by: confidence_gate_node ("above" | "below")
    final_output: str      # Written by: deliver_node or escalate_node
    status: str            # Written by: deliver_node or escalate_node


# ============================================================
# STAGE 3.2 — Confidence Extraction
# ============================================================
# extract_confidence() is imported from guardrails.confidence_guardrails.
#
# CONCEPT: The agent is prompted to append "Confidence: X.XX" to its
# response. extract_confidence() parses that marker using regex patterns
# that handle both decimal (0.85) and percentage (85%) formats.
#
# If no marker is found, returns 0.5 (neutral default — the threshold
# then decides whether to deliver or escalate).
#
# See guardrails/confidence_guardrails.py for the full implementation,
# including configurable default values and threshold gating.


# ============================================================
# STAGE 3.3 — Node Definitions
# ============================================================

def agent_node(state: ConfidenceGateState) -> dict:
    """
    Clinical agent — real LLM call.

    The agent is prompted to include a confidence score in
    its response. The score is extracted and stored separately
    in state for the confidence gate to read.
    """
    llm = get_llm()
    patient = state["patient_case"]

    system = SystemMessage(content=(
        "You are a clinical triage specialist. Assess the patient below. "
        "At the end of your response, include a line:\n"
        "Confidence: X.XX\n"
        "where X.XX is your confidence in this assessment (0.00 to 1.00).\n"
        "Use lower confidence when information is ambiguous or incomplete."
    ))

    prompt = f"""Patient: {patient.get('age')}y {patient.get('sex')}
Complaint: {patient.get('chief_complaint')}
Symptoms: {', '.join(patient.get('symptoms', []))}
Medications: {', '.join(patient.get('current_medications', []))}
Labs: {json.dumps(patient.get('lab_results', {}))}
Vitals: {json.dumps(patient.get('vitals', {}))}"""

    config = build_callback_config(trace_name="confidence_gate_agent")
    response = llm.invoke([system, HumanMessage(content=prompt)], config=config)

    content = response.content
    confidence = extract_confidence(content)

    print(f"    | [Agent] Response length: {len(content)} chars")
    print(f"    | [Agent] Extracted confidence: {confidence:.2f}")

    return {
        "messages": [response],
        "agent_response": content,
        "confidence": confidence,
    }


def confidence_gate_node(state: ConfidenceGateState) -> dict:
    """
    Compare the agent's confidence to the threshold.

    This node does NOT route — it writes the comparison result
    to state. The conditional edge router reads it.
    """
    confidence = state["confidence"]
    threshold = state["threshold"]
    result = "above" if confidence >= threshold else "below"

    print(f"    | [Gate] Confidence: {confidence:.2f}, Threshold: {threshold:.2f} -> {result}")

    return {"gate_result": result}


def route_after_gate(state: ConfidenceGateState) -> Literal["deliver", "escalate"]:
    """
    Route based on the gate result.

    "above" -> deliver (confidence is adequate)
    "below" -> escalate (needs human review)
    """
    if state["gate_result"] == "above":
        return "deliver"
    return "escalate"


def deliver_node(state: ConfidenceGateState) -> dict:
    """Deliver response — confidence meets the threshold."""
    return {
        "final_output": state["agent_response"],
        "status": "delivered",
    }


def escalate_node(state: ConfidenceGateState) -> dict:
    """
    Escalate for human review — confidence below threshold.

    In production, this node would:
        - Create a review ticket
        - Send notification to on-call physician
        - Queue the response for human approval
        - Use interrupt() for true HITL (see script_04c_hitl_review.py)
    """
    confidence = state["confidence"]
    threshold = state["threshold"]
    escalation_msg = (
        f"ESCALATED FOR REVIEW\n"
        f"Confidence: {confidence:.0%} (threshold: {threshold:.0%})\n"
        f"{'=' * 40}\n"
        f"{state['agent_response']}"
    )
    return {
        "final_output": escalation_msg,
        "status": "escalated",
    }


# ============================================================
# STAGE 3.4 — Graph Construction
# ============================================================

def build_confidence_gate_graph():
    """
    Build and compile the confidence gating graph.

    Graph structure:
        START → agent → confidence_gate → (conditional) → deliver or escalate → END
    """
    workflow = StateGraph(ConfidenceGateState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("confidence_gate", confidence_gate_node)
    workflow.add_node("deliver", deliver_node)
    workflow.add_node("escalate", escalate_node)

    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", "confidence_gate")
    workflow.add_conditional_edges(
        "confidence_gate",
        route_after_gate,
        {"deliver": "deliver", "escalate": "escalate"},
    )
    workflow.add_edge("deliver", END)
    workflow.add_edge("escalate", END)

    return workflow.compile()


# ============================================================
# STAGE 3.5 — Test Cases
# ============================================================

def make_state(patient: PatientCase, threshold: float) -> ConfidenceGateState:
    return {
        "messages": [],
        "patient_case": patient.model_dump(),
        "agent_response": "",
        "confidence": 0.0,
        "threshold": threshold,
        "gate_result": "",
        "final_output": "",
        "status": "pending",
    }


def main() -> None:
    print("\n" + "=" * 70)
    print("  CONFIDENCE GATING")
    print("  Pattern: threshold-based routing")
    print("=" * 70)

    print("""
    Graph:

        [START] -> [agent] -> [confidence_gate]
                                    |
                        +-----------+-----------+
                        |                       |
                     deliver                 escalate
                  (>= threshold)           (< threshold)
                        |                       |
                      [END]                   [END]

    The same agent response is routed differently depending
    on the confidence threshold configured at invocation time.
    """)

    graph = build_confidence_gate_graph()

    # ── Test 1: Clear-cut case, high threshold ─────────────────────────
    print("=" * 70)
    print("  TEST 1: Clear-cut COPD case (threshold = 0.60)")
    print("=" * 70)

    copd_patient = PatientCase(
        patient_id="PT-CG-001",
        age=58, sex="M",
        chief_complaint="Persistent cough and dyspnea for 3 weeks",
        symptoms=["cough", "dyspnea", "wheezing"],
        medical_history=["COPD Stage II", "Former smoker"],
        current_medications=["Tiotropium 18mcg inhaler daily"],
        allergies=[],
        lab_results={"FEV1": "58% predicted", "SpO2": "93%"},
        vitals={"BP": "138/85", "HR": "92"},
    )

    r1 = graph.invoke(make_state(copd_patient, threshold=0.60))
    print(f"\n    RESULT: {r1['status'].upper()}")
    print(f"    Confidence: {r1['confidence']:.2f} vs threshold: 0.60")

    # ── Test 2: Ambiguous case, aggressive threshold ───────────────────
    print("\n\n" + "=" * 70)
    print("  TEST 2: Ambiguous case (threshold = 0.90)")
    print("=" * 70)

    ambiguous_patient = PatientCase(
        patient_id="PT-CG-002",
        age=45, sex="F",
        chief_complaint="Intermittent dizziness, cause unclear",
        symptoms=["dizziness", "occasional nausea"],
        medical_history=[],
        current_medications=[],
        allergies=[],
        lab_results={},
        vitals={"BP": "120/80", "HR": "72"},
    )

    r2 = graph.invoke(make_state(ambiguous_patient, threshold=0.90))
    print(f"\n    RESULT: {r2['status'].upper()}")
    print(f"    Confidence: {r2['confidence']:.2f} vs threshold: 0.90")

    # ── Test 3: Same case, relaxed threshold ──────────────────────────
    print("\n\n" + "=" * 70)
    print("  TEST 3: Same ambiguous case (threshold = 0.30)")
    print("=" * 70)

    r3 = graph.invoke(make_state(ambiguous_patient, threshold=0.30))
    print(f"\n    RESULT: {r3['status'].upper()}")
    print(f"    Confidence: {r3['confidence']:.2f} vs threshold: 0.30")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  CONFIDENCE GATING COMPLETE")
    print("=" * 70)
    print(f"""
    Results:
      Test 1 (clear COPD, threshold=0.60) : {r1['status'].upper()}
      Test 2 (ambiguous, threshold=0.90)  : {r2['status'].upper()}
      Test 3 (ambiguous, threshold=0.30)  : {r3['status'].upper()}

    The threshold is configurable per deployment:
      Clinical research  -> low threshold (0.30), accept uncertainty
      Emergency triage   -> high threshold (0.80), escalate often
      Drug interaction   -> very high (0.90), almost always review

    How this differs from output validation:
      Output validation = checks CONTENT (prohibited terms, disclaimers)
      Confidence gating = checks CERTAINTY (agent's self-assessment)
      Both can be stacked: content check first, then confidence gate.

    Next: layered_validation.py — full input->agent->output pipeline.
    """)


if __name__ == "__main__":
    main()
