#!/usr/bin/env python3
"""
============================================================
Conditional Routing Handoff
============================================================
Prerequisite: linear_pipeline.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Add branching to the pipeline using a Python router function.
After triage, a router function inspects the state and decides
whether the patient needs pharmacology review (high risk) or
can go straight to the report (low risk).

The routing decision is made by YOUR CODE — not by the LLM.
No LLM call is spent on the routing step. This keeps routing
deterministic, testable, and zero-cost.

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    [START]
       |
       v
    [triage]
       |
       v
    route_after_triage()        <-- Python function, no LLM call
       |
       +-- "high_risk" --> [pharmacology] --> [report] --> [END]
       |
       +-- "low_risk"  --> [report] --> [END]

    Routing:  add_conditional_edges() with a router function.
    Who decides: YOUR ROUTER FUNCTION (deterministic code).
    LLM influence: NONE — the LLM only runs inside agent nodes.
    Token cost for routing: ZERO.

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. add_conditional_edges() — branching based on state
    2. Router functions — pure Python, no LLM, easily unit-tested
    3. Risk-based routing — high_risk vs low_risk paths
    4. Two test cases through the same compiled graph

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.handoff.conditional_routing
============================================================
"""

# ── Standard library ────────────────────────────────────────────────────────
import sys
import json
from typing import TypedDict, Annotated, Literal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── LangGraph ───────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ── LangChain messages ──────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ── Project imports ─────────────────────────────────────────────────────────
from core.config import get_llm
from core.models import PatientCase, HandoffContext
from tools import (
    analyze_symptoms,
    assess_patient_risk,
    check_drug_interactions,
    lookup_drug_info,
    calculate_dosage_adjustment,
)
from observability.callbacks import build_callback_config


# ============================================================
# STAGE 2.1 — State Definition
# ============================================================
# Same as PipelineState from linear_pipeline.py, plus a
# risk_level field that the router function reads.

class ConditionalState(TypedDict):
    """
    Shared state with a risk_level field for routing.

    The triage node writes risk_level. The router function
    reads it and returns a routing key. The pharmacology node
    only runs if risk_level is "high".
    """
    messages: Annotated[list, add_messages]
    patient_case: dict
    handoff_context: dict
    current_agent: str
    handoff_history: list[str]
    handoff_depth: int
    risk_level: str         # "high" | "low" — written by triage, read by router
    final_report: str


# ============================================================
# STAGE 2.2 — Router Function
# ============================================================
# This function runs BETWEEN nodes. It reads the state,
# applies deterministic logic, and returns a string key
# that maps to the next node name.
#
# It is NOT a node — it does not modify state.
# It is NOT an LLM call — it costs zero tokens.
# It IS easily unit-testable: pass a dict, check the return value.

def route_after_triage(state: ConditionalState) -> Literal["high_risk", "low_risk"]:
    """
    Decide whether the patient needs pharmacology review.

    Returns "high_risk" if the triage node set risk_level to "high",
    otherwise returns "low_risk" to skip pharmacology and go
    directly to report generation.
    """
    risk = state.get("risk_level", "low")
    return "high_risk" if risk == "high" else "low_risk"


# ============================================================
# STAGE 2.3 — Build Graph
# ============================================================

def build_conditional_pipeline():
    """Build the conditional routing graph."""
    llm = get_llm()
    triage_tools = [analyze_symptoms, assess_patient_risk]
    pharma_tools = [check_drug_interactions, lookup_drug_info, calculate_dosage_adjustment]
    triage_llm = llm.bind_tools(triage_tools)
    pharma_llm = llm.bind_tools(pharma_tools)

    # ── Node: triage ────────────────────────────────────────────────────
    def triage_node(state: ConditionalState) -> dict:
        """
        Evaluate the patient. Write risk_level to state.

        The risk_level is derived from the patient's lab values
        and vitals — a simple rule-based assessment. In a production
        system you could parse the LLM's output for a risk score,
        but here we use deterministic rules so the router function
        is predictable for demonstration purposes.
        """
        print("\n    [STAGE 2.3] TRIAGE AGENT")
        print("    " + "-" * 50)

        patient_data = state["patient_case"]

        system_msg = SystemMessage(content=(
            "You are a triage specialist. Evaluate symptoms, "
            "identify urgent concerns, flag issues for specialist review."
        ))

        user_msg = HumanMessage(content=f"""Evaluate this patient:

Age: {patient_data.get('age')}y {patient_data.get('sex')}
Chief Complaint: {patient_data.get('chief_complaint')}
Symptoms: {', '.join(patient_data.get('symptoms', []))}
Medications: {', '.join(patient_data.get('current_medications', []))}
Labs: {json.dumps(patient_data.get('lab_results', {}), indent=2)}
Vitals: {json.dumps(patient_data.get('vitals', {}), indent=2)}

Use your tools to analyze symptoms and assess risk.""")

        config = build_callback_config(trace_name="handoff_cond_triage")
        messages = [system_msg, user_msg]
        response = triage_llm.invoke(messages, config=config)

        while hasattr(response, "tool_calls") and response.tool_calls:
            print(f"    | Triage calling {len(response.tool_calls)} tool(s)")
            tool_node = ToolNode(triage_tools)
            tool_results = tool_node.invoke({"messages": [response]})
            messages.extend([response] + tool_results["messages"])
            response = triage_llm.invoke(messages, config=config)

        # Determine risk level from patient data (deterministic rules)
        # This is what the router function reads.
        k_plus = patient_data.get("lab_results", {}).get("K+", "")
        try:
            k_val = float(k_plus.split()[0]) if k_plus else 0
        except (ValueError, IndexError):
            k_val = 0

        risk = "high" if k_val >= 5.0 else "low"
        print(f"\n    Risk assessment: K+={k_plus} -> risk_level={risk}")
        print(f"    Triage assessment: {len(response.content)} chars")

        # Build HandoffContext (only used if high_risk path runs)
        handoff = HandoffContext(
            from_agent="TriageAgent",
            to_agent="PharmacologyAgent",
            reason=f"Risk level: {risk}. Pharmacology review needed.",
            patient_case=PatientCase(**patient_data),
            task_description="Check drug interactions and renal dosing.",
            relevant_findings=[
                f"K+ = {k_plus}",
                f"Medications: {', '.join(patient_data.get('current_medications', []))}",
            ],
            handoff_depth=state["handoff_depth"] + 1,
        )

        return {
            "messages": [response],
            "handoff_context": handoff.model_dump(),
            "current_agent": "triage",
            "handoff_history": state["handoff_history"] + ["triage"],
            "handoff_depth": state["handoff_depth"] + 1,
            "risk_level": risk,
        }

    # ── Node: pharmacology ──────────────────────────────────────────────
    def pharmacology_node(state: ConditionalState) -> dict:
        """
        Review medications. Only runs on the high_risk path.
        Reads HandoffContext from state.
        """
        print("\n    [STAGE 2.4] PHARMACOLOGY AGENT (high-risk path)")
        print("    " + "-" * 50)

        raw_handoff = state.get("handoff_context", {})
        handoff = HandoffContext(**raw_handoff) if raw_handoff else None
        patient_data = state["patient_case"]

        findings = handoff.relevant_findings if handoff else []
        task = handoff.task_description if handoff else "Review medications."

        system_msg = SystemMessage(content=(
            "You are a clinical pharmacologist. Review medications "
            "for interactions, renal dosing, and safety."
        ))
        user_msg = HumanMessage(content=f"""Handoff from Triage Agent.

Task: {task}
Findings: {json.dumps(findings, indent=2)}
Medications: {chr(10).join('  - ' + m for m in patient_data.get('current_medications', []))}
Labs: {json.dumps(patient_data.get('lab_results', {}), indent=2)}

Use your tools, then provide specific recommendations.""")

        config = build_callback_config(trace_name="handoff_cond_pharma")
        messages = [system_msg, user_msg]
        response = pharma_llm.invoke(messages, config=config)

        while hasattr(response, "tool_calls") and response.tool_calls:
            print(f"    | Pharmacology calling {len(response.tool_calls)} tool(s)")
            tool_node = ToolNode(pharma_tools)
            tool_results = tool_node.invoke({"messages": [response]})
            messages.extend([response] + tool_results["messages"])
            response = pharma_llm.invoke(messages, config=config)

        print(f"    Pharmacology recommendation: {len(response.content)} chars")

        return {
            "messages": [response],
            "current_agent": "pharmacology",
            "handoff_history": state["handoff_history"] + ["pharmacology"],
            "handoff_depth": state["handoff_depth"] + 1,
        }

    # ── Node: report ────────────────────────────────────────────────────
    def report_node(state: ConditionalState) -> dict:
        """Synthesise specialist findings into a summary."""
        print("\n    [STAGE 2.5] REPORT GENERATOR")
        print("    " + "-" * 50)

        agent_outputs = [
            msg.content for msg in state["messages"]
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls
        ]
        synthesis = "\n---\n".join(agent_outputs[-3:])

        report_llm = get_llm()
        config = build_callback_config(trace_name="handoff_cond_report")
        response = report_llm.invoke(
            f"Synthesise into a clinical summary (max 150 words):\n\n{synthesis}",
            config=config,
        )

        print(f"\n    Report ({len(response.content)} chars):")
        for line in response.content.split("\n")[:8]:
            print(f"    | {line}")

        return {"final_report": response.content}

    # ── Wire the graph ──────────────────────────────────────────────────
    # The key difference from linear_pipeline.py:
    #   Instead of add_edge("triage", "pharmacology"),
    #   we use add_conditional_edges with a router function.

    workflow = StateGraph(ConditionalState)

    workflow.add_node("triage", triage_node)
    workflow.add_node("pharmacology", pharmacology_node)
    workflow.add_node("report", report_node)

    workflow.add_edge(START, "triage")

    # STAGE 2.6 — Conditional edge after triage
    # route_after_triage() returns "high_risk" or "low_risk".
    # The mapping dict translates these to node names.
    workflow.add_conditional_edges(
        "triage",
        route_after_triage,
        {"high_risk": "pharmacology", "low_risk": "report"},
    )

    workflow.add_edge("pharmacology", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


# ============================================================
# STAGE 2.7 — Test Cases
# ============================================================

def make_initial_state(patient: PatientCase) -> ConditionalState:
    return {
        "messages": [],
        "patient_case": patient.model_dump(),
        "handoff_context": {},
        "current_agent": "none",
        "handoff_history": [],
        "handoff_depth": 0,
        "risk_level": "low",
        "final_report": "",
    }


def run_high_risk(graph) -> None:
    """Test case 1: High-risk patient (K+ 5.4) -> pharmacology path."""
    print("\n" + "=" * 70)
    print("  TEST 1: HIGH-RISK PATIENT (K+ 5.4 mEq/L)")
    print("  Expected path: triage -> pharmacology -> report")
    print("=" * 70)

    patient = PatientCase(
        patient_id="PT-CR-HIGH",
        age=71, sex="F",
        chief_complaint="Dizziness after medication change",
        symptoms=["dizziness", "fatigue", "ankle edema"],
        medical_history=["Hypertension", "CKD Stage 3a"],
        current_medications=["Lisinopril 20mg", "Spironolactone 25mg", "Metformin 1000mg"],
        allergies=["Sulfa drugs"],
        lab_results={"K+": "5.4 mEq/L", "eGFR": "42 mL/min"},
        vitals={"BP": "105/65", "HR": "88"},
    )

    result = graph.invoke(make_initial_state(patient))
    print(f"\n    Path taken    : {' -> '.join(result['handoff_history'])}")
    print(f"    Risk level    : {result['risk_level']}")
    print(f"    Handoff depth : {result['handoff_depth']}")


def run_low_risk(graph) -> None:
    """Test case 2: Low-risk patient (K+ 4.0) -> skip pharmacology."""
    print("\n\n" + "=" * 70)
    print("  TEST 2: LOW-RISK PATIENT (K+ 4.0 mEq/L)")
    print("  Expected path: triage -> report (pharmacology SKIPPED)")
    print("=" * 70)

    patient = PatientCase(
        patient_id="PT-CR-LOW",
        age=45, sex="M",
        chief_complaint="Routine follow-up, mild cough",
        symptoms=["mild cough", "slight fatigue"],
        medical_history=["Hypertension (well-controlled)"],
        current_medications=["Lisinopril 10mg daily"],
        allergies=[],
        lab_results={"K+": "4.0 mEq/L", "eGFR": "85 mL/min"},
        vitals={"BP": "128/80", "HR": "72"},
    )

    result = graph.invoke(make_initial_state(patient))
    print(f"\n    Path taken    : {' -> '.join(result['handoff_history'])}")
    print(f"    Risk level    : {result['risk_level']}")
    print(f"    Handoff depth : {result['handoff_depth']}")


def main() -> None:
    print("\n" + "=" * 70)
    print("  CONDITIONAL ROUTING HANDOFF")
    print("  Pattern: router function decides the next node")
    print("=" * 70)

    print("""
    Graph topology:

        [START]
           |
           v
        [triage]
           |
           v
        route_after_triage()     <-- Python function, zero tokens
           |
           +-- "high_risk" --> [pharmacology] --+
           |                                    |
           +-- "low_risk"  --------------------+---> [report] --> [END]

    Difference from linear_pipeline.py:
      - add_conditional_edges() replaces add_edge() after triage
      - A router function inspects state["risk_level"]
      - Low-risk patients SKIP pharmacology entirely
    """)

    graph = build_conditional_pipeline()

    run_high_risk(graph)
    run_low_risk(graph)

    print("\n\n" + "=" * 70)
    print("  CONDITIONAL ROUTING COMPLETE")
    print("=" * 70)
    print("""
    What happened:
      Test 1: K+ 5.4 -> risk_level="high" -> triage -> pharmacology -> report
      Test 2: K+ 4.0 -> risk_level="low"  -> triage -> report (pharmacology skipped)

    Key points:
      - The router function is plain Python — zero token cost.
      - It reads state["risk_level"] and returns a string key.
      - Same graph, different paths based on patient data.
      - Routing logic is easily unit-testable.

    Next: command_handoff.py — the LLM itself decides routing via tools.
    """)


if __name__ == "__main__":
    main()
