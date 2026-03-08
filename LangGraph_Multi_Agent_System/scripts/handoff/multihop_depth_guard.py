#!/usr/bin/env python3
"""
============================================================
Multi-Hop Handoff with Depth Guard
============================================================
Prerequisite: All previous handoff scripts

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Allow agents to hand off to each other dynamically, but
enforce a depth limit (circuit breaker) that stops the chain
when it gets too deep. Without a depth guard, LLM-driven
routing can loop indefinitely.

This script combines conditional routing with a counter-based
guard function. Each agent increments handoff_depth. The guard
checks it BEFORE routing to the next agent.

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    [START]
       |
       v
    [triage]
       |
       v
    check_depth()           <-- depth guard (Python function)
       |
       +-- depth < max --> [pharmacology]
       |                       |
       |                       v
       |                   check_depth()
       |                       |
       |                       +-- depth < max --> [guidelines]
       |                       |                       |
       |                       |                       v
       |                       |                   check_depth()
       |                       |                       |
       |                       |                       +-- depth >= max --> TRIP
       |                       |
       |                       +-- depth >= max --> TRIP
       +-- depth >= max --> TRIP
       |
       v
    [report]   <-- runs when guard trips or chain completes
       |
       v
     [END]

    Routing: conditional edges with a depth guard function.
    Guard: increments handoff_depth, checks against max.
    Trip action: redirect to [report] with whatever data is available.

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. Depth guard — loop prevention for multi-hop chains
    2. HandoffLimitReached exception (imported but handled gracefully)
    3. Counter-based circuit breaker pattern
    4. Same graph, different behaviour at different depth limits
    5. Cost control — fewer hops = fewer LLM calls

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.handoff.multihop_depth_guard
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

# ── LangChain ───────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ── Project imports ─────────────────────────────────────────────────────────
from core.config import get_llm
from core.models import PatientCase
from core.exceptions import HandoffLimitReached
from tools import (
    analyze_symptoms,
    assess_patient_risk,
    check_drug_interactions,
    lookup_drug_info,
    calculate_dosage_adjustment,
    lookup_clinical_guideline,
)
from observability.callbacks import build_callback_config


# ============================================================
# STAGE 5.1 — State Definition
# ============================================================

class MultihopState(TypedDict):
    """State with depth tracking for multi-hop chains."""
    messages: Annotated[list, add_messages]
    patient_case: dict
    current_agent: str
    handoff_history: list[str]
    handoff_depth: int
    max_handoff_depth: int
    depth_guard_tripped: bool
    final_report: str


# ============================================================
# STAGE 5.2 — Depth Guard Functions
# ============================================================
# These are router functions — they run BETWEEN nodes and
# return a routing key. They don't modify state.

def check_depth_after_triage(state: MultihopState) -> Literal["continue", "trip"]:
    """Check depth after triage. Route to pharmacology or trip."""
    depth = state["handoff_depth"]
    max_d = state["max_handoff_depth"]
    ok = depth < max_d
    label = "PASS" if ok else f"TRIPPED (depth {depth} >= max {max_d})"
    print(f"\n    [DEPTH GUARD] After triage: depth={depth}, max={max_d} -> {label}")
    return "continue" if ok else "trip"


def check_depth_after_pharma(state: MultihopState) -> Literal["continue", "trip"]:
    """Check depth after pharmacology. Route to guidelines or trip."""
    depth = state["handoff_depth"]
    max_d = state["max_handoff_depth"]
    ok = depth < max_d
    label = "PASS" if ok else f"TRIPPED (depth {depth} >= max {max_d})"
    print(f"\n    [DEPTH GUARD] After pharmacology: depth={depth}, max={max_d} -> {label}")
    return "continue" if ok else "trip"


def check_depth_after_guidelines(state: MultihopState) -> Literal["continue", "trip"]:
    """Check depth after guidelines. Route forward or trip."""
    depth = state["handoff_depth"]
    max_d = state["max_handoff_depth"]
    ok = depth < max_d
    label = "PASS" if ok else f"TRIPPED (depth {depth} >= max {max_d})"
    print(f"\n    [DEPTH GUARD] After guidelines: depth={depth}, max={max_d} -> {label}")
    return "continue" if ok else "trip"


# ============================================================
# STAGE 5.3 — Build Graph
# ============================================================

def build_multihop_pipeline(max_depth: int):
    """
    Build a multi-hop graph with the given depth limit.

    Args:
        max_depth: Maximum number of handoffs allowed.
    """
    llm = get_llm()

    triage_tools = [analyze_symptoms, assess_patient_risk]
    pharma_tools = [check_drug_interactions, lookup_drug_info, calculate_dosage_adjustment]
    guideline_tools = [lookup_clinical_guideline]

    patient = PatientCase(
        patient_id="PT-MH-001",
        age=71, sex="F",
        chief_complaint="Dizziness and fatigue after medication change",
        symptoms=["dizziness", "fatigue", "headache", "bilateral ankle edema"],
        medical_history=["Hypertension", "CKD Stage 3a", "Type 2 Diabetes"],
        current_medications=[
            "Lisinopril 20mg daily", "Spironolactone 25mg daily",
            "Metformin 1000mg BID", "Amlodipine 10mg daily",
        ],
        allergies=["Sulfa drugs"],
        lab_results={"eGFR": "42 mL/min", "K+": "5.4 mEq/L", "Cr": "1.6 mg/dL"},
        vitals={"BP": "105/65", "HR": "88", "SpO2": "95%"},
    )

    # ── Node: triage ────────────────────────────────────────────────────
    def triage_node(state: MultihopState) -> dict:
        print("\n    [HOP 1] TRIAGE AGENT")
        print("    " + "-" * 50)

        triage_llm = llm.bind_tools(triage_tools)
        system_msg = SystemMessage(content="Evaluate symptoms and risk. Be concise.")
        user_msg = HumanMessage(content=f"""Patient: {patient.age}y {patient.sex}
Complaint: {patient.chief_complaint}
Symptoms: {', '.join(patient.symptoms)}
Labs: {json.dumps(patient.lab_results)}

Use your tools, provide a brief assessment.""")

        config = build_callback_config(trace_name="handoff_multihop_triage")
        messages = [system_msg, user_msg]
        response = triage_llm.invoke(messages, config=config)

        while hasattr(response, "tool_calls") and response.tool_calls:
            print(f"    | Triage calling {len(response.tool_calls)} tool(s)")
            tool_node = ToolNode(triage_tools)
            tool_results = tool_node.invoke({"messages": [response]})
            messages.extend([response] + tool_results["messages"])
            response = triage_llm.invoke(messages, config=config)

        print(f"    Assessment: {len(response.content)} chars")

        return {
            "messages": [response],
            "current_agent": "triage",
            "handoff_history": state["handoff_history"] + ["triage"],
            "handoff_depth": state["handoff_depth"] + 1,
        }

    # ── Node: pharmacology ──────────────────────────────────────────────
    def pharmacology_node(state: MultihopState) -> dict:
        print("\n    [HOP 2] PHARMACOLOGY AGENT")
        print("    " + "-" * 50)

        pharma_llm = llm.bind_tools(pharma_tools)
        system_msg = SystemMessage(content="Review medications for interactions. Be concise.")
        user_msg = HumanMessage(content=f"""Medications: {', '.join(patient.current_medications)}
Labs: K+={patient.lab_results.get('K+')}, eGFR={patient.lab_results.get('eGFR')}

Use your tools, provide recommendations.""")

        config = build_callback_config(trace_name="handoff_multihop_pharma")
        messages = [system_msg, user_msg]
        response = pharma_llm.invoke(messages, config=config)

        while hasattr(response, "tool_calls") and response.tool_calls:
            print(f"    | Pharmacology calling {len(response.tool_calls)} tool(s)")
            tool_node = ToolNode(pharma_tools)
            tool_results = tool_node.invoke({"messages": [response]})
            messages.extend([response] + tool_results["messages"])
            response = pharma_llm.invoke(messages, config=config)

        print(f"    Recommendation: {len(response.content)} chars")

        return {
            "messages": [response],
            "current_agent": "pharmacology",
            "handoff_history": state["handoff_history"] + ["pharmacology"],
            "handoff_depth": state["handoff_depth"] + 1,
        }

    # ── Node: guidelines ────────────────────────────────────────────────
    def guidelines_node(state: MultihopState) -> dict:
        print("\n    [HOP 3] GUIDELINES AGENT")
        print("    " + "-" * 50)

        guide_llm = llm.bind_tools(guideline_tools)
        system_msg = SystemMessage(content="Look up clinical guidelines relevant to this case. Be concise.")
        user_msg = HumanMessage(content=f"""Patient: {patient.age}y {patient.sex}
Conditions: {', '.join(patient.medical_history)}
Medications: {', '.join(patient.current_medications)}

Look up the most relevant clinical guideline.""")

        config = build_callback_config(trace_name="handoff_multihop_guidelines")
        messages = [system_msg, user_msg]
        response = guide_llm.invoke(messages, config=config)

        while hasattr(response, "tool_calls") and response.tool_calls:
            print(f"    | Guidelines calling {len(response.tool_calls)} tool(s)")
            tool_node = ToolNode(guideline_tools)
            tool_results = tool_node.invoke({"messages": [response]})
            messages.extend([response] + tool_results["messages"])
            response = guide_llm.invoke(messages, config=config)

        print(f"    Guideline reference: {len(response.content)} chars")

        return {
            "messages": [response],
            "current_agent": "guidelines",
            "handoff_history": state["handoff_history"] + ["guidelines"],
            "handoff_depth": state["handoff_depth"] + 1,
        }

    # ── Node: report ────────────────────────────────────────────────────
    def report_node(state: MultihopState) -> dict:
        print("\n    [REPORT] GENERATING SUMMARY")
        print("    " + "-" * 50)

        tripped = state["handoff_depth"] >= state["max_handoff_depth"]
        if tripped:
            print(f"    NOTE: Depth guard tripped. Report from partial data.")

        agent_outputs = [
            msg.content for msg in state["messages"]
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls
        ]
        synthesis = "\n---\n".join(agent_outputs[-5:])

        report_llm = get_llm()
        config = build_callback_config(trace_name="handoff_multihop_report")

        guard_note = " (DEPTH LIMIT REACHED — partial analysis)" if tripped else ""
        response = report_llm.invoke(
            f"Clinical summary{guard_note} (max 150 words):\n\n{synthesis}",
            config=config,
        )

        print(f"\n    Report ({len(response.content)} chars):")
        for line in response.content.split("\n")[:6]:
            print(f"    | {line}")

        return {
            "final_report": response.content,
            "depth_guard_tripped": tripped,
        }

    # ── Wire the graph ──────────────────────────────────────────────────
    workflow = StateGraph(MultihopState)

    workflow.add_node("triage", triage_node)
    workflow.add_node("pharmacology", pharmacology_node)
    workflow.add_node("guidelines", guidelines_node)
    workflow.add_node("report", report_node)

    workflow.add_edge(START, "triage")

    # After each agent: check depth before routing to next
    workflow.add_conditional_edges(
        "triage",
        check_depth_after_triage,
        {"continue": "pharmacology", "trip": "report"},
    )
    workflow.add_conditional_edges(
        "pharmacology",
        check_depth_after_pharma,
        {"continue": "guidelines", "trip": "report"},
    )
    workflow.add_conditional_edges(
        "guidelines",
        check_depth_after_guidelines,
        {"continue": "report", "trip": "report"},
    )

    workflow.add_edge("report", END)

    return workflow.compile(), patient


# ============================================================
# STAGE 5.4 — Test Runs
# ============================================================

def run_with_depth(depth: int) -> None:
    """Run the pipeline with a specific max_handoff_depth."""
    print(f"\n{'=' * 70}")
    print(f"  TEST: max_handoff_depth = {depth}")
    if depth >= 3:
        print("  Expected: triage -> pharmacology -> guidelines -> report (full chain)")
    elif depth == 2:
        print("  Expected: triage -> pharmacology -> GUARD TRIPS -> report")
    elif depth == 1:
        print("  Expected: triage -> GUARD TRIPS -> report")
    print(f"{'=' * 70}")

    graph, patient = build_multihop_pipeline(max_depth=depth)

    initial_state: MultihopState = {
        "messages": [],
        "patient_case": patient.model_dump(),
        "current_agent": "none",
        "handoff_history": [],
        "handoff_depth": 0,
        "max_handoff_depth": depth,
        "depth_guard_tripped": False,
        "final_report": "",
    }

    result = graph.invoke(initial_state)

    print(f"\n    Handoff chain : {' -> '.join(result['handoff_history'])}")
    print(f"    Final depth   : {result['handoff_depth']} / {depth}")
    print(f"    Guard tripped : {result['depth_guard_tripped']}")


def main() -> None:
    print("\n" + "=" * 70)
    print("  MULTI-HOP HANDOFF WITH DEPTH GUARD")
    print("  Pattern: agents chain to each other, depth limit prevents loops")
    print("=" * 70)

    print("""
    Graph topology (max_depth=3):

        [START] -> [triage] -> check_depth -> [pharmacology] -> check_depth
                                                                     |
                  [report] <- check_depth <- [guidelines] <----------+
                     |
                   [END]

    Graph topology (max_depth=1 — guard trips early):

        [START] -> [triage] -> check_depth -> TRIPPED -> [report] -> [END]

    Each agent increments handoff_depth by 1.
    The guard function checks: depth < max_handoff_depth?
      - YES -> continue to next agent
      - NO  -> redirect to report (partial data)
    """)

    # Run 1: Full chain (depth=3 allows all 3 agents)
    run_with_depth(3)

    # Run 2: Guard trips after pharmacology (depth=2)
    run_with_depth(2)

    # Run 3: Guard trips after triage (depth=1)
    run_with_depth(1)

    print("\n\n" + "=" * 70)
    print("  MULTI-HOP DEPTH GUARD COMPLETE")
    print("=" * 70)
    print("""
    What happened:
      depth=3: triage -> pharmacology -> guidelines -> report (FULL chain)
      depth=2: triage -> pharmacology -> GUARD TRIPS -> report (partial)
      depth=1: triage -> GUARD TRIPS -> report (minimal)

    Key lesson:
      - Every LLM-driven handoff chain needs a depth limit.
      - Without it, agents can loop indefinitely.
      - The guard is a simple Python function (zero token cost).
      - Lower depth = fewer LLM calls = lower cost.
      - Report still runs with whatever data is available.

    Production pattern:
      - Set max_handoff_depth based on your SLA and budget.
      - Log when the guard trips (indicates the problem was complex).
      - Combine with HandoffLimitReached exception for hard failures.

    This completes the handoff pattern series.
    Next: see scripts/tools/ for tool-focused patterns.
    """)


if __name__ == "__main__":
    main()
