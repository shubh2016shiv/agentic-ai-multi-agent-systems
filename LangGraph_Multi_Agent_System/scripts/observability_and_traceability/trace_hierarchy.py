#!/usr/bin/env python3
"""
============================================================
Trace Hierarchy
============================================================
Pattern 1: Langfuse trace -> span -> generation structure
in a clinical multi-agent pipeline.
Prerequisite: None (standalone)

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Langfuse organises telemetry into a hierarchy:

    Trace       one complete workflow execution
    Span        a logical phase within a trace
    Generation  one LLM call (prompt -> completion + usage)

When using LangChain/LangGraph with the Langfuse callback
handler, this hierarchy is created AUTOMATICALLY:

    graph.invoke()                     -> Trace
      node_function()                  -> Span
        llm.invoke(prompt, config)     -> Generation

This script demonstrates:
    1. How build_callback_config() wires Langfuse to LangGraph
    2. How each node's LLM call becomes a traced Generation
    3. Trace metadata: user_id, session_id, tags
    4. Inspecting traces programmatically via get_langfuse_client()
    5. Graceful degradation when Langfuse is not configured

------------------------------------------------------------
OBSERVABILITY vs TRACEABILITY — what this covers
------------------------------------------------------------
    TRACEABILITY (what/who):
        - Which agent ran, in what order
        - What prompt was sent, what completion came back
        - Input/output of each node

    OBSERVABILITY (how/why):
        - Token count per generation
        - Latency per span
        - Model used per call
        - Cost estimation per trace

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [triage]          <-- LLM call -> Generation under Span
       |
       v
    [assessment]      <-- LLM call -> Generation under Span
       |
       v
    [recommendation]  <-- LLM call -> Generation under Span
       |
       v
    [END]

    All three Spans are children of one Trace.

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    main()           graph.invoke()     triage_node        assessment_node     recommendation_node
      |                   |                 |                    |                    |
      |-- build_config -->|                 |                    |                    |
      |   (Langfuse CB)   |                 |                    |                    |
      |-- invoke(state) ->|                 |                    |                    |
      |                   |--- Trace start  |                    |                    |
      |                   |--- Span ------->|                    |                    |
      |                   |                 |-- llm.invoke()     |                    |
      |                   |                 |-- Generation       |                    |
      |                   |                 |   (tokens, cost)   |                    |
      |                   |<-- state -------|                    |                    |
      |                   |--- Span ------->|                    |                    |
      |                   |                 |------------------->|                    |
      |                   |                 |                    |-- llm.invoke()     |
      |                   |                 |                    |-- Generation       |
      |                   |                 |                    |<-- state ----------|
      |                   |--- Span ------->|                    |                    |
      |                   |                 |------------------------------------>   |
      |                   |                 |                    |                    |-- llm.invoke()
      |                   |                 |                    |                    |-- Generation
      |                   |--- Trace end    |                    |                    |
      |<-- result --------|                 |                    |                    |

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.observability_and_traceability.trace_hierarchy
============================================================
"""

# -- Standard library --------------------------------------------------------
import sys
import json
from typing import TypedDict, Annotated

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# -- LangGraph ---------------------------------------------------------------
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# -- LangChain ---------------------------------------------------------------
from langchain_core.messages import HumanMessage, SystemMessage

# -- Project imports ----------------------------------------------------------
# CONNECTION: core/ root module — get_llm() centralises LLM config so all nodes
# use the same provider/model without hardcoding credentials.
from core.config import get_llm
# CONNECTION: observability/ root module — build_callback_config() is the PRIMARY
# entry point that attaches Langfuse tracing to every LLM call via LangChain
# callbacks. It injects trace_name, user_id, session_id, and tags automatically.
# get_langfuse_client() gives programmatic access to inspect traces after the run.
# This script demonstrates HOW to wire them into a LangGraph pipeline.
from observability.callbacks import build_callback_config
from observability.tracer import get_langfuse_client


# ============================================================
# STAGE 1.1 -- State Definition
# ============================================================

class TraceHierarchyState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_summary: str
    triage_result: str
    assessment_result: str
    recommendation_result: str


# ============================================================
# STAGE 1.2 -- Node Definitions
# ============================================================

def triage_node(state: TraceHierarchyState) -> dict:
    """
    Triage agent -- produces urgency classification.

    The LLM call here becomes a GENERATION in Langfuse,
    nested under a SPAN named after this graph node.
    The callback config passed to llm.invoke() is what
    triggers the automatic tracing.
    """
    llm = get_llm()
    patient = state.get("patient_summary", "No patient data")

    system = SystemMessage(content=(
        "You are a clinical triage specialist. Classify the patient's "
        "urgency and provide a brief triage summary (3-4 sentences). "
        "Include urgency level: ROUTINE, URGENT, or EMERGENT."
    ))
    prompt = HumanMessage(content=f"Triage this patient:\n{patient}")

    # build_callback_config creates a config dict with:
    #   - Langfuse CallbackHandler in config["callbacks"]
    #   - Trace metadata in config["metadata"] (langfuse_trace_name, etc.)
    config = build_callback_config(
        trace_name="trace_hierarchy_triage",
        user_id="doctor-demo-001",
        tags=["triage", "trace_hierarchy_demo"],
    )

    response = llm.invoke([system, prompt], config=config)
    triage_result = response.content

    print(f"    | [Triage] Result: {len(triage_result)} chars")
    print(f"    | [Triage] This LLM call is a GENERATION in Langfuse")

    return {
        "messages": [response],
        "triage_result": triage_result,
    }


def assessment_node(state: TraceHierarchyState) -> dict:
    """
    Assessment agent -- produces clinical assessment.

    Reads the triage result from state and produces a
    deeper assessment. Another GENERATION in Langfuse.
    """
    llm = get_llm()
    patient = state.get("patient_summary", "")
    triage = state.get("triage_result", "")

    system = SystemMessage(content=(
        "You are a clinical assessment specialist. Based on the triage "
        "summary, provide a clinical assessment (3-4 sentences). "
        "Include potential diagnoses and risk factors."
    ))
    prompt = HumanMessage(content=f"""Assess this patient:

Patient: {patient}

Triage Summary: {triage}""")

    config = build_callback_config(
        trace_name="trace_hierarchy_assessment",
        user_id="doctor-demo-001",
        tags=["assessment", "trace_hierarchy_demo"],
    )

    response = llm.invoke([system, prompt], config=config)
    assessment_result = response.content

    print(f"    | [Assessment] Result: {len(assessment_result)} chars")
    print(f"    | [Assessment] This is a second GENERATION in the same Trace")

    return {
        "messages": [response],
        "assessment_result": assessment_result,
    }


def recommendation_node(state: TraceHierarchyState) -> dict:
    """
    Recommendation agent -- produces treatment recommendations.

    Reads triage + assessment and synthesises a recommendation.
    Third GENERATION in the trace.
    """
    llm = get_llm()
    patient = state.get("patient_summary", "")
    triage = state.get("triage_result", "")
    assessment = state.get("assessment_result", "")

    system = SystemMessage(content=(
        "You are a clinical recommendation specialist. Based on the "
        "triage and assessment, provide treatment recommendations "
        "(3-4 sentences). Include medication changes, monitoring, "
        "and follow-up. End with 'Consult your healthcare provider.'"
    ))
    prompt = HumanMessage(content=f"""Recommend treatment:

Patient: {patient}

Triage: {triage}

Assessment: {assessment}""")

    config = build_callback_config(
        trace_name="trace_hierarchy_recommendation",
        user_id="doctor-demo-001",
        tags=["recommendation", "trace_hierarchy_demo"],
    )

    response = llm.invoke([system, prompt], config=config)
    recommendation_result = response.content

    print(f"    | [Recommendation] Result: {len(recommendation_result)} chars")
    print(f"    | [Recommendation] Third GENERATION completes the Trace")

    return {
        "messages": [response],
        "recommendation_result": recommendation_result,
    }


# ============================================================
# STAGE 1.3 -- Graph Construction
# ============================================================

def build_trace_hierarchy_graph():
    """
    Build a 3-node clinical pipeline.

    Each node's LLM call is traced via build_callback_config().
    Langfuse auto-creates: Trace -> Span (per node) -> Generation (per LLM call).
    """
    workflow = StateGraph(TraceHierarchyState)

    workflow.add_node("triage", triage_node)
    workflow.add_node("assessment", assessment_node)
    workflow.add_node("recommendation", recommendation_node)

    workflow.add_edge(START, "triage")
    workflow.add_edge("triage", "assessment")
    workflow.add_edge("assessment", "recommendation")
    workflow.add_edge("recommendation", END)

    return workflow.compile()


# ============================================================
# STAGE 1.4 -- Main
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  TRACE HIERARCHY")
    print("  Pattern: Langfuse trace -> span -> generation")
    print("=" * 70)

    print("""
    Langfuse hierarchy for multi-agent systems:

      TRACE (one workflow execution)
        |
        +-- SPAN: triage_node
        |     |
        |     +-- GENERATION: llm.invoke() [tokens, cost, latency]
        |
        +-- SPAN: assessment_node
        |     |
        |     +-- GENERATION: llm.invoke() [tokens, cost, latency]
        |
        +-- SPAN: recommendation_node
              |
              +-- GENERATION: llm.invoke() [tokens, cost, latency]

    The callback handler creates this AUTOMATICALLY.
    You only need: config = build_callback_config(trace_name="...")
    Then pass it: llm.invoke(messages, config=config)
    """)

    # -- Check Langfuse availability ----------------------------------------
    langfuse_client = get_langfuse_client()
    langfuse_available = langfuse_client is not None
    print(f"    Langfuse available: {langfuse_available}")
    if not langfuse_available:
        print("    (Tracing will use NoOp -- pipeline runs normally without Langfuse)")
    print()

    # -- Build and run pipeline ---------------------------------------------
    patient_summary = (
        "68M with Type 2 Diabetes, CKD Stage 3b, Hypertension. "
        "Presenting with persistent fatigue, ankle edema, and elevated "
        "creatinine (2.1 mg/dL). Current medications: Metformin 1000mg BID, "
        "Lisinopril 40mg daily, Amlodipine 10mg daily. "
        "Labs: HbA1c 8.2%, eGFR 38, K+ 5.1 mEq/L, BNP 320 pg/mL."
    )

    initial_state = {
        "messages": [],
        "patient_summary": patient_summary,
        "triage_result": "",
        "assessment_result": "",
        "recommendation_result": "",
    }

    print(f"    Patient: 68M | DM2, CKD 3b, HTN")
    print(f"    Complaint: fatigue, edema, elevated creatinine")
    print()
    print("    " + "-" * 60)

    graph = build_trace_hierarchy_graph()
    result = graph.invoke(initial_state)

    # -- Display results ----------------------------------------------------
    print("\n    " + "=" * 60)
    print("    PIPELINE RESULTS")
    print("    " + "=" * 60)

    print(f"\n    TRIAGE:")
    for line in result["triage_result"][:300].split("\n"):
        if line.strip():
            print(f"      {line}")

    print(f"\n    ASSESSMENT:")
    for line in result["assessment_result"][:300].split("\n"):
        if line.strip():
            print(f"      {line}")

    print(f"\n    RECOMMENDATION:")
    for line in result["recommendation_result"][:300].split("\n"):
        if line.strip():
            print(f"      {line}")

    # -- Trace hierarchy summary --------------------------------------------
    print("\n\n" + "=" * 70)
    print("  TRACE HIERARCHY SUMMARY")
    print("=" * 70)
    print("""
    What happened in Langfuse:

      1. build_callback_config() created a config dict with:
         - CallbackHandler in config["callbacks"]
         - Trace metadata in config["metadata"]

      2. Each llm.invoke(messages, config=config) auto-created:
         - A Generation with prompt, completion, token counts

      3. The Langfuse dashboard shows:
         Trace: trace_hierarchy_*
           Span: triage_node
             Generation: llm call (input tokens, output tokens, cost)
           Span: assessment_node
             Generation: llm call (input tokens, output tokens, cost)
           Span: recommendation_node
             Generation: llm call (input tokens, output tokens, cost)

    Key takeaway:
      Observability is AUTOMATIC. You wire build_callback_config()
      once, and every LLM call is captured with full context.

    Traceability is INHERENT. The trace shows exactly what each
    agent received as input and produced as output, in order.

    Next: agent_metrics_and_cost.py -- per-agent cost breakdown.
    """)


if __name__ == "__main__":
    main()
