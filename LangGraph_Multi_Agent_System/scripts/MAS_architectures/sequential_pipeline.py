#!/usr/bin/env python3
"""
============================================================
Sequential Pipeline
============================================================
Pattern 2: Agents process in a fixed order, each enriching
the context for the next (assembly-line pattern).

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
The pipeline is the simplest MAS architecture. Agents execute
in a predetermined sequence. Each agent receives the accumulated
output from all previous stages.

This is ideal when:
    - The clinical workflow has a natural order (triage first)
    - Each stage depends on the previous stage's output
    - You need the most deterministic and debuggable flow
    - Task order is always the same regardless of patient data

When NOT to use:
    - If agent order should vary per case (use supervisor)
    - If agents can work independently (use parallel voting)

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [triage]          <-- stage 1: urgency classification
       |
       v
    [diagnostic]      <-- stage 2: receives triage context
       |
       v
    [pharmacist]      <-- stage 3: receives triage + diagnostic context
       |
       v
    [synthesizer]     <-- stage 4: compiles all into final report
       |
       v
    [END]

    Context accumulates at each stage ------>

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    main()       triage         diagnostic      pharmacist      synthesizer
      |             |                |               |               |
      |-- invoke -->|                |               |               |
      |             |-- process()    |               |               |
      |             |-- state ------>|               |               |
      |             |  (triage_out)  |-- process()   |               |
      |             |                |-- state ----->|               |
      |             |   (triage_out  |  (diag_out)   |-- process()   |
      |             |    + diag_out) |               |-- state ----->|
      |             |                |      (triage  |  (pharma_out) |-- LLM
      |             |                |   + diag      |               |-- report
      |<-- report --|                |   + pharma)   |               |

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.MAS_architectures.sequential_pipeline
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

# -- Project imports ----------------------------------------------------------
# CONNECTION: agents/ root module — TriageAgent, DiagnosticAgent, PharmacistAgent
# are pre-built reusable agent objects. This script demonstrates the SEQUENTIAL
# PIPELINE ARCHITECTURE PATTERN (fixed-order agent execution), not agent design.
# See agents/ for what each agent does internally.
from agents import TriageAgent, PharmacistAgent, DiagnosticAgent
# CONNECTION: core/ root module — get_llm() centralises LLM config.
# PatientCase is the canonical domain model passed through state.
from core.config import get_llm
from core.models import PatientCase
# CONNECTION: observability/ root module — build_callback_config() attaches
# Langfuse tracing to every LLM call across the sequential pipeline.
from observability.callbacks import build_callback_config


# ============================================================
# STAGE 2.1 -- State Definition
# ============================================================

class PipelineState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_case: dict
    stage_number: int
    triage_output: str
    diagnostic_output: str
    pharmacist_output: str
    accumulated_context: str
    final_report: str


# ============================================================
# STAGE 2.2 -- Agent Instances
# ============================================================

triage_agent = TriageAgent()
diagnostic_agent = DiagnosticAgent()
pharmacist_agent = PharmacistAgent()


# ============================================================
# STAGE 2.3 -- Node Definitions
# ============================================================

def triage_stage(state: PipelineState) -> dict:
    """
    Stage 1: Triage -- no upstream context, works from raw patient data.
    Sets the foundation for all downstream stages.
    """
    patient = state["patient_case"]
    result = triage_agent.process_with_context(patient)

    accumulated = f"[TRIAGE]:\n{result}"
    print(f"    | Stage 1 [Triage]: {result[:120]}...")

    return {
        "triage_output": result,
        "accumulated_context": accumulated,
        "stage_number": 1,
    }


def diagnostic_stage(state: PipelineState) -> dict:
    """
    Stage 2: Diagnostic -- receives triage output as upstream context.
    Builds differential diagnosis informed by urgency classification.
    """
    patient = state["patient_case"]
    upstream_context = state.get("accumulated_context", "")

    result = diagnostic_agent.process_with_context(patient, context=upstream_context)

    accumulated = f"{upstream_context}\n\n[DIAGNOSTIC]:\n{result}"
    print(f"    | Stage 2 [Diagnostic]: {result[:120]}...")

    return {
        "diagnostic_output": result,
        "accumulated_context": accumulated,
        "stage_number": 2,
    }


def pharmacist_stage(state: PipelineState) -> dict:
    """
    Stage 3: Pharmacist -- receives triage + diagnostic as upstream context.
    Medication review informed by urgency and differential diagnosis.
    """
    patient = state["patient_case"]
    upstream_context = state.get("accumulated_context", "")

    result = pharmacist_agent.process_with_context(patient, context=upstream_context)

    accumulated = f"{upstream_context}\n\n[PHARMACIST]:\n{result}"
    print(f"    | Stage 3 [Pharmacist]: {result[:120]}...")

    return {
        "pharmacist_output": result,
        "accumulated_context": accumulated,
        "stage_number": 3,
    }


def synthesizer_stage(state: PipelineState) -> dict:
    """
    Stage 4: Synthesizer -- compiles all upstream findings into a report.
    This node is not a BaseAgent subclass; it is a pipeline-specific
    aggregation step.
    """
    llm = get_llm()
    accumulated = state.get("accumulated_context", "")

    prompt = f"""Compile these staged clinical findings into a structured report:

{accumulated}

Format as: 1) Triage Summary  2) Differential Diagnosis  3) Medication Review  4) Final Recommendations
Keep under 200 words."""

    config = build_callback_config(trace_name="pipeline_synthesizer")
    response = llm.invoke(prompt, config=config)
    print(f"    | Stage 4 [Synthesizer]: {len(response.content)} chars generated")

    return {
        "final_report": response.content,
        "stage_number": 4,
    }


# ============================================================
# STAGE 2.4 -- Graph Construction
# ============================================================

def build_pipeline_graph():
    """
    Build a fixed-order sequential pipeline.

    Key LangGraph pattern:
        - Only add_edge (no conditional edges)
        - Simplest possible graph structure
        - Context accumulates via state['accumulated_context']
    """
    workflow = StateGraph(PipelineState)

    workflow.add_node("triage", triage_stage)
    workflow.add_node("diagnostic", diagnostic_stage)
    workflow.add_node("pharmacist", pharmacist_stage)
    workflow.add_node("synthesizer", synthesizer_stage)

    workflow.add_edge(START, "triage")
    workflow.add_edge("triage", "diagnostic")
    workflow.add_edge("diagnostic", "pharmacist")
    workflow.add_edge("pharmacist", "synthesizer")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()


# ============================================================
# STAGE 2.5 -- Main
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  SEQUENTIAL PIPELINE")
    print("  Pattern: fixed-order assembly line with context accumulation")
    print("=" * 70)

    print("""
    Pipeline:

        [Triage] --> [Diagnostic] --> [Pharmacist] --> [Synthesizer]
           |              |                |               |
           +-- context grows richer at each stage -------->+

    Each stage receives ALL previous outputs via accumulated_context.
    """)

    patient = PatientCase(
        patient_id="PT-ARCH-002",
        age=68, sex="M",
        chief_complaint="Chest pain and shortness of breath with elevated troponin",
        symptoms=["chest pain", "dyspnea", "diaphoresis", "nausea"],
        medical_history=["Hypertension", "Type 2 Diabetes", "Hyperlipidemia"],
        current_medications=["Lisinopril 20mg", "Metformin 1000mg BID", "Atorvastatin 40mg"],
        allergies=["Penicillin"],
        lab_results={"Troponin": "0.15 ng/mL", "BNP": "380 pg/mL", "HbA1c": "7.2%"},
        vitals={"BP": "158/95", "HR": "102", "SpO2": "94%"},
    )

    initial_state = {
        "messages": [],
        "patient_case": patient.model_dump(),
        "stage_number": 0,
        "triage_output": "",
        "diagnostic_output": "",
        "pharmacist_output": "",
        "accumulated_context": "",
        "final_report": "",
    }

    print(f"    Patient: {patient.patient_id} | {patient.age}y {patient.sex}")
    print(f"    Complaint: {patient.chief_complaint}")
    print()
    print("    " + "-" * 60)

    graph = build_pipeline_graph()
    result = graph.invoke(initial_state)

    # -- Display context growth --------------------------------------------
    print("\n    " + "=" * 60)
    print("    CONTEXT ACCUMULATION")
    print("    " + "-" * 60)
    context = result.get("accumulated_context", "")
    print(f"    Total context length: {len(context)} chars across 3 stages")

    # -- Display final report ----------------------------------------------
    print("\n    " + "=" * 60)
    print("    CLINICAL REPORT")
    print("    " + "-" * 60)
    for line in result.get("final_report", "").split("\n"):
        print(f"    | {line}")

    # -- Summary -----------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  SEQUENTIAL PIPELINE SUMMARY")
    print("=" * 70)
    print("""
    What you learned:

      1. Pipeline = simplest architecture (only add_edge, no routing)
      2. Context accumulates via accumulated_context state field
      3. Each stage receives ALL previous outputs
      4. Same BaseAgent subclasses used as in supervisor pattern
      5. Deterministic, fully reproducible execution order

    When to use:
      - Fixed clinical workflow (triage -> diagnose -> prescribe)
      - Each step genuinely needs the previous step's output
      - Maximum debuggability (execution order is always the same)

    When NOT to use:
      - If agents can work independently (use parallel voting)
      - If agent order should vary (use supervisor)

    Next: parallel_voting.py
    """)


if __name__ == "__main__":
    main()
