#!/usr/bin/env python3
"""
============================================================
Map-Reduce Fan-Out
============================================================
Pattern 6: Split a problem into independent sub-problems,
process them in parallel, then aggregate results.

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Map-Reduce is the workhorse pattern for parallelizable work.
Unlike voting (where agents assess the SAME thing), map-reduce
gives each agent a DIFFERENT sub-task:

    MAP:    Split the patient case into sub-problems
            - Sub-problem 1: cardiac assessment (Triage Agent)
            - Sub-problem 2: medication review (Pharmacist Agent)
            - Sub-problem 3: differential diagnosis (Diagnostic Agent)

    REDUCE: Merge all specialist results into one aggregated view

    PRODUCE: Synthesize the aggregated view into a final report

This is ideal when:
    - The problem has naturally independent sub-problems
    - Sub-tasks can be defined upfront (no dynamic routing)
    - Latency must be minimized (parallel > sequential)

When NOT to use:
    - When sub-tasks depend on each other (use pipeline)
    - When you want independent views of the SAME problem (use voting)

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [mapper]              <-- defines sub-tasks and fans out via Send
       |
       +-- Send --> [worker] (triage sub-task)      --+
       +-- Send --> [worker] (pharmacist sub-task)  --+--> [reducer]
       +-- Send --> [worker] (diagnostic sub-task)  --+       |
                                                              v
                                                         [producer]
                                                              |
                                                              v
                                                           [END]

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    main()     mapper         worker(x3)         reducer         producer
      |           |               |                 |               |
      |-- invoke->|               |                 |               |
      |           |-- Send(triage)|                 |               |
      |           |-- Send(pharma)|                 |               |
      |           |-- Send(diag)  |                 |               |
      |           |          [parallel execution]   |               |
      |           |     triage ---|                  |               |
      |           |     pharma ---|                  |               |
      |           |     diag   --|                  |               |
      |           |              +-- operator.add ->|               |
      |           |              | (all results     |-- aggregate   |
      |           |              |  merged into     |---- state --->|
      |           |              |  single list)    |               |-- LLM
      |<-- report-|              |                  |               |-- report

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.MAS_architectures.map_reduce_fanout
============================================================
"""

# -- Standard library --------------------------------------------------------
import sys
import json
import operator
from typing import TypedDict, Annotated

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# -- LangGraph ---------------------------------------------------------------
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send

# -- Project imports ----------------------------------------------------------
# CONNECTION: agents/ root module — TriageAgent, DiagnosticAgent, PharmacistAgent
# are pre-built reusable agent objects. This script demonstrates the MAP-REDUCE
# FANOUT PATTERN (parallel map phase + aggregated reduce phase), not agent design.
# See agents/ for what each agent does internally.
from agents import TriageAgent, PharmacistAgent, DiagnosticAgent
# CONNECTION: core/ root module — get_llm() centralises LLM config for the
# reduce/aggregator LLM. PatientCase is the canonical domain model.
from core.config import get_llm
from core.models import PatientCase
# CONNECTION: observability/ root module — build_callback_config() attaches
# Langfuse tracing to every LLM call in both the map and reduce phases.
from observability.callbacks import build_callback_config


# ============================================================
# STAGE 6.1 -- State Definitions
# ============================================================

class WorkerInput(TypedDict):
    """Input for each parallel worker instance."""
    worker_type: str
    sub_task_description: str
    patient_case: dict


class MapReduceState(TypedDict):
    """
    Main state for the map-reduce graph.

    worker_results uses operator.add as a reducer so that
    parallel worker outputs are merged (not overwritten).
    """
    patient_case: dict
    worker_results: Annotated[list[dict], operator.add]
    aggregated_findings: str
    final_report: str


# ============================================================
# STAGE 6.2 -- Agent Instances
# ============================================================

WORKER_AGENTS = {
    "triage": TriageAgent(),
    "pharmacist": PharmacistAgent(),
    "diagnostic": DiagnosticAgent(),
}


# ============================================================
# STAGE 6.3 -- Sub-Task Definitions
# ============================================================

# Each sub-task is a DIFFERENT aspect of the same patient case.
# Unlike voting (same question, different perspectives), map-reduce
# gives each worker a UNIQUE sub-problem.

SUB_TASKS = {
    "triage": "Classify urgency and identify critical vital signs. Focus ONLY on triage assessment.",
    "pharmacist": "Review medication regimen for interactions and dose adjustments. Focus ONLY on medications.",
    "diagnostic": "Generate differential diagnosis and recommend further workup. Focus ONLY on diagnosis.",
}


# ============================================================
# STAGE 6.4 -- Node Definitions
# ============================================================

def mapper_node(state: MapReduceState) -> dict:
    """
    MAP phase: prepare for fan-out. The actual Send list is returned
    by the conditional edge router so LangGraph treats it as routing.
    """
    print(f"    | [Mapper] MAP phase: {len(SUB_TASKS)} sub-tasks dispatched")
    for worker_type, sub_task in SUB_TASKS.items():
        print(f"    |   {worker_type}: {sub_task[:60]}...")
    return {}


def route_mapper(state: MapReduceState) -> list[Send]:
    """Conditional edge router: return list of Send for parallel workers."""
    patient = state["patient_case"]
    sends = []
    for worker_type, sub_task_description in SUB_TASKS.items():
        sends.append(
            Send(
                "worker",
                WorkerInput(
                    worker_type=worker_type,
                    sub_task_description=sub_task_description,
                    patient_case=patient,
                ),
            )
        )
    return sends


def worker_node(state: WorkerInput) -> dict:
    """
    Worker: executes one sub-task using the appropriate agent.

    Each worker:
        1. Gets its sub-task description as extra context
        2. Processes via the shared BaseAgent.process()
        3. Returns result tagged with worker_type

    Multiple instances run concurrently; results merge via operator.add.
    """
    worker_type = state["worker_type"]
    patient = state["patient_case"]
    sub_task = state["sub_task_description"]

    agent = WORKER_AGENTS[worker_type]
    result = agent.process_with_context(patient, context=f"Sub-task focus: {sub_task}")

    print(f"    | [Worker:{worker_type}] Completed: {result[:80]}...")

    return {
        "worker_results": [
            {
                "worker": worker_type,
                "sub_task": sub_task,
                "result": result,
            }
        ],
    }


def reducer_node(state: MapReduceState) -> dict:
    """
    REDUCE phase: aggregate all worker results into a unified view.

    This node runs AFTER all parallel workers complete.
    worker_results contains all results merged via operator.add.

    The reducer organizes findings by category without adding
    interpretation (that is the producer's job).
    """
    results = state.get("worker_results", [])

    aggregated_sections = []
    for worker_result in results:
        worker_name = worker_result["worker"]
        finding = worker_result["result"]
        aggregated_sections.append(f"[{worker_name.upper()} FINDINGS]:\n{finding}")

    aggregated = "\n\n".join(aggregated_sections)
    print(f"    | [Reducer] REDUCE phase: {len(results)} results aggregated ({len(aggregated)} chars)")

    return {"aggregated_findings": aggregated}


def producer_node(state: MapReduceState) -> dict:
    """
    PRODUCE phase: synthesize aggregated findings into a final report.

    This is the value-add step — the LLM connects dots across
    sub-tasks that individual workers could not see.

    For example: the triage finding (K+ 5.4) + pharmacist finding
    (Lisinopril + Spironolactone) = potassium risk that neither
    sub-task would catch alone.
    """
    llm = get_llm()
    aggregated = state.get("aggregated_findings", "")

    prompt = f"""You are a clinical synthesizer. Individual specialists have independently
assessed different aspects of the same patient case. Their findings:

{aggregated}

IMPORTANT: Look for connections between findings that individual specialists
could not see (e.g., medication interactions that affect lab values).

Produce a UNIFIED clinical report:
1) Critical Findings (across all sub-tasks)
2) Cross-Domain Connections (findings that interact)
3) Integrated Recommendation
Keep under 200 words."""

    config = build_callback_config(trace_name="map_reduce_producer", tags=["map_reduce"])
    response = llm.invoke(prompt, config=config)
    print(f"    | [Producer] PRODUCE phase: {len(response.content)} chars generated")

    return {"final_report": response.content}


# ============================================================
# STAGE 6.5 -- Graph Construction
# ============================================================

def build_map_reduce_graph():
    """
    Build the map-reduce fan-out graph.

    Key patterns:
        - mapper returns list[Send] (fan-out)
        - worker runs in parallel (multiple instances)
        - operator.add reducer merges results
        - reducer organizes, producer synthesizes
    """
    workflow = StateGraph(MapReduceState)

    workflow.add_node("mapper", mapper_node)
    workflow.add_node("worker", worker_node)
    workflow.add_node("reducer", reducer_node)
    workflow.add_node("producer", producer_node)

    workflow.add_edge(START, "mapper")
    workflow.add_conditional_edges("mapper", route_mapper, ["worker"])
    workflow.add_edge("worker", "reducer")
    workflow.add_edge("reducer", "producer")
    workflow.add_edge("producer", END)

    return workflow.compile()


# ============================================================
# STAGE 6.6 -- Main
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  MAP-REDUCE FAN-OUT")
    print("  Pattern: split -> parallel process -> aggregate -> synthesize")
    print("=" * 70)

    print("""
    Three phases:

      MAP:     Split patient case into 3 independent sub-tasks
               [Triage sub-task] [Pharmacist sub-task] [Diagnostic sub-task]

      REDUCE:  Aggregate all specialist findings into one view
               [All findings organized by category]

      PRODUCE: Synthesize cross-domain connections
               [Unified report with insights no single agent could see]

    Key difference from voting:
      Voting  = same question, different perspectives
      Map-Reduce = different questions, different specialists
    """)

    patient = PatientCase(
        patient_id="PT-ARCH-006",
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
        "patient_case": patient.model_dump(),
        "worker_results": [],
        "aggregated_findings": "",
        "final_report": "",
    }

    print(f"    Patient: {patient.patient_id} | {patient.age}y {patient.sex}")
    print(f"    Sub-tasks: {list(SUB_TASKS.keys())}")
    print()
    print("    " + "-" * 60)

    graph = build_map_reduce_graph()
    result = graph.invoke(initial_state)

    # -- Display worker results --------------------------------------------
    print("\n    " + "=" * 60)
    print("    MAP PHASE RESULTS (individual sub-tasks)")
    print("    " + "-" * 60)
    for worker_result in result.get("worker_results", []):
        name = worker_result["worker"]
        text = worker_result["result"][:200]
        print(f"    [{name.upper()}]: {text}...")
        print()

    # -- Display final report ----------------------------------------------
    print("    " + "=" * 60)
    print("    PRODUCE PHASE (synthesized report)")
    print("    " + "-" * 60)
    for line in result.get("final_report", "").split("\n"):
        print(f"    | {line}")

    # -- Summary -----------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  MAP-REDUCE FAN-OUT SUMMARY")
    print("=" * 70)
    print("""
    What you learned:

      1. MAP: mapper returns list[Send] with DIFFERENT sub-tasks
      2. Workers run in parallel, each with a unique sub-problem
      3. REDUCE: operator.add merges results; reducer organizes them
      4. PRODUCE: synthesizer finds cross-domain connections
      5. Same BaseAgent subclasses, different sub-task descriptions

    Key difference from other parallel patterns:
      Voting:     same input  -> same question   -> consensus
      Map-Reduce: same input  -> different tasks  -> synthesis
      Fan-Out:    same input  -> different agents -> merge

    When to use:
      - Independent sub-problems that can be defined upfront
      - Latency-critical workflows (parallel > sequential)
      - When cross-domain synthesis is the goal

    Next: reflection_self_critique.py
    """)


if __name__ == "__main__":
    main()
