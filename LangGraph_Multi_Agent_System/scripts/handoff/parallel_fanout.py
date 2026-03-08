#!/usr/bin/env python3
"""
============================================================
Parallel Fan-Out Handoff
============================================================
Prerequisite: supervisor.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Fan out a patient case to multiple specialist agents that
run IN PARALLEL, then merge their results in a reduce step.
Uses LangGraph's Send API to create parallel task instances
at runtime.

All previous patterns were sequential — one agent at a time.
This pattern runs agents concurrently whenever they don't
depend on each other's output.

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    [START]
       |
       v
    [coordinator]
       |
       +-- Send("triage", state)    --> [specialist]  (instance 1)
       |                                     |
       +-- Send("pharma", state)   --> [specialist]  (instance 2)
       |                                     |
       v                                     v
    [merge]  <---------- results from both specialists
       |
       v
    [report]
       |
       v
     [END]

    Routing:  Send API — creates parallel node instances.
    Who decides: the coordinator node emits Send objects.
    Parallelism: agents run concurrently, not sequentially.

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. Send(node_name, state) — create parallel execution branches
    2. Reducer functions (operator.add) for merging parallel results
    3. Fan-out from a coordinator node
    4. Fan-in via a merge node that collects all specialist outputs
    5. When to use parallel vs sequential handoffs

------------------------------------------------------------
WHEN TO USE
------------------------------------------------------------
    Use parallel_fanout when multiple agents can run independently
    and concurrently to reduce total workflow latency.
    All parallel branches are merged before the next sequential step.

    When NOT to use:
    - If agents need each other's output as input (use supervisor.py)
    - If execution order matters (use linear_pipeline or supervisor)

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.handoff.parallel_fanout
============================================================
"""

# ── Standard library ────────────────────────────────────────────────────────
import sys
import json
import operator
from typing import TypedDict, Annotated

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── LangGraph ───────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.prebuilt import ToolNode

# ── LangChain ───────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage

# ── Project imports ─────────────────────────────────────────────────────────
# CONNECTION: core/ root module — get_llm() centralises LLM config.
# PatientCase is the canonical domain model fanned out to each specialist.
from core.config import get_llm
from core.models import PatientCase
# CONNECTION: tools/ root module — domain tool functions (component layer).
# Each parallel specialist binds its own scoped tool subset.
from tools import (
    analyze_symptoms,
    assess_patient_risk,
    check_drug_interactions,
    lookup_drug_info,
    calculate_dosage_adjustment,
)
# CONNECTION: observability/ root module — build_callback_config() attaches
# Langfuse trace_name and tags to every LLM call automatically.
from observability.callbacks import build_callback_config


# ============================================================
# STAGE 5.1 — State Definitions
# ============================================================
# Two state types:
#   1. SpecialistInput — what each parallel specialist receives
#   2. FanoutState — the main graph state with a reducer

class SpecialistInput(TypedDict):
    """Input passed to each parallel specialist instance."""
    specialist_type: str    # "triage" or "pharmacology"
    patient_case: dict
    task_description: str


class FanoutState(TypedDict):
    """
    Main state for the fan-out graph.

    specialist_results uses operator.add as a reducer.
    When multiple specialist instances write to this field,
    their results are concatenated into a single list
    instead of overwriting each other.
    """
    patient_case: dict
    specialist_results: Annotated[list[dict], operator.add]
    final_report: str


# ============================================================
# STAGE 5.2 — Build Graph
# ============================================================

def build_fanout_pipeline():
    """Build a parallel fan-out graph using Send API."""
    llm = get_llm()

    triage_tools = [analyze_symptoms, assess_patient_risk]
    pharma_tools = [check_drug_interactions, lookup_drug_info, calculate_dosage_adjustment]

    patient = PatientCase(
        patient_id="PT-FAN-001",
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

    # ── Node: coordinator ───────────────────────────────────────────────
    # Returns a list of Send objects that create parallel branches.
    def coordinator_node(state: FanoutState) -> dict:
        """
        Coordinator node simply prints and prepares for fan-out.
        """
        print("\n    [Step 5.2] COORDINATOR — fanning out to specialists")
        print("    " + "-" * 50)
        return {}

    def route_fanout(state: FanoutState) -> list[Send]:
        """
        A conditional edge router that returns a list of Send objects.
        Each Send creates a parallel instance of the "specialist" node.
        """
        patient_data = state["patient_case"]

        tasks = [
            {
                "specialist_type": "triage",
                "patient_case": patient_data,
                "task_description": (
                    "Evaluate symptoms, identify urgent concerns, "
                    "flag issues. Focus on symptom severity and risk."
                ),
            },
            {
                "specialist_type": "pharmacology",
                "patient_case": patient_data,
                "task_description": (
                    "Review medications for drug interactions and "
                    "renal dosing. Focus on K+ level and eGFR."
                ),
            },
        ]

        sends = []
        for task in tasks:
            print(f"    | Routing to: {task['specialist_type']}")
            sends.append(Send("specialist", task))

        print(f"    | Total parallel branches: {len(sends)}")
        return sends

    # ── Node: specialist (parallel instances) ───────────────────────────
    # This same node function runs in parallel for each Send.
    # The specialist_type field in the input determines which
    # tools and prompt to use.
    def specialist_node(state: SpecialistInput) -> dict:
        """
        A specialist agent that runs in parallel.

        The specialist_type determines tools and system prompt.
        Multiple instances of this node run concurrently.
        """
        spec_type = state["specialist_type"]
        patient_data = state["patient_case"]
        task = state["task_description"]

        print(f"\n    [PARALLEL] {spec_type.upper()} SPECIALIST")
        print("    " + "-" * 50)

        if spec_type == "triage":
            tools = triage_tools
            system = "You are a triage specialist. Evaluate symptoms and risk."
        elif spec_type == "pharmacology":
            tools = pharma_tools
            system = "You are a clinical pharmacologist. Review medications."
        else:
            tools = []
            system = f"You are a {spec_type} specialist."

        spec_llm = llm.bind_tools(tools) if tools else llm

        system_msg = SystemMessage(content=system)
        user_msg = HumanMessage(content=f"""Task: {task}

Patient: {patient_data.get('age')}y {patient_data.get('sex')}
Chief Complaint: {patient_data.get('chief_complaint')}
Symptoms: {', '.join(patient_data.get('symptoms', []))}
Medications: {', '.join(patient_data.get('current_medications', []))}
Labs: {json.dumps(patient_data.get('lab_results', {}))}
Vitals: {json.dumps(patient_data.get('vitals', {}))}

Use your tools, then provide your assessment.""")

        config = build_callback_config(trace_name=f"handoff_fanout_{spec_type}")
        messages = [system_msg, user_msg]
        response = spec_llm.invoke(messages, config=config)

        while hasattr(response, "tool_calls") and response.tool_calls:
            print(f"    | {spec_type} calling {len(response.tool_calls)} tool(s)")
            tool_node = ToolNode(tools)
            tool_results = tool_node.invoke({"messages": [response]})
            messages.extend([response] + tool_results["messages"])
            response = spec_llm.invoke(messages, config=config)

        print(f"    | {spec_type} assessment: {len(response.content)} chars")

        # Return to the reducer. Each parallel instance appends its
        # result to specialist_results (operator.add merges them).
        return {
            "specialist_results": [
                {
                    "specialist": spec_type,
                    "assessment": response.content,
                    "char_count": len(response.content),
                }
            ]
        }

    # ── Node: merge ─────────────────────────────────────────────────────
    # After all parallel specialists complete, the reducer has
    # merged their results. This node reads the merged list.
    def merge_node(state: FanoutState) -> dict:
        """
        Merge results from all parallel specialists.

        By the time this node runs, specialist_results contains
        outputs from ALL parallel instances (merged by operator.add).
        """
        print("\n    [Step 5.3] MERGE — collecting parallel results")
        print("    " + "-" * 50)

        results = state.get("specialist_results", [])
        print(f"    Received {len(results)} specialist result(s)")
        for r in results:
            print(f"      - {r['specialist']}: {r['char_count']} chars")

        return {}  # Nothing to write — data already in state

    # ── Node: report ────────────────────────────────────────────────────
    def report_node(state: FanoutState) -> dict:
        """Synthesise all specialist assessments into a final report."""
        print("\n    [Step 5.4] REPORT GENERATOR")
        print("    " + "-" * 50)

        results = state.get("specialist_results", [])
        combined = "\n\n--- Next Specialist ---\n\n".join(
            f"[{r['specialist'].upper()}]\n{r['assessment']}" for r in results
        )

        report_llm = get_llm()
        config = build_callback_config(trace_name="handoff_fanout_report")
        response = report_llm.invoke(
            f"Synthesise these parallel specialist assessments into a "
            f"clinical summary (max 200 words):\n\n{combined}",
            config=config,
        )

        print(f"\n    Report ({len(response.content)} chars):")
        for line in response.content.split("\n")[:8]:
            print(f"    | {line}")

        return {"final_report": response.content}

    # ── Wire the graph ──────────────────────────────────────────────────
    workflow = StateGraph(FanoutState)

    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("specialist", specialist_node)
    workflow.add_node("merge", merge_node)
    workflow.add_node("report", report_node)

    workflow.add_edge(START, "coordinator")
    # coordinator uses a conditional edge to fan out via Send objects
    workflow.add_conditional_edges("coordinator", route_fanout, ["specialist"])
    workflow.add_edge("specialist", "merge")
    workflow.add_edge("merge", "report")
    workflow.add_edge("report", END)

    return workflow.compile(), patient


# ============================================================
# STAGE 5.5 — Execute
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  PARALLEL FAN-OUT HANDOFF")
    print("  Pattern: Send API for parallel agent execution")
    print("=" * 70)

    print("""
    Graph topology:

        [START]
           |
           v
        [coordinator]
           |
           +-- Send("specialist", triage_input)    ──> [specialist] (triage)
           |                                                |
           +-- Send("specialist", pharma_input)    ──> [specialist] (pharma)
           |                                                |
           v                                                v
        [merge]  <── operator.add merges specialist_results
           |
           v
        [report]
           |
           v
         [END]

    Key mechanism:
      - coordinator returns a list of Send objects
      - Each Send creates a parallel instance of "specialist"
      - Results are merged via the reducer (operator.add)
      - All specialists run concurrently
    """)

    graph, patient = build_fanout_pipeline()

    print(f"    Patient: {patient.patient_id} | {patient.age}y {patient.sex}")
    print(f"    K+={patient.lab_results['K+']} | eGFR={patient.lab_results['eGFR']}")
    print()

    initial_state: FanoutState = {
        "patient_case": patient.model_dump(),
        "specialist_results": [],
        "final_report": "",
    }

    result = graph.invoke(initial_state)

    print("\n" + "=" * 70)
    print("  PARALLEL FAN-OUT COMPLETE")
    print("=" * 70)
    specialists = [r["specialist"] for r in result["specialist_results"]]
    print(f"    Specialists run : {', '.join(specialists)}")
    print(f"    Results merged  : {len(result['specialist_results'])} specialist(s)")

    print("""
    What happened:
      1. Coordinator emitted Send objects for triage and pharmacology.
      2. Both specialists ran in PARALLEL (not sequentially).
      3. operator.add merged their specialist_results into one list.
      4. Report node synthesised all results.

    When to use parallel vs sequential:
      - Parallel: agents don't depend on each other's output
      - Sequential: agent B needs agent A's findings (use linear_pipeline)

    Next: multihop_depth_guard.py — agents re-route to each other with a depth limit.
    """)


if __name__ == "__main__":
    main()
