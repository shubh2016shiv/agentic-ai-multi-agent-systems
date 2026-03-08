#!/usr/bin/env python3
"""
============================================================
Supervisor Orchestration
============================================================
Pattern 1: A central supervisor LLM dynamically routes
tasks to specialist agents and decides when to finish.

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
The supervisor pattern is the most common production MAS
architecture. A coordinator agent:

    1. Receives the task
    2. Decides which specialist should work next
    3. Reviews the specialist's output
    4. Routes to the next specialist or finishes

This is ideal when:
    - Task order is NOT predetermined
    - The orchestrator needs to adapt routing based on results
    - You need centralized control and clear accountability

When NOT to use:
    - If the task order is always the same (use pipeline)
    - If you need maximum parallelism (use fan-out)

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [supervisor]  <------+------+------+
       |                 |      |      |
       |-- "triage" ---->|      |      |
       |-- "diagnostic" ------->|      |
       |-- "pharmacist" -------------->|
       |-- "FINISH" ---> [report] --> [END]

    All specialist agents return to supervisor after execution.
    Supervisor re-evaluates and routes to the next or finishes.

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    main()       supervisor        triage        diagnostic     pharmacist      report
      |              |                |               |              |             |
      |-- invoke --> |                |               |              |             |
      |              |-- route -----> |               |              |             |
      |              |<-- result ---- |               |              |             |
      |              |-- route -------|-------------> |              |             |
      |              |<-- result -----|-------------- |              |             |
      |              |-- route -------|---------------|------------>  |             |
      |              |<-- result -----|---------------|------------- |             |
      |              |-- FINISH ------|---------------|--------------|-----------> |
      |<-- report ---|               |               |              |             |

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.MAS_architectures.supervisor_orchestration
============================================================
"""

# -- Standard library --------------------------------------------------------
import sys
import json
from typing import TypedDict, Annotated, Literal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# -- LangGraph ---------------------------------------------------------------
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# -- LangChain ---------------------------------------------------------------
from langchain_core.messages import HumanMessage, SystemMessage

# -- Project imports ----------------------------------------------------------
# CONNECTION: agents/ root module — TriageAgent, DiagnosticAgent, PharmacistAgent
# are pre-built reusable agent objects. This script demonstrates the SUPERVISOR
# ARCHITECTURE PATTERN (how to route between agents), not agent implementation.
# See agents/ for what each agent does internally.
from agents import TriageAgent, PharmacistAgent, DiagnosticAgent
# CONNECTION: core/ root module — get_llm() centralises LLM config for the
# supervisor LLM. PatientCase is the canonical domain model passed through state.
from core.config import get_llm
from core.models import PatientCase
# CONNECTION: observability/ root module — build_callback_config() attaches
# Langfuse tracing to every LLM call (supervisor decisions + agent calls).
from observability.callbacks import build_callback_config


# ============================================================
# STAGE 1.1 -- State Definition
# ============================================================

class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_case: dict
    next_agent: str                    # supervisor's routing decision
    completed_agents: list[str]        # agents that have already run
    agent_outputs: dict                # {agent_name: output_string}
    iteration: int
    max_iterations: int
    final_report: str


# ============================================================
# STAGE 1.2 -- Agent Instances (shared, reusable)
# ============================================================

triage_agent = TriageAgent()
pharmacist_agent = PharmacistAgent()
diagnostic_agent = DiagnosticAgent()

AGENT_REGISTRY = {
    "triage": triage_agent,
    "pharmacist": pharmacist_agent,
    "diagnostic": diagnostic_agent,
}


# ============================================================
# STAGE 1.3 -- Node Definitions
# ============================================================

def supervisor_node(state: SupervisorState) -> dict:
    """
    Supervisor LLM -- decides which agent to route to next.

    The supervisor sees:
        - The patient case summary
        - Which agents have already completed
        - The outputs from completed agents

    It returns one of: "triage", "diagnostic", "pharmacist", or "FINISH".
    """
    llm = get_llm()
    patient = state["patient_case"]
    completed = state.get("completed_agents", [])
    outputs = state.get("agent_outputs", {})
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 5)

    # Safety guard: force finish if max iterations reached
    if iteration >= max_iterations:
        print(f"    | [Supervisor] Max iterations ({max_iterations}) reached, finishing")
        return {"next_agent": "FINISH", "iteration": iteration + 1}

    available_agents = [name for name in AGENT_REGISTRY if name not in completed]

    if not available_agents:
        print(f"    | [Supervisor] All agents completed, finishing")
        return {"next_agent": "FINISH", "iteration": iteration + 1}

    # Build context for supervisor
    completed_summary = ""
    if outputs:
        completed_summary = "\n".join(
            f"- {agent}: {output[:200]}" for agent, output in outputs.items()
        )

    supervisor_prompt = f"""You are a clinical supervisor orchestrating a patient assessment.

Patient: {patient.get('age')}y {patient.get('sex')}, {patient.get('chief_complaint')}
Vitals: {json.dumps(patient.get('vitals', {}))}
Labs: {json.dumps(patient.get('lab_results', {}))}

Completed agents: {', '.join(completed) if completed else 'None'}
Available agents: {', '.join(available_agents)}

{f'Results so far:{chr(10)}{completed_summary}' if completed_summary else ''}

Which agent should run NEXT? Consider clinical workflow priority:
1. Triage first (urgency classification)
2. Diagnostic second (differential diagnosis)
3. Pharmacist third (medication review)

Respond with ONLY the agent name (one of: {', '.join(available_agents)}) or "FINISH" if all necessary work is done."""

    config = build_callback_config(
        trace_name="supervisor_routing",
        tags=["mas_architecture", "supervisor"],
    )
    response = llm.invoke(supervisor_prompt, config=config)
    decision = response.content.strip().lower()

    # Parse the decision
    if "finish" in decision:
        next_agent = "FINISH"
    else:
        # Find the best matching agent name
        next_agent = None
        for agent_name in available_agents:
            if agent_name in decision:
                next_agent = agent_name
                break
        if next_agent is None:
            next_agent = available_agents[0]  # fallback

    print(f"    | [Supervisor] Iteration {iteration + 1}: route -> {next_agent}")

    return {
        "next_agent": next_agent,
        "iteration": iteration + 1,
    }


def triage_worker_node(state: SupervisorState) -> dict:
    """Triage worker -- delegates to the shared TriageAgent."""
    patient = state["patient_case"]
    context = "\n".join(
        f"{k}: {v[:150]}" for k, v in state.get("agent_outputs", {}).items()
    )

    result = triage_agent.process_with_context(patient, context)
    print(f"    | [Triage] {result[:120]}...")

    completed = list(state.get("completed_agents", []))
    completed.append("triage")
    outputs = dict(state.get("agent_outputs", {}))
    outputs["triage"] = result

    return {
        "completed_agents": completed,
        "agent_outputs": outputs,
    }


def diagnostic_worker_node(state: SupervisorState) -> dict:
    """Diagnostic worker -- delegates to the shared DiagnosticAgent."""
    patient = state["patient_case"]
    context = "\n".join(
        f"{k}: {v[:150]}" for k, v in state.get("agent_outputs", {}).items()
    )

    result = diagnostic_agent.process_with_context(patient, context)
    print(f"    | [Diagnostic] {result[:120]}...")

    completed = list(state.get("completed_agents", []))
    completed.append("diagnostic")
    outputs = dict(state.get("agent_outputs", {}))
    outputs["diagnostic"] = result

    return {
        "completed_agents": completed,
        "agent_outputs": outputs,
    }


def pharmacist_worker_node(state: SupervisorState) -> dict:
    """Pharmacist worker -- delegates to the shared PharmacistAgent."""
    patient = state["patient_case"]
    context = "\n".join(
        f"{k}: {v[:150]}" for k, v in state.get("agent_outputs", {}).items()
    )

    result = pharmacist_agent.process_with_context(patient, context)
    print(f"    | [Pharmacist] {result[:120]}...")

    completed = list(state.get("completed_agents", []))
    completed.append("pharmacist")
    outputs = dict(state.get("agent_outputs", {}))
    outputs["pharmacist"] = result

    return {
        "completed_agents": completed,
        "agent_outputs": outputs,
    }


def report_node(state: SupervisorState) -> dict:
    """Generate final report from all agent outputs."""
    llm = get_llm()
    outputs = state.get("agent_outputs", {})

    all_findings = "\n\n".join(
        f"[{agent.upper()}]:\n{output}" for agent, output in outputs.items()
    )

    prompt = f"""Compile these specialist findings into a structured clinical summary:

{all_findings}

Format: 1) Triage  2) Diagnosis  3) Medication Review  4) Recommendations
Keep under 200 words."""

    config = build_callback_config(trace_name="supervisor_report")
    response = llm.invoke(prompt, config=config)
    print(f"    | [Report] Generated: {len(response.content)} chars")

    return {"final_report": response.content}


# ============================================================
# STAGE 1.4 -- Routing Logic
# ============================================================

def route_supervisor(state: SupervisorState) -> str:
    """Route from supervisor to the selected agent or report."""
    next_agent = state.get("next_agent", "FINISH")
    routing_map = {
        "triage": "triage_worker",
        "diagnostic": "diagnostic_worker",
        "pharmacist": "pharmacist_worker",
        "FINISH": "report",
    }
    return routing_map.get(next_agent, "report")


# ============================================================
# STAGE 1.5 -- Graph Construction
# ============================================================

def build_supervisor_graph():
    """
    Build the supervisor-orchestrated multi-agent graph.

    Key LangGraph patterns:
        - add_conditional_edges: supervisor routes dynamically
        - All workers return to supervisor via add_edge
        - Supervisor loop terminates via "FINISH" -> report
    """
    workflow = StateGraph(SupervisorState)

    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("triage_worker", triage_worker_node)
    workflow.add_node("diagnostic_worker", diagnostic_worker_node)
    workflow.add_node("pharmacist_worker", pharmacist_worker_node)
    workflow.add_node("report", report_node)

    # Entry
    workflow.add_edge(START, "supervisor")

    # Supervisor routes conditionally
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "triage_worker": "triage_worker",
            "diagnostic_worker": "diagnostic_worker",
            "pharmacist_worker": "pharmacist_worker",
            "report": "report",
        },
    )

    # All workers return to supervisor
    workflow.add_edge("triage_worker", "supervisor")
    workflow.add_edge("diagnostic_worker", "supervisor")
    workflow.add_edge("pharmacist_worker", "supervisor")

    # Report ends
    workflow.add_edge("report", END)

    return workflow.compile()


# ============================================================
# STAGE 1.6 -- Main
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  SUPERVISOR ORCHESTRATION")
    print("  Pattern: central coordinator routes to specialist agents")
    print("=" * 70)

    print("""
    Architecture:

        [Supervisor] --+--> [Triage]     --+
                       |                   |
                       +--> [Diagnostic]  --+--> [Supervisor] --> ...
                       |                   |
                       +--> [Pharmacist]  --+
                       |
                       +--> [Report] --> END

    The supervisor LLM decides agent order at runtime.
    Workers return to supervisor after each task.
    """)

    patient = PatientCase(
        patient_id="PT-ARCH-001",
        age=68, sex="M",
        chief_complaint="Chest pain and shortness of breath with elevated troponin",
        symptoms=["chest pain", "dyspnea", "diaphoresis", "nausea"],
        medical_history=["Hypertension", "Type 2 Diabetes", "Hyperlipidemia"],
        current_medications=["Lisinopril 20mg", "Metformin 1000mg BID", "Atorvastatin 40mg"],
        allergies=["Penicillin"],
        lab_results={"Troponin": "0.15 ng/mL", "BNP": "380 pg/mL", "HbA1c": "7.2%", "LDL": "128 mg/dL"},
        vitals={"BP": "158/95", "HR": "102", "SpO2": "94%", "Temp": "37.1C"},
    )

    initial_state = {
        "messages": [],
        "patient_case": patient.model_dump(),
        "next_agent": "",
        "completed_agents": [],
        "agent_outputs": {},
        "iteration": 0,
        "max_iterations": 5,
        "final_report": "",
    }

    print(f"    Patient: {patient.patient_id} | {patient.age}y {patient.sex}")
    print(f"    Complaint: {patient.chief_complaint}")
    print(f"    Agents: {list(AGENT_REGISTRY.keys())}")
    print()
    print("    " + "-" * 60)

    graph = build_supervisor_graph()
    result = graph.invoke(initial_state)

    # -- Display results ---------------------------------------------------
    print("\n    " + "=" * 60)
    print("    SUPERVISOR ROUTING TRACE")
    print("    " + "-" * 60)
    print(f"    Agents completed: {result['completed_agents']}")
    print(f"    Iterations: {result['iteration']}")

    print("\n    " + "=" * 60)
    print("    CLINICAL REPORT")
    print("    " + "-" * 60)
    for line in result.get("final_report", "").split("\n"):
        print(f"    | {line}")

    # -- Summary -----------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  SUPERVISOR ORCHESTRATION SUMMARY")
    print("=" * 70)
    print("""
    What you learned:

      1. Supervisor LLM dynamically decides routing order
      2. add_conditional_edges maps supervisor decisions to nodes
      3. All workers return to supervisor via add_edge
      4. Max iteration guard prevents infinite loops
      5. Agents are shared BaseAgent subclasses imported from agents/

    When to use:
      - Dynamic task ordering (supervisor adapts to results)
      - Centralized control and accountability
      - Complex workflows where agent order matters

    When NOT to use:
      - Fixed-order workflows (use sequential_pipeline instead)
      - Maximum parallelism needed (use map_reduce_fanout)

    Next: sequential_pipeline.py
    """)


if __name__ == "__main__":
    main()
