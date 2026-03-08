#!/usr/bin/env python3
"""
============================================================
Supervisor Pattern Handoff
============================================================
Prerequisite: command_handoff.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Implement a supervisor LLM that sits above worker agents and
decides which worker runs next. Workers always return to the
supervisor. The supervisor routes conditionally until it
decides the task is complete.

This is the most structured multi-agent routing pattern.
Unlike command_handoff.py (peer-to-peer), here all routing
decisions flow through a single coordinator.

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    [START]
       |
       v
    [supervisor]  <-------+-------+
       |                  |       |
       |  "triage"        |       |
       +---------> [triage_worker] (always returns to supervisor)
       |
       |  "pharmacology"
       +---------> [pharma_worker]  (always returns to supervisor)
       |
       |  "FINISH"
       +---------> [report] -----> [END]

    Routing:  supervisor LLM picks the next worker via structured output.
    Who decides: THE SUPERVISOR LLM.
    Workers: execute their task and return to supervisor.

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. Supervisor node — LLM that routes to workers
    2. Worker-to-supervisor edges (all workers return to supervisor)
    3. add_conditional_edges from supervisor to workers
    4. Termination condition: supervisor says "FINISH"
    5. Tracking completed workers to avoid redundant calls

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.handoff.supervisor
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
from tools import (
    analyze_symptoms,
    assess_patient_risk,
    check_drug_interactions,
    lookup_drug_info,
    calculate_dosage_adjustment,
)
from observability.callbacks import build_callback_config


# ============================================================
# STAGE 4.1 — State Definition
# ============================================================

class SupervisorState(TypedDict):
    """
    State for the supervisor pattern.

    next_worker: which worker the supervisor wants to run next.
    completed_workers: list of workers that have already run.
    iteration: how many supervisor decisions have been made.
    """
    messages: Annotated[list, add_messages]
    patient_case: dict
    next_worker: str
    completed_workers: list[str]
    iteration: int
    max_iterations: int
    final_report: str


# ============================================================
# STAGE 4.2 — Build Graph
# ============================================================

WORKERS = ["triage", "pharmacology"]


def build_supervisor_pipeline():
    """Build a supervisor-coordinated multi-agent graph."""
    llm = get_llm()

    triage_tools = [analyze_symptoms, assess_patient_risk]
    pharma_tools = [check_drug_interactions, lookup_drug_info, calculate_dosage_adjustment]

    patient = PatientCase(
        patient_id="PT-SUP-001",
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

    # ── Node: supervisor ────────────────────────────────────────────────
    def supervisor_node(state: SupervisorState) -> dict:
        """
        Supervisor LLM decides which worker runs next.

        The supervisor sees:
          - The patient case summary
          - Which workers have already completed
          - The outputs from completed workers

        It returns one of: "triage", "pharmacology", or "FINISH".
        """
        print(f"\n    [STAGE 4.2] SUPERVISOR (iteration {state['iteration'] + 1})")
        print("    " + "-" * 50)

        completed = state.get("completed_workers", [])
        remaining = [w for w in WORKERS if w not in completed]

        print(f"    Completed workers: {completed}")
        print(f"    Remaining workers: {remaining}")

        # Guard: if all workers have run or max iterations reached, finish
        if not remaining or state["iteration"] >= state["max_iterations"]:
            print("    -> All workers done or max iterations. Routing to FINISH.")
            return {"next_worker": "FINISH", "iteration": state["iteration"] + 1}

        # Collect worker outputs for context
        worker_outputs = [
            msg.content for msg in state["messages"]
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls
        ]
        output_summary = "\n---\n".join(worker_outputs[-5:]) if worker_outputs else "No outputs yet."

        supervisor_llm = get_llm()
        config = build_callback_config(trace_name="handoff_supervisor_decide")

        prompt = f"""You are a clinical workflow supervisor. You coordinate
specialist agents to evaluate a patient.

PATIENT:
  {patient.age}y {patient.sex} — {patient.chief_complaint}
  Medications: {', '.join(patient.current_medications)}
  Labs: K+={patient.lab_results.get('K+')}, eGFR={patient.lab_results.get('eGFR')}

AVAILABLE WORKERS: {remaining}
COMPLETED WORKERS: {completed}

WORKER OUTPUTS SO FAR:
{output_summary}

Decide which worker should run next. Consider:
  - "triage" evaluates symptoms, identifies risks
  - "pharmacology" reviews drug interactions and dosing

Respond with EXACTLY one word: either the name of the next
worker ({', '.join(remaining)}) or "FINISH" if all necessary
work is done.

YOUR DECISION:"""

        response = supervisor_llm.invoke(prompt, config=config)
        decision = response.content.strip().lower()

        # Parse the decision
        if decision in remaining:
            next_worker = decision
        elif "finish" in decision:
            next_worker = "FINISH"
        elif remaining:
            # Fallback: pick the first remaining worker
            next_worker = remaining[0]
        else:
            next_worker = "FINISH"

        print(f"    Supervisor decision: {next_worker}")

        return {
            "next_worker": next_worker,
            "iteration": state["iteration"] + 1,
        }

    # ── Router function for supervisor ──────────────────────────────────
    def route_supervisor(state: SupervisorState) -> Literal["triage", "pharmacology", "report"]:
        """Map next_worker to node names."""
        nw = state.get("next_worker", "FINISH")
        if nw == "triage":
            return "triage"
        elif nw == "pharmacology":
            return "pharmacology"
        else:
            return "report"

    # ── Node: triage worker ─────────────────────────────────────────────
    def triage_worker_node(state: SupervisorState) -> dict:
        """
        Triage worker. Runs its assessment, then returns to supervisor.
        The return-to-supervisor happens via add_edge(), not via
        the worker's decision.
        """
        print("\n    [STAGE 4.3] TRIAGE WORKER")
        print("    " + "-" * 50)

        triage_llm = llm.bind_tools(triage_tools)
        system_msg = SystemMessage(content=(
            "You are a triage specialist. Evaluate the patient, "
            "identify urgent concerns, and flag issues."
        ))
        user_msg = HumanMessage(content=f"""Evaluate:
Patient: {patient.age}y {patient.sex} — {patient.chief_complaint}
Symptoms: {', '.join(patient.symptoms)}
Medications: {', '.join(patient.current_medications)}
Labs: {json.dumps(patient.lab_results)}
Vitals: {json.dumps(patient.vitals)}

Use your tools, then provide your triage assessment.""")

        config = build_callback_config(trace_name="handoff_supervisor_triage")
        messages = [system_msg, user_msg]
        response = triage_llm.invoke(messages, config=config)

        while hasattr(response, "tool_calls") and response.tool_calls:
            print(f"    | Triage calling {len(response.tool_calls)} tool(s)")
            tool_node = ToolNode(triage_tools)
            tool_results = tool_node.invoke({"messages": [response]})
            messages.extend([response] + tool_results["messages"])
            response = triage_llm.invoke(messages, config=config)

        print(f"    Triage assessment: {len(response.content)} chars")

        return {
            "messages": [response],
            "completed_workers": state.get("completed_workers", []) + ["triage"],
        }

    # ── Node: pharmacology worker ───────────────────────────────────────
    def pharma_worker_node(state: SupervisorState) -> dict:
        """Pharmacology worker. Returns to supervisor after completion."""
        print("\n    [STAGE 4.4] PHARMACOLOGY WORKER")
        print("    " + "-" * 50)

        pharma_llm = llm.bind_tools(pharma_tools)
        system_msg = SystemMessage(content=(
            "You are a clinical pharmacologist. Review medications "
            "for interactions and dosing safety."
        ))
        user_msg = HumanMessage(content=f"""Review this patient's medications:
Medications: {chr(10).join('  - ' + m for m in patient.current_medications)}
Labs: K+={patient.lab_results.get('K+')}, eGFR={patient.lab_results.get('eGFR')}
Allergies: {', '.join(patient.allergies)}

Use your tools, then provide recommendations.""")

        config = build_callback_config(trace_name="handoff_supervisor_pharma")
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
            "completed_workers": state.get("completed_workers", []) + ["pharmacology"],
        }

    # ── Node: report ────────────────────────────────────────────────────
    def report_node(state: SupervisorState) -> dict:
        """Generate final report when supervisor says FINISH."""
        print("\n    [STAGE 4.6] REPORT GENERATOR")
        print("    " + "-" * 50)

        agent_outputs = [
            msg.content for msg in state["messages"]
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls
        ]
        synthesis = "\n---\n".join(agent_outputs[-4:])

        report_llm = get_llm()
        config = build_callback_config(trace_name="handoff_supervisor_report")
        response = report_llm.invoke(
            f"Synthesise into a clinical summary (max 150 words):\n\n{synthesis}",
            config=config,
        )

        print(f"\n    Report ({len(response.content)} chars):")
        for line in response.content.split("\n")[:8]:
            print(f"    | {line}")

        return {"final_report": response.content}

    # ── STAGE 4.5 — Wire the graph ─────────────────────────────────────
    # Key wiring:
    #   1. Workers ALWAYS return to supervisor (add_edge).
    #   2. Supervisor routes conditionally (add_conditional_edges).
    #   3. Report goes to END.

    workflow = StateGraph(SupervisorState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("triage", triage_worker_node)
    workflow.add_node("pharmacology", pharma_worker_node)
    workflow.add_node("report", report_node)

    # Start at supervisor
    workflow.add_edge(START, "supervisor")

    # Workers always return to supervisor
    workflow.add_edge("triage", "supervisor")
    workflow.add_edge("pharmacology", "supervisor")

    # Supervisor routes conditionally
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {"triage": "triage", "pharmacology": "pharmacology", "report": "report"},
    )

    # Report goes to END
    workflow.add_edge("report", END)

    return workflow.compile(), patient


# ============================================================
# STAGE 4.7 — Execute
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  SUPERVISOR PATTERN HANDOFF")
    print("  Pattern: coordinator LLM dispatches to workers")
    print("=" * 70)

    print("""
    Graph topology:

        [START]
           |
           v
        [supervisor]  <--------+--------+
           |                   |        |
           +-- "triage" -----> [triage] |
           |                            |
           +-- "pharmacology" -------> [pharmacology]
           |
           +-- "FINISH" -----> [report] --> [END]

    Key edges:
      - supervisor -> workers: conditional (supervisor decides)
      - workers -> supervisor: fixed (add_edge, always return)
      - supervisor -> report: conditional ("FINISH" terminates)
    """)

    graph, patient = build_supervisor_pipeline()

    print(f"    Patient: {patient.patient_id} | {patient.age}y {patient.sex}")
    print(f"    K+={patient.lab_results['K+']} | eGFR={patient.lab_results['eGFR']}")
    print()

    initial_state: SupervisorState = {
        "messages": [],
        "patient_case": patient.model_dump(),
        "next_worker": "",
        "completed_workers": [],
        "iteration": 0,
        "max_iterations": 5,
        "final_report": "",
    }

    result = graph.invoke(initial_state)

    print("\n" + "=" * 70)
    print("  SUPERVISOR COMPLETE")
    print("=" * 70)
    print(f"    Completed workers : {result['completed_workers']}")
    print(f"    Iterations        : {result['iteration']}")

    print("""
    What happened:
      1. Supervisor LLM received the patient case.
      2. It decided which worker to dispatch first (triage or pharmacology).
      3. Each worker ran its assessment and returned to supervisor.
      4. Supervisor checked remaining workers and dispatched the next.
      5. When all workers completed, supervisor said "FINISH" -> report.

    Key difference from command_handoff.py:
      - command_handoff: peer-to-peer, agents hand off to each other
      - supervisor: centralised, all routing goes through coordinator

    Next: parallel_fanout.py — fan out to multiple agents in parallel.
    """)


if __name__ == "__main__":
    main()
