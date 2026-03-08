#!/usr/bin/env python3
"""
============================================================
Linear Pipeline Handoff
============================================================
Prerequisite: script_03_tools_and_handoffs.py (Sections 1-2)

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Demonstrate the simplest handoff pattern: a fixed-edge
pipeline where the developer decides the execution order
at graph-build time. No routing logic, no LLM-driven
decisions — just add_edge() from one node to the next.

This is the baseline pattern. Every other handoff script
in this directory builds on top of what you learn here.

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    Stage 1.1       Stage 1.2           Stage 1.3         Stage 1.4
    Define State    Triage Agent        Pharmacology      Report
                                        Agent             Generator

    [START]
       |
       v
    [triage]  ──add_edge()──>  [pharmacology]  ──add_edge()──>  [report]
       |                            |                              |
       | writes:                    | reads:                       | reads:
       |  - messages                |  - handoff_context           |  - messages
       |  - handoff_context         | writes:                      | writes:
       |  - handoff_depth           |  - messages                  |  - final_report
       |                            |  - handoff_depth             |
       v                            v                              v
                                                                 [END]

    Routing:  ALL edges are add_edge() — fixed at build time.
    Who decides next agent: YOU (the developer).
    LLM influence on routing: NONE.

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. HandoffContext as a first-class data structure
       (sender builds it, receiver reads it)
    2. Fixed-edge wiring with add_edge()
    3. ToolNode.invoke() for manual tool execution inside a node
    4. handoff_depth as a tracking counter
    5. Partial state updates — each node returns only what it changes

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.handoff.linear_pipeline
============================================================
"""

# ── Standard library ────────────────────────────────────────────────────────
import sys
import json
from typing import TypedDict, Annotated

# ── Encoding fix for Windows console ────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Path setup ──────────────────────────────────────────────────────────────

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
# STAGE 1.1 — State Definition
# ============================================================
# One TypedDict holds the full shared state. Every node reads
# from and writes to this same dict. Separation of concerns
# comes from WHICH FIELDS each node touches, not from having
# separate state classes.

class PipelineState(TypedDict):
    """
    Shared state for the linear pipeline.

    Fields
    ------
    messages : list
        Full message history across all agents.
        Annotated with add_messages so new messages are APPENDED.

    patient_case : dict
        Serialised PatientCase. Available to every node.

    handoff_context : dict
        The most recent HandoffContext, written by the SENDER
        and read by the RECEIVER. This is the scoped data packet
        that prevents context bleed between agents.

    current_agent : str
        Name of the agent that just executed. For logging.

    handoff_history : list[str]
        Ordered list of agents that have run. Provides a
        complete audit trail of the handoff chain.

    handoff_depth : int
        Counter incremented by each agent. Acts as a loop
        prevention metric (used in multihop_depth_guard.py).

    final_report : str
        The synthesised summary produced by the report node.
    """
    messages: Annotated[list, add_messages]
    patient_case: dict
    handoff_context: dict
    current_agent: str
    handoff_history: list[str]
    handoff_depth: int
    final_report: str


# ============================================================
# STAGE 1.2 — Triage Agent Node
# ============================================================
# The triage agent:
#   1. Evaluates the patient's presentation using triage tools.
#   2. Builds a HandoffContext with SCOPED findings.
#   3. Stores the HandoffContext in state for the next agent.
#
# It does NOT decide who runs next — the graph edge does that.

def build_pipeline():
    """
    Build and return the compiled linear pipeline graph.

    All node functions are defined inside this builder so they
    share the same LLM instance via closure. This avoids
    redundant client setup.
    """
    llm = get_llm()

    triage_tools = [analyze_symptoms, assess_patient_risk]
    pharma_tools = [check_drug_interactions, lookup_drug_info, calculate_dosage_adjustment]

    triage_llm = llm.bind_tools(triage_tools)
    pharma_llm = llm.bind_tools(pharma_tools)

    # ── Patient case (shared by closure) ────────────────────────────────
    patient = PatientCase(
        patient_id="PT-2026-LP01",
        age=71,
        sex="F",
        chief_complaint="Dizziness and fatigue after recent medication change",
        symptoms=["dizziness", "fatigue", "headache", "bilateral ankle edema"],
        medical_history=["Hypertension", "CKD Stage 3a", "Type 2 Diabetes"],
        current_medications=[
            "Lisinopril 20mg daily",
            "Spironolactone 25mg daily",
            "Metformin 1000mg BID",
            "Amlodipine 10mg daily",
        ],
        allergies=["Sulfa drugs"],
        lab_results={
            "eGFR": "42 mL/min",
            "K+": "5.4 mEq/L",
            "Cr": "1.6 mg/dL",
        },
        vitals={
            "BP": "105/65",
            "HR": "88",
            "SpO2": "95%",
        },
    )

    # ── Node: triage ────────────────────────────────────────────────────
    def triage_node(state: PipelineState) -> dict:
        """
        Evaluate the patient and produce a triage assessment.

        Steps:
            1. Build a system prompt scoped to triage only.
            2. Call the LLM with triage tools bound.
            3. Execute any tool calls (manual ReAct loop).
            4. Construct a HandoffContext for the next agent.
            5. Return a partial state update.
        """
        print("\n    [STAGE 1.2] TRIAGE AGENT")
        print("    " + "-" * 50)

        system_msg = SystemMessage(content=(
            "You are a triage specialist. Evaluate the patient's "
            "presentation, identify urgent concerns, and flag issues "
            "that require specialist review. Do not prescribe or "
            "adjust medications."
        ))

        user_msg = HumanMessage(content=f"""Evaluate this patient:

Patient: {patient.age}y {patient.sex}
Chief Complaint: {patient.chief_complaint}
Symptoms: {', '.join(patient.symptoms)}
Medications: {', '.join(patient.current_medications)}
Key Labs: K+ = {patient.lab_results.get('K+')}, eGFR = {patient.lab_results.get('eGFR')}
Vitals: BP = {patient.vitals.get('BP')}, HR = {patient.vitals.get('HR')}

Use your tools to analyze symptoms and assess risk, then provide
your triage assessment. Flag issues for pharmacology review.""")

        config = build_callback_config(trace_name="handoff_linear_triage")
        messages = [system_msg, user_msg]
        response = triage_llm.invoke(messages, config=config)

        # Manual tool loop — execute tool calls until the LLM
        # produces a final text response.
        while hasattr(response, "tool_calls") and response.tool_calls:
            print(f"    | Triage calling {len(response.tool_calls)} tool(s):")
            for tc in response.tool_calls:
                print(f"      -> {tc['name']}({json.dumps(tc['args'])[:80]}...)")

            tool_node = ToolNode(triage_tools)
            tool_results = tool_node.invoke({"messages": [response]})
            messages.extend([response] + tool_results["messages"])
            response = triage_llm.invoke(messages, config=config)

        triage_text = response.content
        print(f"\n    Triage assessment ({len(triage_text)} chars):")
        for line in triage_text.split("\n")[:6]:
            print(f"    | {line}")
        if len(triage_text.split("\n")) > 6:
            print(f"    | ... ({len(triage_text.split(chr(10))) - 6} more lines)")

        # ── Build HandoffContext ────────────────────────────────────────
        # The SENDER (triage) constructs this. The RECEIVER (pharmacology)
        # reads it from state["handoff_context"]. The receiver never
        # constructs the handoff it receives.
        handoff = HandoffContext(
            from_agent="TriageAgent",
            to_agent="PharmacologyAgent",
            reason=(
                "Patient on dual potassium-raising agents (Lisinopril + "
                "Spironolactone) with K+ 5.4 mEq/L. BP 105/65 despite "
                "multiple antihypertensives. Pharmacology review required."
            ),
            patient_case=patient,
            task_description=(
                "1. Check drug-drug interactions in the medication list. "
                "2. Assess Metformin safety at eGFR 42 mL/min. "
                "3. Recommend specific dose changes or substitutions."
            ),
            relevant_findings=[
                f"Hyperkalemia risk: K+ {patient.lab_results['K+']} with Lisinopril + Spironolactone",
                f"Possible hypotension from over-treatment: BP {patient.vitals['BP']}",
                f"Metformin safety concern: eGFR {patient.lab_results['eGFR']}",
                "Sulfa allergy on record — Spironolactone is NOT a sulfa drug",
            ],
            handoff_depth=state["handoff_depth"] + 1,
        )

        print(f"\n    HandoffContext built:")
        print(f"      from: {handoff.from_agent} -> to: {handoff.to_agent}")
        print(f"      scoped findings: {len(handoff.relevant_findings)}")

        return {
            "messages": [response],
            "handoff_context": handoff.model_dump(),
            "current_agent": "triage",
            "handoff_history": state["handoff_history"] + ["triage"],
            "handoff_depth": state["handoff_depth"] + 1,
        }

    # ============================================================
    # STAGE 1.3 — Pharmacology Agent Node
    # ============================================================
    # The pharmacology agent:
    #   1. Reads the HandoffContext from state (built by triage).
    #   2. Uses ONLY the scoped findings — not the full message history.
    #   3. Runs its own tool loop with pharmacology tools.
    #
    # Why read HandoffContext instead of state["messages"]?
    #   - state["messages"] contains triage's internal reasoning,
    #     tool call arguments, and raw tool results — noise for
    #     a pharmacologist.
    #   - HandoffContext contains only the curated findings that
    #     the pharmacologist needs. This is context scoping.

    def pharmacology_node(state: PipelineState) -> dict:
        """
        Review medications for interactions and dosing safety.

        Reads the HandoffContext written by the triage node.
        Does NOT re-read the triage agent's message history.
        """
        print("\n    [STAGE 1.3] PHARMACOLOGY AGENT")
        print("    " + "-" * 50)

        # Read the HandoffContext from state
        raw_handoff = state.get("handoff_context", {})
        handoff = HandoffContext(**raw_handoff) if raw_handoff else None

        if handoff:
            print(f"    Received HandoffContext from {handoff.from_agent}")
            print(f"    Reason: {handoff.reason[:80]}...")
            print(f"    Scoped findings: {len(handoff.relevant_findings)}")
            for f in handoff.relevant_findings:
                print(f"      - {f}")
            handoff_reason = handoff.reason
            handoff_task = handoff.task_description
            handoff_findings = handoff.relevant_findings
        else:
            print("    WARNING: No HandoffContext found. Using raw patient data.")
            handoff_reason = "Direct pharmacology review requested."
            handoff_task = "Review all medications."
            handoff_findings = []

        system_msg = SystemMessage(content=(
            "You are a clinical pharmacologist. Review medication "
            "regimens for drug-drug interactions, renal dosing, and "
            "safety. Provide specific, actionable recommendations."
        ))

        # Prompt built from HandoffContext fields — not from messages
        user_msg = HumanMessage(content=f"""You received a HANDOFF from the Triage Agent.

HANDOFF REASON:
{handoff_reason}

YOUR TASK:
{handoff_task}

SCOPED FINDINGS FROM TRIAGE:
{json.dumps(handoff_findings, indent=2)}

PATIENT MEDICATIONS:
{chr(10).join('  - ' + med for med in patient.current_medications)}

KEY LABS:
  K+   : {patient.lab_results.get('K+')}
  eGFR : {patient.lab_results.get('eGFR')}
  Cr   : {patient.lab_results.get('Cr')}

Use your tools to check drug interactions and dosing. Then provide:
  1. Drug interaction findings (severity: critical / moderate / minor)
  2. Renal dosing recommendations
  3. Specific medication changes with alternatives""")

        config = build_callback_config(trace_name="handoff_linear_pharma")
        messages = [system_msg, user_msg]
        response = pharma_llm.invoke(messages, config=config)

        while hasattr(response, "tool_calls") and response.tool_calls:
            print(f"    | Pharmacology calling {len(response.tool_calls)} tool(s):")
            for tc in response.tool_calls:
                print(f"      -> {tc['name']}({json.dumps(tc['args'])[:80]}...)")

            tool_node = ToolNode(pharma_tools)
            tool_results = tool_node.invoke({"messages": [response]})
            messages.extend([response] + tool_results["messages"])
            response = pharma_llm.invoke(messages, config=config)

        print(f"\n    Pharmacology recommendation ({len(response.content)} chars):")
        for line in response.content.split("\n")[:6]:
            print(f"    | {line}")
        if len(response.content.split("\n")) > 6:
            print(f"    | ... ({len(response.content.split(chr(10))) - 6} more lines)")

        return {
            "messages": [response],
            "current_agent": "pharmacology",
            "handoff_history": state["handoff_history"] + ["pharmacology"],
            "handoff_depth": state["handoff_depth"] + 1,
        }

    # ============================================================
    # STAGE 1.4 — Report Generator Node
    # ============================================================
    # No tools. Reads the last few AIMessage outputs and
    # synthesises them into a clinical summary.
    #
    # Deliberately separate from the specialist agents:
    #   - Specialists are domain experts.
    #   - Synthesis is a different skill.
    # Mixing them makes each specialist harder to test and reuse.

    def report_node(state: PipelineState) -> dict:
        """
        Synthesise specialist findings into a clinical summary.

        Reads only the final AIMessage content from each agent —
        skips tool messages and intermediate reasoning.
        """
        print("\n    [STAGE 1.4] REPORT GENERATOR")
        print("    " + "-" * 50)

        agent_outputs = [
            msg.content
            for msg in state["messages"]
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls
        ]

        synthesis_input = "\n--- Next specialist ---\n".join(agent_outputs[-3:])

        prompt = f"""Synthesise these specialist assessments into a concise
clinical action summary.

SPECIALIST FINDINGS:
{synthesis_input}

FORMAT:
1. Key Findings (2-3 bullet points, most urgent first)
2. Immediate Actions Required (numbered, specific)
3. Follow-up Plan (monitoring, repeat labs, timeline)

Maximum 200 words. Be specific."""

        config = build_callback_config(trace_name="handoff_linear_report")
        report_llm = get_llm()
        response = report_llm.invoke(prompt, config=config)

        print("\n    " + "=" * 50)
        print("    FINAL CLINICAL REPORT")
        print("    " + "=" * 50)
        for line in response.content.split("\n"):
            print(f"    | {line}")

        return {"final_report": response.content}

    # ============================================================
    # STAGE 1.5 — Wire the Graph
    # ============================================================
    # All edges are add_edge() — fixed, unconditional transitions.
    # After triage completes, pharmacology ALWAYS runs.
    # After pharmacology completes, report ALWAYS runs.
    # No branching, no routing function, no LLM involvement
    # in the sequencing decision.

    workflow = StateGraph(PipelineState)

    workflow.add_node("triage", triage_node)
    workflow.add_node("pharmacology", pharmacology_node)
    workflow.add_node("report", report_node)

    workflow.add_edge(START, "triage")
    workflow.add_edge("triage", "pharmacology")
    workflow.add_edge("pharmacology", "report")
    workflow.add_edge("report", END)

    return workflow.compile(), patient


# ============================================================
# STAGE 1.6 — Execute
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  LINEAR PIPELINE HANDOFF")
    print("  Pattern: fixed edges, developer-defined execution order")
    print("=" * 70)

    print("""
    Graph topology:

        [START]
           |
           v
        [triage]          <- triage tools: analyze_symptoms, assess_patient_risk
           |
           | (add_edge — unconditional)
           v
        [pharmacology]    <- pharma tools: check_drug_interactions, lookup_drug_info,
           |                               calculate_dosage_adjustment
           | (add_edge — unconditional)
           v
        [report]          <- no tools, synthesises agent outputs
           |
           v
         [END]
    """)

    graph, patient = build_pipeline()

    print(f"    Patient: {patient.patient_id} | {patient.age}y {patient.sex}")
    print(f"    Complaint: {patient.chief_complaint}")
    print(f"    Key risk: K+={patient.lab_results['K+']} with ACEi + MRA")
    print()

    initial_state: PipelineState = {
        "messages": [],
        "patient_case": patient.model_dump(),
        "handoff_context": {},
        "current_agent": "none",
        "handoff_history": [],
        "handoff_depth": 0,
        "final_report": "",
    }

    result = graph.invoke(initial_state)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"    Handoff chain  : {' -> '.join(result['handoff_history'])}")
    print(f"    Final depth    : {result['handoff_depth']}")
    print(f"    Messages total : {len(result['messages'])}")

    print("""
    What happened:
      1. Triage agent evaluated the patient with triage-scoped tools.
      2. Triage built a HandoffContext with scoped findings.
      3. Pharmacology read the HandoffContext (not the full message history).
      4. Report generator synthesised both agents' outputs.

    What to notice:
      - All edges were add_edge() — no routing decision at runtime.
      - HandoffContext carried only what the pharmacologist needed.
      - Each node returned a PARTIAL state update.

    Next: conditional_routing.py — add branching based on a router function.
    """)


if __name__ == "__main__":
    main()
