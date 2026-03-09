#!/usr/bin/env python3
"""
============================================================
Hierarchical Delegation
============================================================
Pattern 5: Multi-level management structure where decisions
flow top-down and results flow bottom-up.

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
In large organizations (hospitals, enterprises), work is
decomposed into layers:

    Level 1: Executive (Chief Medical Officer)
        - Sets the strategic direction
        - Makes final decisions from team-lead summaries

    Level 2: Team Leads (Cardiac Lead, Renal Lead)
        - Coordinate their specialists
        - Summarize specialist findings for the executive

    Level 3: Specialists (our 3 clinical agents)
        - Execute specific tasks
        - Report results to their team lead

This mirrors real hospital org charts and is ideal when:
    - The problem naturally decomposes into management layers
    - Different levels need different granularity of information
    - Clear chain of command is needed for accountability

When NOT to use:
    - Flat team structures (use voting or pipeline)
    - When specialists need peer-to-peer communication (use debate)

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [level_3_specialists]  <-- triage, diagnostic, pharmacist
       |
       v
    [level_2_team_leads]   <-- cardiac lead, general medicine lead
       |
       v
    [level_1_executive]    <-- CMO makes final decision
       |
       v
    [END]

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    main()     L3_specialists    L2_team_leads     L1_executive
      |              |                |                  |
      |-- invoke --> |                |                  |
      |              |-- triage       |                  |
      |              |-- diagnostic   |                  |
      |              |-- pharmacist   |                  |
      |              |---- L3 results -->|               |
      |              |                |-- cardiac_lead   |
      |              |                |-- gen_med_lead   |
      |              |                |---- L2 summaries -->|
      |              |                |                  |-- CMO decision
      |<-- final decision --------------------------------|

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.MAS_architectures.hierarchical_delegation
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
# are the Level 3 specialist agents (component layer). This script demonstrates
# the HIERARCHICAL DELEGATION PATTERN (L3 → L2 leads → L1 executive), not agent
# implementation. See agents/ for what each agent does internally.
from agents import TriageAgent, PharmacistAgent, DiagnosticAgent
# CONNECTION: core/ root module — get_llm() centralises LLM config for Level 2
# team leads and Level 1 executive (which use raw LLM calls with system prompts).
# PatientCase is the canonical domain model passed through state.
from core.config import get_llm
from core.models import PatientCase
# CONNECTION: observability/ root module — build_callback_config() attaches
# Langfuse tracing to every LLM call across all three hierarchy levels.
from observability.callbacks import build_callback_config


# ============================================================
# STAGE 5.1 -- State Definition
# ============================================================

class HierarchyState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_case: dict

    # Level 3 outputs (specialist raw findings)
    specialist_outputs: dict

    # Level 2 outputs (team lead summaries)
    team_lead_outputs: dict

    # Level 1 output (executive decision)
    executive_decision: str


# ============================================================
# STAGE 5.2 -- Agent Instances (Level 3 specialists)
# ============================================================

triage_agent = TriageAgent()
diagnostic_agent = DiagnosticAgent()
pharmacist_agent = PharmacistAgent()


# ============================================================
# STAGE 5.3 -- Node Definitions
# ============================================================

def level_3_specialists_node(state: HierarchyState) -> dict:
    """
    Level 3: Specialist execution.

    All three clinical agents execute their assessments.
    Results are stored in specialist_outputs dict.
    In production, these could run in parallel.
    """
    patient = state["patient_case"]
    outputs = {}

    # Triage
    triage_result = triage_agent.process_with_context(patient)
    outputs["triage"] = triage_result
    print(f"    | [L3] Triage: {triage_result[:100]}...")

    # Diagnostic
    diagnostic_result = diagnostic_agent.process_with_context(patient)
    outputs["diagnostic"] = diagnostic_result
    print(f"    | [L3] Diagnostic: {diagnostic_result[:100]}...")

    # Pharmacist
    pharmacist_result = pharmacist_agent.process_with_context(patient)
    outputs["pharmacist"] = pharmacist_result
    print(f"    | [L3] Pharmacist: {pharmacist_result[:100]}...")

    return {"specialist_outputs": outputs}


def level_2_team_leads_node(state: HierarchyState) -> dict:
    """
    Level 2: Team Lead coordination.

    Team leads receive raw specialist outputs and provide
    summarized, actionable findings for the executive.

    Two team leads:
        - Cardiac Lead: oversees triage + diagnostic findings
        - General Medicine Lead: oversees pharmacist findings + overall
    """
    llm = get_llm()
    specialist_outputs = state.get("specialist_outputs", {})

    team_lead_outputs = {}

    # -- Cardiac Lead (oversees triage + diagnostic) -----------------------
    cardiac_input = (
        f"Triage Findings:\n{specialist_outputs.get('triage', '')}\n\n"
        f"Diagnostic Findings:\n{specialist_outputs.get('diagnostic', '')}"
    )
    cardiac_prompt = f"""You are the Cardiac Team Lead. Your specialists have reported:

{cardiac_input}

Provide a TEAM-LEVEL SUMMARY for the Chief Medical Officer:
1. Key cardiac findings
2. Risk assessment
3. Your team's recommendation
Keep under 100 words."""

    config = build_callback_config(trace_name="hierarchy_cardiac_lead", tags=["hierarchy", "L2"])
    cardiac_response = llm.invoke(cardiac_prompt, config=config)
    team_lead_outputs["cardiac_lead"] = cardiac_response.content
    print(f"    | [L2] Cardiac Lead: {cardiac_response.content[:100]}...")

    # -- General Medicine Lead (oversees pharmacist + overall) --------------
    gen_med_input = (
        f"Pharmacist Findings:\n{specialist_outputs.get('pharmacist', '')}\n\n"
        f"Triage Summary:\n{specialist_outputs.get('triage', '')[:200]}"
    )
    gen_med_prompt = f"""You are the General Medicine Team Lead. Your specialists have reported:

{gen_med_input}

Provide a TEAM-LEVEL SUMMARY for the Chief Medical Officer:
1. Medication concerns
2. Comorbidity management
3. Your team's recommendation
Keep under 100 words."""

    config = build_callback_config(trace_name="hierarchy_gen_med_lead", tags=["hierarchy", "L2"])
    gen_med_response = llm.invoke(gen_med_prompt, config=config)
    team_lead_outputs["general_medicine_lead"] = gen_med_response.content
    print(f"    | [L2] General Medicine Lead: {gen_med_response.content[:100]}...")

    return {"team_lead_outputs": team_lead_outputs}


def level_1_executive_node(state: HierarchyState) -> dict:
    """
    Level 1: Executive decision.

    The Chief Medical Officer receives ONLY team-lead summaries
    (not raw specialist outputs). This is deliberate:
        - Executives operate on aggregated information
        - Detail filtering happens at L2
        - Decision at L1 is strategic, not tactical
    """
    llm = get_llm()
    patient = state["patient_case"]
    team_lead_outputs = state.get("team_lead_outputs", {})

    team_reports = "\n\n".join(
        f"[{lead.upper()}]:\n{summary}"
        for lead, summary in team_lead_outputs.items()
    )

    executive_prompt = f"""You are the Chief Medical Officer. Your team leads report:

{team_reports}

Patient: {patient.get('age')}y {patient.get('sex')}, {patient.get('chief_complaint')}

Make the EXECUTIVE DECISION:
1. DECISION: What is the clinical plan?
2. PRIORITY: What must happen first?
3. ESCALATION: Does this case need external consultation?
4. FOLLOW-UP: Timeline for reassessment

This is a strategic decision, not a detailed plan. Keep under 120 words."""

    config = build_callback_config(trace_name="hierarchy_cmo_decision", tags=["hierarchy", "L1"])
    response = llm.invoke(executive_prompt, config=config)
    print(f"    | [L1] CMO: {response.content[:120]}...")

    return {"executive_decision": response.content}


# ============================================================
# STAGE 5.4 -- Graph Construction
# ============================================================

def build_hierarchy_graph():
    """
    Build the three-level hierarchical graph.

    Bottom-up information flow:
        L3 (details) -> L2 (summaries) -> L1 (decision)

    Each level filters and aggregates information for the next.
    """
    workflow = StateGraph(HierarchyState)

    workflow.add_node("level_3_specialists", level_3_specialists_node)
    workflow.add_node("level_2_team_leads", level_2_team_leads_node)
    workflow.add_node("level_1_executive", level_1_executive_node)

    workflow.add_edge(START, "level_3_specialists")
    workflow.add_edge("level_3_specialists", "level_2_team_leads")
    workflow.add_edge("level_2_team_leads", "level_1_executive")
    workflow.add_edge("level_1_executive", END)

    return workflow.compile()


# ============================================================
# STAGE 5.5 -- Main
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  HIERARCHICAL DELEGATION")
    print("  Pattern: multi-level management (L3 -> L2 -> L1)")
    print("=" * 70)

    print("""
    Architecture:

        Level 1 (Strategic):   [CMO]
                                 |
        Level 2 (Coordination): [Cardiac Lead]  [Gen Med Lead]
                                 |        |      |         |
        Level 3 (Execution):   [Triage] [Diag] [Pharma]  ...

    Information flows bottom-up: details -> summaries -> decisions.
    Each level filters complexity for the next level up.
    """)

    patient = PatientCase(
        patient_id="PT-ARCH-005",
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
        "specialist_outputs": {},
        "team_lead_outputs": {},
        "executive_decision": "",
    }

    print(f"    Patient: {patient.patient_id} | {patient.age}y {patient.sex}")
    print(f"    Complaint: {patient.chief_complaint}")
    print()
    print("    " + "-" * 60)

    graph = build_hierarchy_graph()
    result = graph.invoke(initial_state)

    # -- Display results by level ------------------------------------------
    print("\n    " + "=" * 60)
    print("    LEVEL 3: SPECIALIST FINDINGS")
    print("    " + "-" * 60)
    for name, output in result.get("specialist_outputs", {}).items():
        print(f"    [{name.upper()}]: {output[:150]}...")
        print()

    print("    " + "=" * 60)
    print("    LEVEL 2: TEAM LEAD SUMMARIES")
    print("    " + "-" * 60)
    for name, output in result.get("team_lead_outputs", {}).items():
        print(f"    [{name.upper()}]: {output[:200]}...")
        print()

    print("    " + "=" * 60)
    print("    LEVEL 1: EXECUTIVE DECISION")
    print("    " + "-" * 60)
    for line in result.get("executive_decision", "").split("\n"):
        print(f"    | {line}")

    # -- Summary -----------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  HIERARCHICAL DELEGATION SUMMARY")
    print("=" * 70)
    print("""
    What you learned:

      1. Three levels: Execution (L3) -> Coordination (L2) -> Decision (L1)
      2. Each level FILTERS information for the level above
      3. The CMO never sees raw specialist outputs (only summaries)
      4. Same BaseAgent subclasses at L3, LLM prompts at L2 and L1
      5. Mirrors real hospital organizational structures

    When to use:
      - When the problem naturally decomposes into management layers
      - Large teams needing clear information hierarchy
      - Accountability and audit trail are critical

    When NOT to use:
      - Flat, small teams (unnecessary overhead)
      - When specialists need peer-to-peer communication

    Next: map_reduce_fanout.py
    """)


if __name__ == "__main__":
    main()
