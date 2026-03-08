#!/usr/bin/env python3
"""
============================================================
Parallel Voting
============================================================
Pattern 3: All agents independently assess the same case,
then an aggregator determines consensus.

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
In high-stakes decisions (diagnosis, treatment eligibility),
relying on one agent creates single-point-of-failure risk.
The voting pattern mitigates this:

    1. Each agent independently assesses the SAME input
    2. No agent sees another agent's output (prevents anchoring)
    3. An aggregator compares outputs and determines consensus
    4. Disagreements are flagged for human review

This is ideal when:
    - Decision quality matters more than speed
    - You want to detect when agents disagree (uncertainty signal)
    - Independent perspectives reduce individual agent bias

When NOT to use:
    - When agents genuinely need each other's output (use pipeline)
    - When you need a single authoritative decision-maker (use supervisor)

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [coordinator]    <-- fans out to 3 specialists via Send API
       |
       +---> [specialist] (triage instance)      --+
       +---> [specialist] (diagnostic instance)   --+--> [aggregator]
       +---> [specialist] (pharmacist instance)   --+        |
                                                             v
                                                           [END]

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    main()       coordinator      specialist(x3)     aggregator
      |              |                |                  |
      |-- invoke --> |                |                  |
      |              |-- Send(triage) |                  |
      |              |-- Send(diag)   |                  |
      |              |-- Send(pharma) |                  |
      |              |            [parallel execution]   |
      |              |       triage --|                   |
      |              |       diag  --|                   |
      |              |       pharma--|                   |
      |              |               +-- results ------->|
      |              |               |   (merged via     |
      |              |               |    operator.add)  |-- LLM(consensus)
      |<-- consensus + report -------|-------------------|

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.MAS_architectures.parallel_voting
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
# NOTE: Agents are imported from the ROOT agents/ package (the "library layer").
#       This script is a "pattern demo layer" — it fans out to agents in parallel
#       but does NOT define its own agents. See agents/ for implementations.
from core.config import get_llm
from core.models import PatientCase
from observability.callbacks import build_callback_config
from agents import TriageAgent, PharmacistAgent, DiagnosticAgent


# ============================================================
# STAGE 3.1 -- State Definitions
# ============================================================

class SpecialistInput(TypedDict):
    """Input passed to each parallel specialist instance."""
    specialist_type: str
    patient_case: dict


class VotingState(TypedDict):
    """
    Main state for the voting graph.

    specialist_results uses operator.add as a reducer.
    When multiple specialist instances write to this field,
    their results are concatenated into a single list
    (not overwritten).
    """
    patient_case: dict
    specialist_results: Annotated[list[dict], operator.add]
    consensus_report: str
    agreement_score: float


# ============================================================
# STAGE 3.2 -- Agent Instances
# ============================================================

AGENT_MAP = {
    "triage": TriageAgent(),
    "diagnostic": DiagnosticAgent(),
    "pharmacist": PharmacistAgent(),
}


# ============================================================
# STAGE 3.3 -- Node Definitions
# ============================================================

def coordinator_node(state: VotingState) -> list[Send]:
    """
    Fan out the patient case to all specialists in parallel.

    Returns a list of Send objects. Each Send creates a parallel
    instance of the "specialist" node with its own input.
    LangGraph runs all instances concurrently.
    """
    patient = state["patient_case"]

    sends = []
    for specialist_type in AGENT_MAP.keys():
        sends.append(
            Send(
                "specialist",
                SpecialistInput(
                    specialist_type=specialist_type,
                    patient_case=patient,
                ),
            )
        )

    print(f"    | [Coordinator] Fan-out: {len(sends)} specialists dispatched in parallel")
    return sends


def specialist_node(state: SpecialistInput) -> dict:
    """
    A specialist agent that runs in parallel.

    Multiple instances of this node run concurrently.
    Each instance receives a SpecialistInput with:
        - specialist_type: which agent to use
        - patient_case: the patient data

    The result is written to specialist_results using operator.add,
    so all parallel results merge into a single list.
    """
    specialist_type = state["specialist_type"]
    patient = state["patient_case"]

    agent = AGENT_MAP[specialist_type]
    result = agent.process_with_context(patient)

    print(f"    | [Specialist:{specialist_type}] Assessment: {result[:100]}...")

    return {
        "specialist_results": [
            {
                "agent": specialist_type,
                "assessment": result,
            }
        ],
    }


def aggregator_node(state: VotingState) -> dict:
    """
    Aggregator -- collects all specialist results and determines consensus.

    By the time this node runs, specialist_results contains
    outputs from ALL parallel instances (merged by operator.add).

    The aggregator:
        1. Presents all assessments to a judge LLM
        2. Identifies areas of agreement and disagreement
        3. Produces a consensus report with an agreement score
    """
    llm = get_llm()
    results = state.get("specialist_results", [])

    # Format all assessments for the judge
    assessments = "\n\n".join(
        f"[{r['agent'].upper()} ASSESSMENT]:\n{r['assessment']}"
        for r in results
    )

    prompt = f"""You are a clinical consensus judge. Three independent specialists have assessed the same patient case WITHOUT seeing each other's work.

{assessments}

Determine:
1. CONSENSUS: What do all specialists agree on? (key findings)
2. DISAGREEMENTS: Where do they differ? (flag for human review)
3. AGREEMENT SCORE: Rate agreement from 0.0 (total disagreement) to 1.0 (complete agreement)
4. FINAL RECOMMENDATION: Based on the consensus

Respond with the analysis. Include the agreement score as a number on its own line starting with "Agreement: "
Keep under 200 words."""

    config = build_callback_config(
        trace_name="voting_aggregator",
        tags=["mas_architecture", "voting"],
    )
    response = llm.invoke(prompt, config=config)

    # Parse agreement score
    agreement_score = 0.7  # default
    for line in response.content.split("\n"):
        if line.strip().lower().startswith("agreement"):
            try:
                score_text = line.split(":")[-1].strip().rstrip(".")
                agreement_score = float(score_text)
            except (ValueError, IndexError):
                pass

    print(f"    | [Aggregator] Consensus generated, agreement: {agreement_score:.0%}")

    return {
        "consensus_report": response.content,
        "agreement_score": agreement_score,
    }


# ============================================================
# STAGE 3.4 -- Graph Construction
# ============================================================

def build_voting_graph():
    """
    Build the parallel voting graph with Send API.

    Key LangGraph patterns:
        - coordinator returns list[Send] (fan-out)
        - specialist runs in parallel (multiple instances)
        - operator.add reducer merges parallel results
        - aggregator runs after ALL parallel instances complete
    """
    workflow = StateGraph(VotingState)

    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("specialist", specialist_node)
    workflow.add_node("aggregator", aggregator_node)

    workflow.add_edge(START, "coordinator")
    # coordinator returns Send objects -> specialist instances
    # specialist results auto-merge via operator.add reducer
    workflow.add_edge("specialist", "aggregator")
    workflow.add_edge("aggregator", END)

    return workflow.compile()


# ============================================================
# STAGE 3.5 -- Main
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  PARALLEL VOTING")
    print("  Pattern: independent assessment + consensus aggregation")
    print("=" * 70)

    print("""
    Architecture:

        [Coordinator] --+--> [Triage]      --+
                        +--> [Diagnostic]  --+--> [Aggregator] --> END
                        +--> [Pharmacist]  --+

    Key difference from pipeline:
      - Agents do NOT see each other's work (prevents anchoring bias)
      - All run in PARALLEL (faster than pipeline)
      - Aggregator detects agreement/disagreement
    """)

    patient = PatientCase(
        patient_id="PT-ARCH-003",
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
        "specialist_results": [],
        "consensus_report": "",
        "agreement_score": 0.0,
    }

    print(f"    Patient: {patient.patient_id} | {patient.age}y {patient.sex}")
    print(f"    Complaint: {patient.chief_complaint}")
    print(f"    Voters: {list(AGENT_MAP.keys())}")
    print()
    print("    " + "-" * 60)

    graph = build_voting_graph()
    result = graph.invoke(initial_state)

    # -- Display individual assessments ------------------------------------
    print("\n    " + "=" * 60)
    print("    INDIVIDUAL ASSESSMENTS")
    print("    " + "-" * 60)
    for assessment in result.get("specialist_results", []):
        agent_name = assessment["agent"]
        text = assessment["assessment"][:200]
        print(f"    [{agent_name.upper()}]:")
        for line in text.split("\n"):
            if line.strip():
                print(f"      {line}")
        print()

    # -- Display consensus -------------------------------------------------
    print("    " + "=" * 60)
    print(f"    CONSENSUS (Agreement: {result.get('agreement_score', 0):.0%})")
    print("    " + "-" * 60)
    for line in result.get("consensus_report", "").split("\n"):
        print(f"    | {line}")

    # -- Summary -----------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  PARALLEL VOTING SUMMARY")
    print("=" * 70)
    print("""
    What you learned:

      1. Send API fans out patient data to parallel specialist instances
      2. operator.add reducer merges parallel results into one list
      3. Specialist instances run concurrently (true parallelism)
      4. Aggregator receives ALL results after all instances complete
      5. Agreement score quantifies inter-agent consistency

    When to use:
      - High-stakes decisions (diagnosis, treatment eligibility)
      - When independent perspectives reduce bias
      - When detecting disagreement is itself valuable

    When NOT to use:
      - When agents need each other's output (use pipeline)
      - When one authoritative decision is preferred (use supervisor)

    Next: adversarial_debate.py
    """)


if __name__ == "__main__":
    main()
