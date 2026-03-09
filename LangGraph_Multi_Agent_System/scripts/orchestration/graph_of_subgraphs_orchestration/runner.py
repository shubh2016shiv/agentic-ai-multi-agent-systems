#!/usr/bin/env python3
"""
============================================================
Graph-of-Subgraphs Orchestration — Runner
============================================================
PATTERN: 4 of 5
PREREQUISITE: Basic understanding of LangGraph state

------------------------------------------------------------
KEY CONCEPTS
------------------------------------------------------------
    - Encapsulating multi-step workflows as subgraphs
    - Using compiled StateGraphs as nodes in a parent graph
    - State translation via wrapper nodes
    - Abstraction of complexity

Pattern 4 of 5: Each specialty has its own multi-step workflow
(subgraph) that the parent graph orchestrates as an atomic unit.

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
In production systems, specialists don't just give a single
answer — they run a multi-step workflow:
    1. Initial assessment (gather findings)
    2. Risk analysis (score severity)
    3. Recommendation (propose actions)

LangGraph subgraphs let you encapsulate this multi-step
workflow as a single node in the parent graph. The parent
doesn't need to know the subgraph's internal structure.

This is ideal for:
    - Team-based workflows where each team has internal steps
    - Reusable specialty modules across different orchestrations
    - Clean separation between orchestration and specialist logic

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.orchestration.graph_of_subgraphs_orchestration.runner
============================================================
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# CONNECTION: orchestration/models.py — SHARED_PATIENT is the demo patient case
# used by all 5 patterns so learners can compare orchestration behavior on identical data.
from orchestration.models import SHARED_PATIENT
from scripts.orchestration.graph_of_subgraphs_orchestration.graph import build_graph_of_subgraphs


# ============================================================
# STAGE 4.4 — Main Execution
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  GRAPH-OF-SUBGRAPHS ORCHESTRATION")
    print("  Pattern 4/5: each specialty runs a 3-step subgraph")
    print("=" * 70)

    print("""
    Parent graph:

        [Pulm Subgraph] --> [Cardio Subgraph] --> [Nephro Subgraph] --> [Synthesis]

    Each subgraph internally:

        [Assessment] --> [Risk Analysis] --> [Recommendation]

    The parent sees 3 nodes. Each node runs 3 internal steps.
    Total: 9 specialist LLM calls + 1 synthesis = 10 LLM calls.
    """)

    patient = SHARED_PATIENT

    initial_state = {
        "messages": [],
        "patient_case": patient.model_dump(),
        "pulmonology_result": {},
        "cardiology_result": {},
        "nephrology_result": {},
        "final_report": "",
    }

    print(f"    Patient: {patient.patient_id} | {patient.age}y {patient.sex}")
    print(f"    Complaint: {patient.chief_complaint}")
    print()
    print("    " + "-" * 60)

    graph = build_graph_of_subgraphs()
    result = graph.invoke(initial_state)

    # -- Display subgraph results --------------------------------------
    for specialty in ["pulmonology", "cardiology", "nephrology"]:
        subgraph_result = result.get(f"{specialty}_result", {})
        print(f"\n    " + "=" * 60)
        print(f"    [{specialty.upper()} SUBGRAPH]")
        print(f"    " + "-" * 60)
        print(f"    Assessment:     {subgraph_result.get('assessment', '')[:150]}...")
        print(f"    Risk:           {subgraph_result.get('risk', '')[:100]}...")
        print(f"    Recommendation: {subgraph_result.get('recommendation', '')[:150]}...")

    # -- Display final report ------------------------------------------
    print(f"\n    " + "=" * 60)
    print("    SYNTHESIZED REPORT")
    print("    " + "-" * 60)
    for line in result.get("final_report", "").split("\n"):
        print(f"    | {line}")

    # -- Summary -------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  GRAPH-OF-SUBGRAPHS SUMMARY")
    print("=" * 70)
    print("""
    What you learned:

      1. Subgraphs encapsulate multi-step specialist workflows
      2. Parent graph treats each subgraph as an atomic node
      3. Wrapper nodes translate between parent and subgraph state
      4. Factory functions create reusable subgraph nodes
      5. Each specialty runs Assessment -> Risk -> Recommendation

    Design benefits:
      + Specialties are reusable, self-contained modules
      + Parent graph stays simple (3 nodes + synthesis)
      + Internal complexity hidden from the orchestrator
      + Each subgraph can be independently tested

    When to use:
      - Each specialist team has its own multi-step workflow
      - You need modular, reusable specialist components
      - Separation between orchestration and specialist logic

    Next: hybrid_orchestration
    """)


if __name__ == "__main__":
    main()
