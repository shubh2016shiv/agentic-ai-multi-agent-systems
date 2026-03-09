#!/usr/bin/env python3
"""
============================================================
Hybrid Orchestration — Runner
============================================================
PATTERN: 5 of 5
PREREQUISITE: Basic understanding of LangGraph state

------------------------------------------------------------
KEY CONCEPTS
------------------------------------------------------------
    - Combining conditional routing with sequential clusters
    - Department-level routing (Supervisor)
    - Within-team collaboration (Peer-to-Peer)
    - Modeling real-world organizational structures

Pattern 5 of 5: Supervisor routes at department level,
P2P within the cardiopulmonary department.

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Real healthcare organizations use HYBRID coordination:
    - A Chief Medical Officer routes to departments
    - Within the Cardiopulmonary team, pulm and cardio
      specialists collaborate peer-to-peer
    - Nephrology runs as a standalone assessment

This combines the best of both worlds:
    - Supervisor: dynamic high-level routing
    - P2P: collaborative detail within teams

------------------------------------------------------------
WHEN TO USE
------------------------------------------------------------
    Use:
      - Multi-department organizations
      - When some teams need internal collaboration
      - When high-level routing is dynamic but team work is fixed
    Don't use:
      - Simple workflows (use P2P or pipeline)
      - When all agents are independent (use voting)

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.orchestration.hybrid_orchestration.runner
============================================================
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# CONNECTION: orchestration/models.py — SHARED_PATIENT is the demo patient case
# used by all 5 patterns so learners can compare orchestration behavior on identical data.
from orchestration.models import SHARED_PATIENT
from scripts.orchestration.hybrid_orchestration.graph import build_hybrid_graph


# ============================================================
# STAGE 5.4 — Main Execution
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  HYBRID ORCHESTRATION")
    print("  Pattern 5/5: supervisor + P2P cluster")
    print("=" * 70)

    print("""
    Architecture:

        [Supervisor] --+--> [Cardiopulm P2P Cluster]
                       |        |
                       |    [Pulm peer] --> [Cardio peer]
                       |        |
                       |        +--> back to supervisor
                       |
                       +--> [Nephrology]
                       |        |
                       |        +--> back to supervisor
                       |
                       +--> [Synthesis] --> END

    Mixes conditional edges (supervisor) with sequential
    edges (P2P within cardiopulmonary cluster).
    """)

    patient = SHARED_PATIENT

    initial_state = {
        "messages": [],
        "patient_case": patient.model_dump(),
        "supervisor_routing": "",
        "cardiopulmonary_findings": [],
        "renal_findings": "",
        "cluster_outputs": {},
        "routing_decisions": [],
        "iteration": 0,
        "max_iterations": 4,
        "final_report": "",
    }

    print(f"    Patient: {patient.patient_id} | {patient.age}y {patient.sex}")
    print(f"    Complaint: {patient.chief_complaint}")
    print()
    print("    " + "-" * 60)

    graph = build_hybrid_graph()
    result = graph.invoke(initial_state)

    # -- Display routing trace -----------------------------------------
    print("\n    " + "=" * 60)
    print("    ROUTING TRACE")
    print("    " + "-" * 60)
    print(f"    Decisions: {' -> '.join(result.get('routing_decisions', []))}")
    print(f"    Iterations: {result['iteration']}")

    # -- Display cluster outputs ----------------------------------------
    for cluster, output in result.get("cluster_outputs", {}).items():
        print(f"\n    [{cluster.upper()}]:")
        for line in output[:250].split("\n"):
            if line.strip():
                print(f"      {line}")

    # -- Display P2P findings -------------------------------------------
    if result.get("cardiopulmonary_findings"):
        print(f"\n    " + "=" * 60)
        print("    CARDIOPULMONARY P2P FINDINGS (accumulated)")
        print("    " + "-" * 60)
        for finding_index, finding in enumerate(result["cardiopulmonary_findings"]):
            print(f"    {finding_index + 1}. {finding[:150]}...")

    # -- Display report -------------------------------------------------
    print(f"\n    " + "=" * 60)
    print("    SYNTHESIZED REPORT")
    print("    " + "-" * 60)
    for line in result.get("final_report", "").split("\n"):
        print(f"    | {line}")

    # -- Summary --------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  HYBRID ORCHESTRATION SUMMARY")
    print("=" * 70)
    print("""
    What you learned:

      1. Supervisor routes at DEPARTMENT level (conditional edges)
      2. P2P cluster runs internally (sequential edges)
      3. Cardiopulmonary cluster: pulm -> cardio (buildig context)
      4. P2P cluster returns to supervisor after completion
      5. Combines dynamic routing with collaborative execution

    The 5 orchestration patterns compared:

      1. Supervisor:    loop-based, dynamic routing
      2. Peer-to-Peer:  no coordinator, shared findings
      3. Dynamic Router: one-shot classification, selective routing
      4. Subgraphs:     multi-step workflows per specialty
      5. Hybrid:        supervisor + P2P within departments

    Each pattern uses the SAME shared base (BaseOrchestrator,
    OrchestrationResult, SHARED_PATIENT) and the SAME 3
    specialties — demonstrating that orchestration logic,
    not agent logic, determines system behavior.
    """)


if __name__ == "__main__":
    main()
