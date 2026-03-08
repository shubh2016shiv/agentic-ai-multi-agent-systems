#!/usr/bin/env python3
"""
============================================================
Peer-to-Peer Orchestration — Runner
============================================================
PATTERN: 2 of 5
PREREQUISITE: Basic understanding of LangGraph state

------------------------------------------------------------
KEY CONCEPTS
------------------------------------------------------------
    - Decentralized collaboration (no supervisor)
    - Accumulating context via a shared state list
    - Linear execution pipelines without loops

Pattern 2 of 5: No central coordinator. Agents share findings
via a common state list, each building on prior work.

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
The P2P pattern eliminates the supervisor entirely. Instead,
agents communicate through a shared findings pool:
    - NO agent decides what other agents do
    - EACH agent reads ALL prior findings and adds its own
    - Context accumulates naturally through sequential execution

This mirrors a medical team huddle where specialists take
turns sharing their perspective, each informed by what
came before.

------------------------------------------------------------
WHEN TO USE vs. NOT
------------------------------------------------------------
    Use:
      - Simple collaboration where order is known
      - When no centralized routing is needed
      - When agents should build on each other's findings
    Don't use:
      - When routing should be dynamic (use supervisor)
      - When agents should work independently (use voting)

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.orchestration.peer_to_peer_orchestration.runner
============================================================
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# CONNECTION: orchestration/models.py — SHARED_PATIENT is the demo patient case
# used by all 5 patterns so learners can compare orchestration behavior on identical data.
from orchestration.models import SHARED_PATIENT
from scripts.orchestration.peer_to_peer_orchestration.graph import build_peer_to_peer_graph


# ============================================================
# STAGE 2.4 — Main Execution
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  PEER-TO-PEER ORCHESTRATION")
    print("  Pattern 2/5: no coordinator, shared findings list")
    print("=" * 70)

    print("""
    Graph topology:

        [Pulmonology] --> [Cardiology] --> [Nephrology] --> [Synthesis]
            |                  |                |
            +-- shared findings grow at each step ------>

    No supervisor. No conditional edges. Context accumulates
    via the shared_findings list in state.
    """)

    patient = SHARED_PATIENT

    initial_state = {
        "messages": [],
        "patient_case": patient.model_dump(),
        "shared_findings": [],
        "current_peer": "",
        "peer_order": ["pulmonology", "cardiology", "nephrology"],
        "final_report": "",
    }

    print(f"    Patient: {patient.patient_id} | {patient.age}y {patient.sex}")
    print(f"    Complaint: {patient.chief_complaint}")
    print()
    print("    " + "-" * 60)

    graph = build_peer_to_peer_graph()
    result = graph.invoke(initial_state)

    # -- Display shared findings growth --------------------------------
    print("\n    " + "=" * 60)
    print("    SHARED FINDINGS (accumulated)")
    print("    " + "-" * 60)
    for finding_index, finding in enumerate(result.get("shared_findings", [])):
        print(f"    {finding_index + 1}. {finding[:200]}...")
        print()

    # -- Display final report ------------------------------------------
    print("    " + "=" * 60)
    print("    SYNTHESIZED REPORT")
    print("    " + "-" * 60)
    for line in result.get("final_report", "").split("\n"):
        print(f"    | {line}")

    # -- Summary -------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  PEER-TO-PEER ORCHESTRATION SUMMARY")
    print("=" * 70)
    print("""
    What you learned:

      1. No supervisor = no conditional edges (simplest graph)
      2. Coordination via shared_findings list in state
      3. Each peer reads ALL prior findings before contributing
      4. Factory function (_make_peer_node) avoids code duplication
      5. Same BaseOrchestrator.invoke_specialist used as supervisor

    Trade-offs vs. Supervisor:
      + Simpler graph topology (no loops)
      + No single-point-of-failure coordinator
      - Fixed execution order (not dynamic)
      - Later agents may be biased by earlier findings

    Next: dynamic_router_orchestration
    """)


if __name__ == "__main__":
    main()
