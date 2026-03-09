#!/usr/bin/env python3
"""
============================================================
Dynamic Router Orchestration — Runner
============================================================
PATTERN: 3 of 5
PREREQUISITE: Basic understanding of LangGraph state

------------------------------------------------------------
KEY CONCEPTS
------------------------------------------------------------
    - One-shot classification for routing
    - Bypassing irrelevant nodes for efficiency
    - Primary and secondary conditional routing
    - Structured LLM outputs via Pydantic

Pattern 3 of 5: One-shot input classification routes the
patient case to the most relevant specialist(s).

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Unlike Supervisor (which loops) and P2P (which runs all
agents sequentially), the dynamic router makes a SINGLE
classification decision and routes to only the relevant
specialist(s).

This reduces latency and cost by skipping irrelevant agents.

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    runner     classifier     primary_spec    secondary_spec    report
      |            |               |               |              |
      |-- invoke ->|               |               |              |
      |            |-- classify    |               |              |
      |            |-- route ----> |               |              |
      |            |               |-- assess      |              |
      |            |               |-- route ------+-> (if any)   |
      |            |               |               |-- assess     |
      |            |               |               |-- route ---->|
      |            |               |               |              |-- synthesize
      |<-- report -|               |               |              |

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.orchestration.dynamic_router_orchestration.runner
============================================================
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# CONNECTION: orchestration/models.py — SHARED_PATIENT is the demo patient case
# used by all 5 patterns so learners can compare orchestration behavior on identical data.
from orchestration.models import SHARED_PATIENT
from scripts.orchestration.dynamic_router_orchestration.graph import build_dynamic_router_graph


# ============================================================
# STAGE 3.4 — Main Execution
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  DYNAMIC ROUTER ORCHESTRATION")
    print("  Pattern 3/5: one-shot classification routes to specialist(s)")
    print("=" * 70)

    print("""
    Graph topology:

        [Classifier] --+--> [Primary Specialist]
                       |         |
                       |    [Secondary Specialist] (optional)
                       |         |
                       +-------> [Report] --> END

    Key difference from Supervisor:
      - ONE classification decision (no loop-back)
      - Routes to 1-2 specialists only (not all 3)
      - Lower latency and cost for clear-cut cases
    """)

    patient = SHARED_PATIENT

    initial_state = {
        "messages": [],
        "patient_case": patient.model_dump(),
        "classification": {},
        "primary_output": "",
        "secondary_output": "",
        "final_report": "",
    }

    print(f"    Patient: {patient.patient_id} | {patient.age}y {patient.sex}")
    print(f"    Complaint: {patient.chief_complaint}")
    print()
    print("    " + "-" * 60)

    graph = build_dynamic_router_graph()
    result = graph.invoke(initial_state)

    # -- Display classification ----------------------------------------
    classification = result.get("classification", {})
    print("\n    " + "=" * 60)
    print("    CLASSIFICATION RESULT")
    print("    " + "-" * 60)
    print(f"    Primary:   {classification.get('primary_specialty', '?')}")
    print(f"    Secondary: {classification.get('secondary_specialty', 'NONE')}")
    print(f"    Urgency:   {classification.get('urgency', '?')}")
    print(f"    Reasoning: {classification.get('reasoning', '?')[:150]}")

    # -- Display specialist outputs ------------------------------------
    if result.get("primary_output"):
        print(f"\n    [PRIMARY] {classification.get('primary_specialty', '').upper()}:")
        for line in result["primary_output"][:250].split("\n"):
            if line.strip():
                print(f"      {line}")

    if result.get("secondary_output"):
        print(f"\n    [SECONDARY] {classification.get('secondary_specialty', '').upper()}:")
        for line in result["secondary_output"][:250].split("\n"):
            if line.strip():
                print(f"      {line}")

    # -- Display report ------------------------------------------------
    print("\n    " + "=" * 60)
    print("    SYNTHESIZED REPORT")
    print("    " + "-" * 60)
    for line in result.get("final_report", "").split("\n"):
        print(f"    | {line}")

    # -- Summary -------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  DYNAMIC ROUTER ORCHESTRATION SUMMARY")
    print("=" * 70)
    print("""
    What you learned:

      1. Input classifier makes a ONE-SHOT routing decision
      2. Two-stage conditional edges: primary + optional secondary
      3. CaseClassification Pydantic model structures the routing output
      4. Skips irrelevant specialists (efficiency over completeness)
      5. No loop-back — deterministic after classification

    Trade-offs:
      + Lower latency (fewer agents run)
      + Lower cost (fewer LLM calls)
      - May miss cross-specialty interactions
      - Classification errors propagate (no self-correction)

    When to use:
      - High-volume triage where speed matters
      - Cases with a clearly dominant specialty
      - Cost-sensitive deployments

    Next: graph_of_subgraphs_orchestration
    """)


if __name__ == "__main__":
    main()
