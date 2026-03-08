#!/usr/bin/env python3
"""
============================================================
Supervisor Orchestration — Runner
============================================================
PATTERN: 1 of 5
PREREQUISITE: Basic understanding of LangGraph state

------------------------------------------------------------
KEY CONCEPTS
------------------------------------------------------------
    - Centralized routing via a Supervisor LLM
    - add_conditional_edges for dynamic workflow
    - Return-to-supervisor loops via add_edge
    - Safe loop termination (FINISH)

Pattern 1 of 5: Central supervisor dynamically routes to
specialist agents and decides when to finish.

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
The supervisor is the most common production orchestration
pattern. One coordinator agent:
    1. Receives the patient case
    2. Decides which specialist should work next
    3. Reviews the specialist's output
    4. Routes to the next specialist or finishes

This script demonstrates:
    - Separation of concern: models.py / agents.py / graph.py / runner.py
    - BaseOrchestrator providing shared specialist invocation
    - SupervisorOrchestrator implementing routing logic
    - LangGraph conditional edges and worker-return loops

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    runner        supervisor        pulm          cardio        nephro        report
      |              |               |              |             |             |
      |-- invoke --> |               |              |             |             |
      |              |-- route ----> |              |             |             |
      |              |<-- output --- |              |             |             |
      |              |-- route ------|------------>  |             |             |
      |              |<-- output ----|------------- |             |             |
      |              |-- route ------|--------------|---------->  |             |
      |              |<-- output ----|--------------|----------- |             |
      |              |-- FINISH -----|--------------|------------|-----------> |
      |<-- report ---|              |              |             |             |

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.orchestration.supervisor_orchestration.runner
============================================================
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


from resilience import TokenBudgetConfig, TokenManager
# CONNECTION: orchestration/models.py — SHARED_PATIENT is the demo patient case
# used by all 5 patterns so learners can compare orchestration behavior on identical data.
from orchestration.models import SHARED_PATIENT
from scripts.orchestration.supervisor_orchestration.graph import build_supervisor_graph

# ============================================================
# STAGE 1.4 — Main Execution
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  SUPERVISOR ORCHESTRATION")
    print("  Pattern 1/5: central coordinator routes to specialists")
    print("=" * 70)

    print("""
    Graph topology:

        [supervisor]  <------+------+------+
           |                 |      |      |
           +-- "pulm" ------>|      |      |
           +-- "cardio" ----------->|      |
           +-- "nephro" ------------------>|
           +-- "FINISH" ---> [report] --> END

    Each specialist returns to supervisor after completing.
    Supervisor re-evaluates and routes to next or finishes.

    File structure:
        models.py  -> SupervisorState (data only)
        agents.py  -> SupervisorOrchestrator + worker nodes
        graph.py   -> build_supervisor_graph() (wiring only)
        runner.py  -> this file (execution + scenario)
    """)

    patient = SHARED_PATIENT

    # ── Token Budget Configuration ──────────────────────────────────────
    # ENTERPRISE PATTERN: One budget per workflow run (not shared globally).
    # Max 8,000 tokens = ~$0.01–0.03 depending on model (GPT-4o vs GPT-4-turbo).
    # With 3 specialists + synthesis = 4 calls, this allows ~2,000 tokens/call.
    # Prevents runaway costs from one expensive workflow.
    
    token_manager = TokenManager(TokenBudgetConfig(
        max_tokens_per_workflow=8_000,  # Workflow ceiling
        max_tokens_per_agent=3_000,      # Per-call ceiling
    ))

    initial_state = {
        "messages": [],
        "patient_case": patient.model_dump(),
        "next_worker": "",
        "completed_workers": [],
        "worker_outputs": {},
        "iteration": 0,
        "max_iterations": 6,
        "final_report": "",
        "token_manager": token_manager,
        "token_usage_summary": None,
    }

    print(f"    Patient: {patient.patient_id} | {patient.age}y {patient.sex}")
    print(f"    Complaint: {patient.chief_complaint}")
    print("    Specialists: pulmonology, cardiology, nephrology")
    print()
    print("    " + "-" * 60)

    graph = build_supervisor_graph()
    result = graph.invoke(initial_state)

    # -- Display routing trace -----------------------------------------
    print("\n    " + "=" * 60)
    print("    ROUTING TRACE")
    print("    " + "-" * 60)
    print(f"    Completed: {result['completed_workers']}")
    print(f"    Iterations: {result['iteration']}")

    # -- Display specialist outputs ------------------------------------
    for specialty, output in result.get("worker_outputs", {}).items():
        print(f"\n    [{specialty.upper()}]:")
        for line in output[:250].split("\n"):
            if line.strip():
                print(f"      {line}")

    # -- Display final report ------------------------------------------
    print("\n    " + "=" * 60)
    print("    SYNTHESIZED CLINICAL REPORT")
    print("    " + "-" * 60)
    for line in result.get("final_report", "").split("\n"):
        print(f"    | {line}")

    # -- Display Token Usage (Cost Observability) ──────────────────────
    if result.get("token_manager"):
        summary = result["token_manager"].get_workflow_summary()
        print("\n" + "=" * 70)
        print("  TOKEN USAGE (Cost Shield)")
        print("=" * 70)
        print(f"    Total tokens: {summary['total_tokens']:,}")
        print(f"    Budget limit: {summary['budget_limit']:,}")
        print(f"    Utilization:  {summary['utilization_pct']:.1f}%")
        print(f"    Remaining:    {summary['remaining']:,}")
        print()
        print("    Per-agent breakdown:")
        for agent_summary in result["token_manager"].get_all_agents_summary():
            print(f"      {agent_summary['agent']:20} {agent_summary['total_tokens']:>6,} tokens "
                  f"({agent_summary['calls']} calls)")

    # -- Summary -------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  SUPERVISOR ORCHESTRATION SUMMARY")
    print("=" * 70)
    print("""
    What you learned:

      1. Supervisor LLM dynamically decides agent routing order
      2. add_conditional_edges maps routing decisions to worker nodes
      3. All workers return to supervisor via fixed add_edge
      4. Max iteration guard prevents infinite loops
      5. Separation of concern: models / agents / graph / runner

    SOLID principles:
      - Single Responsibility: each file has one job
      - Open/Closed: SupervisorOrchestrator extends BaseOrchestrator
      - Dependency Inversion: graph.py depends on abstract orchestrator

    When to use:
      - Dynamic task ordering (supervisor adapts based on results)
      - Centralized accountability
      - Audit trail of routing decisions

    Next: peer_to_peer_orchestration
    """)


if __name__ == "__main__":
    main()
