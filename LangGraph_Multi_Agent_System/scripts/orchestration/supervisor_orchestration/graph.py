"""
============================================================
Supervisor Orchestration — Graph Construction
============================================================
Builds the LangGraph StateGraph for the supervisor pattern.

Separation of concern:
    This file defines ONLY the graph topology and wiring.
    Agent behavior is in agents.py. State shape is in models.py.

    GRAPH TOPOLOGY
    ==============

        [START]
           |
           v
        [supervisor]  <------+------+------+
           |                 |      |      |
           |-- "pulm" ------>|      |      |
           |-- "cardio" ----------->|      |
           |-- "nephro" ------------------>|
           |-- "FINISH" ---> [report] --> [END]

        All specialists return to supervisor via add_edge.
        Supervisor decides next worker via add_conditional_edges.
============================================================
"""

# ============================================================
# STAGE 1.3 — Graph Construction
# ============================================================

from langgraph.graph import StateGraph, START, END

from scripts.orchestration.supervisor_orchestration.models import SupervisorState
from scripts.orchestration.supervisor_orchestration.agents import (
    SupervisorOrchestrator,
    pulmonology_worker_node,
    cardiology_worker_node,
    nephrology_worker_node,
    report_synthesis_node,
)


# ============================================================
# Routing Function
# ============================================================

def route_supervisor_decision(state: SupervisorState) -> str:
    """
    Map the supervisor's next_worker decision to a graph node name.

    This is the function passed to add_conditional_edges().
    """
    next_worker = state.get("next_worker", "FINISH")
    routing_table = {
        "pulmonology": "pulmonology_worker",
        "cardiology": "cardiology_worker",
        "nephrology": "nephrology_worker",
        "FINISH": "report_synthesis",
    }
    return routing_table.get(next_worker, "report_synthesis")


# ============================================================
# Graph Builder
# ============================================================

_supervisor = SupervisorOrchestrator()


def supervisor_decision_node(state: SupervisorState) -> dict:
    """Wrapper that calls the supervisor's routing logic."""
    return _supervisor.decide_next_worker(state)


def build_supervisor_graph():
    """
    Build the supervisor-orchestrated multi-agent graph.

    Key LangGraph patterns demonstrated:
        1. add_conditional_edges: supervisor routes dynamically
        2. add_edge: workers always return to supervisor (fixed edges)
        3. Termination: supervisor says "FINISH" -> report -> END
        4. Loop guard: max_iterations prevents infinite cycling
    """
    workflow = StateGraph(SupervisorState)

    # -- Nodes -------------------------------------------------------
    workflow.add_node("supervisor", supervisor_decision_node)
    workflow.add_node("pulmonology_worker", pulmonology_worker_node)
    workflow.add_node("cardiology_worker", cardiology_worker_node)
    workflow.add_node("nephrology_worker", nephrology_worker_node)
    workflow.add_node("report_synthesis", report_synthesis_node)

    # -- Entry -------------------------------------------------------
    workflow.add_edge(START, "supervisor")

    # -- Supervisor routes conditionally -----------------------------
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor_decision,
        {
            "pulmonology_worker": "pulmonology_worker",
            "cardiology_worker": "cardiology_worker",
            "nephrology_worker": "nephrology_worker",
            "report_synthesis": "report_synthesis",
        },
    )

    # -- Workers always return to supervisor -------------------------
    workflow.add_edge("pulmonology_worker", "supervisor")
    workflow.add_edge("cardiology_worker", "supervisor")
    workflow.add_edge("nephrology_worker", "supervisor")

    # -- Report ends the graph ---------------------------------------
    workflow.add_edge("report_synthesis", END)

    return workflow.compile()
