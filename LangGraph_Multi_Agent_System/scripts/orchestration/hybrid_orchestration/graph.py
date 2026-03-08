"""
============================================================
Hybrid Orchestration — Graph Construction
============================================================

    GRAPH TOPOLOGY
    ==============

    [START]
       |
       v
    [supervisor]  <-----+-----+
       |                |     |
       |-- "cardiopulm" |     |
       |    |           |     |
       |    v           |     |
       |  [pulm_peer] --+     |
       |    |                 |
       |    v                 |
       |  [cardio_peer] -----+   (returns to supervisor)
       |
       |-- "renal"
       |    |
       |    v
       |  [nephro] ----------+   (returns to supervisor)
       |
       |-- "FINISH"
       |    |
       |    v
       |  [synthesis] --> [END]

    The HYBRID:
        - Supervisor routes at DEPARTMENT level (conditional edges)
        - Within cardiopulmonary: P2P chain (sequential edges)
        - Nephrology runs as a standalone specialist
============================================================
"""

# ============================================================
# STAGE 5.3 — Graph Construction
# ============================================================

from langgraph.graph import StateGraph, START, END

from scripts.orchestration.hybrid_orchestration.models import HybridState
from scripts.orchestration.hybrid_orchestration.agents import (
    hybrid_supervisor_node,
    cardiopulmonary_pulmonology_node,
    cardiopulmonary_cardiology_node,
    renal_specialist_node,
    hybrid_synthesis_node,
)


# ============================================================
# Routing
# ============================================================

def route_hybrid_supervisor(state: HybridState) -> str:
    """Map supervisor department routing to graph nodes."""
    routing = state.get("supervisor_routing", "FINISH")
    routing_table = {
        "cardiopulmonary": "pulmonology_peer",
        "renal": "renal_specialist",
        "FINISH": "synthesis",
    }
    return routing_table.get(routing, "synthesis")


# ============================================================
# Graph Builder
# ============================================================

def build_hybrid_graph():
    """
    Build the hybrid graph combining supervisor + P2P.

    Key patterns:
        1. Supervisor routes to departments (conditional edges)
        2. Cardiopulmonary cluster runs P2P (sequential edges)
        3. P2P cluster returns to supervisor after completion
        4. Renal runs standalone and returns to supervisor
        5. Supervisor loop terminates with "FINISH" -> synthesis
    """
    workflow = StateGraph(HybridState)

    # -- Nodes -------------------------------------------------------
    workflow.add_node("supervisor", hybrid_supervisor_node)
    workflow.add_node("pulmonology_peer", cardiopulmonary_pulmonology_node)
    workflow.add_node("cardiology_peer", cardiopulmonary_cardiology_node)
    workflow.add_node("renal_specialist", renal_specialist_node)
    workflow.add_node("synthesis", hybrid_synthesis_node)

    # -- Entry -------------------------------------------------------
    workflow.add_edge(START, "supervisor")

    # -- Supervisor conditional routing (department level) ------------
    workflow.add_conditional_edges(
        "supervisor",
        route_hybrid_supervisor,
        {
            "pulmonology_peer": "pulmonology_peer",
            "renal_specialist": "renal_specialist",
            "synthesis": "synthesis",
        },
    )

    # -- Cardiopulmonary P2P cluster (sequential edges) ---------------
    workflow.add_edge("pulmonology_peer", "cardiology_peer")
    workflow.add_edge("cardiology_peer", "supervisor")  # return to supervisor

    # -- Renal returns to supervisor ----------------------------------
    workflow.add_edge("renal_specialist", "supervisor")

    # -- Synthesis ends -----------------------------------------------
    workflow.add_edge("synthesis", END)

    return workflow.compile()
