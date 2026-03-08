"""
============================================================
Dynamic Router Orchestration — Graph Construction
============================================================

    GRAPH TOPOLOGY
    ==============

        [START]
           |
           v
        [input_classifier]           <-- classifies case ONCE
           |
        route_to_primary()           <-- conditional edge
           |
        +--+---+---+
        |      |   |
        | pulm | cardio | nephro    <-- primary specialist
        +--+---+---+
           |
        route_to_secondary()         <-- optional second specialist
           |
        +--+---+---+---+
        |      |   |   |
        | pulm | c | n | skip      <-- secondary OR skip
        +--+---+---+---+
           |
           v
        [report]
           |
           v
        [END]

    Key difference from Supervisor:
        - ONE-SHOT classification (no loop)
        - TWO routing decisions (primary + secondary)
        - Deterministic after classification
============================================================
"""

# ============================================================
# STAGE 3.3 — Graph Construction
# ============================================================

from langgraph.graph import StateGraph, START, END

from scripts.orchestration.dynamic_router_orchestration.models import DynamicRouterState
from scripts.orchestration.dynamic_router_orchestration.agents import (
    input_classifier_node,
    pulmonology_specialist_node,
    cardiology_specialist_node,
    nephrology_specialist_node,
    router_report_node,
)


# ============================================================
# Routing Functions
# ============================================================

SPECIALTY_TO_NODE = {
    "pulmonology": "primary_pulmonology",
    "cardiology": "primary_cardiology",
    "nephrology": "primary_nephrology",
}

SECONDARY_TO_NODE = {
    "pulmonology": "secondary_pulmonology",
    "cardiology": "secondary_cardiology",
    "nephrology": "secondary_nephrology",
}


def route_to_primary_specialist(state: DynamicRouterState) -> str:
    """Route to the primary specialist based on classification."""
    classification = state.get("classification", {})
    primary = classification.get("primary_specialty", "pulmonology")
    return SPECIALTY_TO_NODE.get(primary, "primary_pulmonology")


def route_to_secondary_specialist(state: DynamicRouterState) -> str:
    """Route to secondary specialist or skip to report."""
    classification = state.get("classification", {})
    secondary = classification.get("secondary_specialty", "")

    if secondary and secondary in SECONDARY_TO_NODE:
        return SECONDARY_TO_NODE[secondary]
    return "report"


# ============================================================
# Graph Builder
# ============================================================

def build_dynamic_router_graph():
    """
    Build the dynamic router graph.

    Key patterns:
        1. Input classifier runs ONCE (no loop)
        2. Two conditional edges: primary + secondary routing
        3. Secondary can skip directly to report
        4. Deterministic execution after classification
    """
    workflow = StateGraph(DynamicRouterState)

    # -- Nodes -------------------------------------------------------
    workflow.add_node("input_classifier", input_classifier_node)

    # Primary specialist nodes
    workflow.add_node("primary_pulmonology", pulmonology_specialist_node)
    workflow.add_node("primary_cardiology", cardiology_specialist_node)
    workflow.add_node("primary_nephrology", nephrology_specialist_node)

    # Secondary specialist nodes (reuse same functions)
    workflow.add_node("secondary_pulmonology", pulmonology_specialist_node)
    workflow.add_node("secondary_cardiology", cardiology_specialist_node)
    workflow.add_node("secondary_nephrology", nephrology_specialist_node)

    workflow.add_node("report", router_report_node)

    # -- Entry -------------------------------------------------------
    workflow.add_edge(START, "input_classifier")

    # -- Primary routing (one-shot) -----------------------------------
    workflow.add_conditional_edges(
        "input_classifier",
        route_to_primary_specialist,
        {
            "primary_pulmonology": "primary_pulmonology",
            "primary_cardiology": "primary_cardiology",
            "primary_nephrology": "primary_nephrology",
        },
    )

    # -- After primary: route to secondary or report ------------------
    for primary_node in ["primary_pulmonology", "primary_cardiology", "primary_nephrology"]:
        workflow.add_conditional_edges(
            primary_node,
            route_to_secondary_specialist,
            {
                "secondary_pulmonology": "secondary_pulmonology",
                "secondary_cardiology": "secondary_cardiology",
                "secondary_nephrology": "secondary_nephrology",
                "report": "report",
            },
        )

    # -- Secondary specialists go to report ----------------------------
    for secondary_node in ["secondary_pulmonology", "secondary_cardiology", "secondary_nephrology"]:
        workflow.add_edge(secondary_node, "report")

    # -- Report ends --------------------------------------------------
    workflow.add_edge("report", END)

    return workflow.compile()
