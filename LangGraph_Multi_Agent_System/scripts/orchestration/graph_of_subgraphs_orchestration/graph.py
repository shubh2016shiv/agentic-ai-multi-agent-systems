"""
============================================================
Graph-of-Subgraphs Orchestration — Graph Construction
============================================================

    PARENT GRAPH TOPOLOGY
    =====================

        [START]
           |
           v
        [pulmonology_subgraph]   <-- 3 internal nodes (assessment -> risk -> recommendation)
           |
           v
        [cardiology_subgraph]    <-- 3 internal nodes
           |
           v
        [nephrology_subgraph]    <-- 3 internal nodes
           |
           v
        [synthesis]
           |
           v
        [END]

    SUBGRAPH INTERNAL TOPOLOGY (each specialty)
    ============================================

        [initial_assessment] --> [risk_analysis] --> [recommendation]

    The parent graph sees each subgraph as ONE node.
    The subgraph encapsulates its own multi-step workflow.
============================================================
"""

# ============================================================
# STAGE 4.3 — Graph Construction
# ============================================================

from langgraph.graph import StateGraph, START, END

from scripts.orchestration.graph_of_subgraphs_orchestration.models import (
    SubgraphInternalState,
    ParentGraphState,
)
from scripts.orchestration.graph_of_subgraphs_orchestration.agents import (
    make_initial_assessment_node,
    make_risk_analysis_node,
    make_recommendation_node,
    _make_subgraph_wrapper,
    synthesis_node,
)


# ============================================================
# Subgraph Builders
# ============================================================

def build_specialty_subgraph(specialty: str):
    """
    Build a 3-step subgraph for a given specialty.

    Each specialty subgraph runs:
        1. initial_assessment: raw findings
        2. risk_analysis: risk scoring
        3. recommendation: actionable recommendation

    Returns a compiled graph that can be used as a node.
    """
    workflow = StateGraph(SubgraphInternalState)

    workflow.add_node("initial_assessment", make_initial_assessment_node(specialty))
    workflow.add_node("risk_analysis", make_risk_analysis_node(specialty))
    workflow.add_node("recommendation", make_recommendation_node(specialty))

    workflow.add_edge(START, "initial_assessment")
    workflow.add_edge("initial_assessment", "risk_analysis")
    workflow.add_edge("risk_analysis", "recommendation")
    workflow.add_edge("recommendation", END)

    return workflow.compile()


# ============================================================
# Parent Graph Builder
# ============================================================

def build_graph_of_subgraphs():
    """
    Build the parent graph that orchestrates specialty subgraphs.

    Each specialty subgraph is compiled FIRST, then wrapped
    as a node in the parent graph. The parent graph treats
    each subgraph as an opaque, atomic unit.

    Key LangGraph pattern:
        - Subgraphs have their own internal state type
        - Wrapper nodes translate between parent and subgraph state
        - Parent graph uses simple sequential edges
    """
    # Build and compile each specialty subgraph
    pulm_subgraph = build_specialty_subgraph("pulmonology")
    cardio_subgraph = build_specialty_subgraph("cardiology")
    nephro_subgraph = build_specialty_subgraph("nephrology")

    # Build the parent graph
    parent = StateGraph(ParentGraphState)

    parent.add_node("pulmonology_subgraph", _make_subgraph_wrapper("pulmonology", pulm_subgraph))
    parent.add_node("cardiology_subgraph", _make_subgraph_wrapper("cardiology", cardio_subgraph))
    parent.add_node("nephrology_subgraph", _make_subgraph_wrapper("nephrology", nephro_subgraph))
    parent.add_node("synthesis", synthesis_node)

    parent.add_edge(START, "pulmonology_subgraph")
    parent.add_edge("pulmonology_subgraph", "cardiology_subgraph")
    parent.add_edge("cardiology_subgraph", "nephrology_subgraph")
    parent.add_edge("nephrology_subgraph", "synthesis")
    parent.add_edge("synthesis", END)

    return parent.compile()
