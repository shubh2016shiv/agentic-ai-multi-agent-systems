"""
============================================================
Peer-to-Peer Orchestration — Graph Construction
============================================================
Builds a linear chain where each agent gets all prior findings.

    GRAPH TOPOLOGY
    ==============

        [START]
           |
           v
        [pulmonology_peer]    <-- reads: nothing (first peer)
           |
           v
        [cardiology_peer]     <-- reads: pulmonology findings
           |
           v
        [nephrology_peer]     <-- reads: pulmonology + cardiology
           |
           v
        [synthesis]           <-- reads: all three findings
           |
           v
        [END]

    Key difference from Supervisor:
        - NO conditional edges (fixed order)
        - NO return-to-coordinator loop
        - Context grows at each stage via shared_findings
============================================================
"""

# ============================================================
# STAGE 2.3 — Graph Construction
# ============================================================

from langgraph.graph import StateGraph, START, END

from scripts.orchestration.peer_to_peer_orchestration.models import PeerToPeerState
from scripts.orchestration.peer_to_peer_orchestration.agents import (
    pulmonology_peer_node,
    cardiology_peer_node,
    nephrology_peer_node,
    synthesis_node,
)


def build_peer_to_peer_graph():
    """
    Build the peer-to-peer graph.

    This is the simplest possible multi-agent graph:
    pure sequential edges, no conditional routing.
    The "orchestration" happens via the shared_findings list
    that grows at each stage.
    """
    workflow = StateGraph(PeerToPeerState)

    workflow.add_node("pulmonology_peer", pulmonology_peer_node)
    workflow.add_node("cardiology_peer", cardiology_peer_node)
    workflow.add_node("nephrology_peer", nephrology_peer_node)
    workflow.add_node("synthesis", synthesis_node)

    workflow.add_edge(START, "pulmonology_peer")
    workflow.add_edge("pulmonology_peer", "cardiology_peer")
    workflow.add_edge("cardiology_peer", "nephrology_peer")
    workflow.add_edge("nephrology_peer", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()
