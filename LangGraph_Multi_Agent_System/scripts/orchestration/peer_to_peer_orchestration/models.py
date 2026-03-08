"""
============================================================
Peer-to-Peer Orchestration — State Models
============================================================
TypedDict for the P2P pattern. No supervisor, no coordinator.
Agents communicate via a shared findings list.

Key difference from Supervisor:
    Supervisor: one agent decides routing
    P2P: each agent reads all prior findings and contributes
============================================================
"""

# ============================================================
# STAGE 2.1 — State Definition
# ============================================================

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class PeerToPeerState(TypedDict):
    """
    LangGraph state for peer-to-peer orchestration.

    Fields:
        messages: conversation history
        patient_case: patient data as dict
        shared_findings: list of findings from all agents (grows incrementally)
        current_peer: name of the currently active peer agent
        peer_order: list defining the order peers run in
        final_report: synthesized report
    """
    messages: Annotated[list, add_messages]
    patient_case: dict
    shared_findings: list[str]
    current_peer: str
    peer_order: list[str]
    final_report: str
