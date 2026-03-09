"""
============================================================
Hybrid Orchestration — State Models
============================================================
Combines Supervisor + P2P: a supervisor routes at the
department level, and within each department agents
collaborate peer-to-peer.

Key LangGraph concept:
    This pattern mixes conditional edges (supervisor routing)
    with sequential edges (P2P within clusters).
============================================================
"""

# ============================================================
# STAGE 5.1 — State Definition
# ============================================================

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class HybridState(TypedDict):
    """
    LangGraph state for hybrid orchestration.

    Fields:
        messages: conversation history
        patient_case: patient data
        supervisor_routing: which department cluster to activate
        cardiopulmonary_findings: P2P results from cardio + pulm cluster
        renal_findings: results from nephrology assessment
        cluster_outputs: dict of cluster name -> findings
        routing_decisions: list tracking supervisor's routing trace
        iteration: current iteration count
        max_iterations: safety limit
        final_report: synthesized report
    """
    messages: Annotated[list, add_messages]
    patient_case: dict
    supervisor_routing: str
    cardiopulmonary_findings: list[str]
    renal_findings: str
    cluster_outputs: dict
    routing_decisions: list[str]
    iteration: int
    max_iterations: int
    final_report: str
