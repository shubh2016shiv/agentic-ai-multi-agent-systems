"""
============================================================
Graph-of-Subgraphs Orchestration — State Models
============================================================
Each specialty has its OWN internal subgraph with multiple
steps. The parent graph orchestrates these subgraphs as
opaque units.

Key LangGraph concept:
    A subgraph is a compiled StateGraph used as a node inside
    a parent graph. The parent does not see the subgraph's
    internal nodes — only its input/output.
============================================================
"""

# ============================================================
# STAGE 4.1 — State Definition
# ============================================================

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class SubgraphInternalState(TypedDict):
    """
    Internal state for each specialty subgraph.

    Each subgraph runs a multi-step workflow:
        1. initial_assessment: raw specialist findings
        2. risk_analysis: risk scoring based on assessment
        3. recommendation: final recommendation
    """
    patient_case: dict
    initial_assessment: str
    risk_analysis: str
    recommendation: str


class ParentGraphState(TypedDict):
    """
    Parent graph state that orchestrates subgraphs.

    Each subgraph's output is stored as a dict in the
    corresponding specialty field.
    """
    messages: Annotated[list, add_messages]
    patient_case: dict
    pulmonology_result: dict
    cardiology_result: dict
    nephrology_result: dict
    final_report: str
