"""
============================================================
Dynamic Router Orchestration — State Models
============================================================
State for the one-shot routing pattern.

Key difference from Supervisor:
    Supervisor: loops back after each worker (multi-shot)
    Dynamic Router: classifies ONCE, routes to ONE specialist,
    then optionally routes to a second based on classification.
    No loop-back. One-shot decision.
============================================================
"""

# ============================================================
# STAGE 3.1 — State Definition
# ============================================================

from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class CaseClassification(BaseModel):
    """
    Result of the router's input classification.

    The router analyzes the patient case and determines:
        - primary_specialty: the most relevant specialty
        - secondary_specialty: a backup if the case spans domains
        - urgency: classification for routing priority
        - reasoning: explanation of the routing decision
    """
    primary_specialty: str = Field(description="Most relevant specialty for this case")
    secondary_specialty: str = Field(default="", description="Secondary specialty if applicable")
    urgency: str = Field(default="standard", description="emergency / urgent / standard")
    reasoning: str = Field(default="", description="Why this routing was chosen")


class DynamicRouterState(TypedDict):
    """
    LangGraph state for dynamic router orchestration.

    Fields:
        messages: conversation history
        patient_case: patient data as dict
        classification: routing decision from the classifier
        primary_output: output from the primary specialist
        secondary_output: output from the secondary specialist (if any)
        final_report: synthesized report
    """
    messages: Annotated[list, add_messages]
    patient_case: dict
    classification: dict
    primary_output: str
    secondary_output: str
    final_report: str
