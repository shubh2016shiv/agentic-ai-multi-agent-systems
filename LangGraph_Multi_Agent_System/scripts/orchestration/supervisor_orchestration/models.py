"""
============================================================
Supervisor Orchestration — State Models
============================================================
TypedDict and Pydantic models for the supervisor pattern.

Separation of concern:
    This file defines ONLY the data shapes. No logic, no LLM
    calls, no graph construction. Other files import these.
============================================================
"""

# ============================================================
# STAGE 1.1 — State Definition
# ============================================================

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class SupervisorState(TypedDict):
    """
    LangGraph state for the supervisor orchestration pattern.

    Fields:
        messages: conversation history (managed by add_messages reducer)
        patient_case: patient data as a dict
        next_worker: which specialist the supervisor routes to next
        completed_workers: list of specialties that have finished
        worker_outputs: dict mapping specialty -> assessment string
        iteration: current supervisor decision count
        max_iterations: safety limit to prevent infinite loops
        final_report: synthesized clinical report
        token_manager: TokenManager instance (per-workflow budget tracker)
        token_usage_summary: Final token usage report after workflow completes
        
    Token Manager (Enterprise Pattern):
        token_manager: ONE TokenManager per workflow run. All specialist/synthesis
                       nodes share this instance so budget accumulates across agents.
                       
                       ENTERPRISE PATTERN: Token budget is per-workflow, not global.
                       Each graph.invoke() gets a fresh budget. This prevents one
                       expensive workflow from affecting another.
                       
        token_usage_summary: Dict containing final usage stats (total_tokens,
                             utilization_pct, per-agent breakdown). Populated after
                             workflow completes for cost observability.
    """
    messages: Annotated[list, add_messages]
    patient_case: dict
    next_worker: str
    completed_workers: list[str]
    worker_outputs: dict
    iteration: int
    max_iterations: int
    final_report: str
    token_manager: object | None
    token_usage_summary: dict | None
