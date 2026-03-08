"""
============================================================
Graph-of-Subgraphs Orchestration — Agent Definitions
============================================================
Each specialty is a self-contained multi-step subgraph:

    [initial_assessment] --> [risk_analysis] --> [recommendation]

The parent graph treats each subgraph as a single atomic node.

This pattern demonstrates NESTED GRAPHS in LangGraph: the parent
graph coordinates subgraphs, and each subgraph has its own internal
state, nodes, and edges.

CIRCUIT BREAKER (resilience/circuit_breaker.py):
    DESTINATION — ALL subgraph nodes now trigger the circuit breaker
    via _ORCHESTRATION_CALLER.call(llm.invoke, ...) which routes through
    ResilientCaller → circuit_breaker.call():
        initial_assessment_node (×3 specialties)
        risk_analysis_node (×3 specialties)
        recommendation_node (×3 specialties)
        synthesis_node (parent graph)

    All 10 nodes (9 subgraph + 1 synthesis) go through the resilience stack.

RATE LIMITER (resilience/rate_limiter.py):
    DESTINATION — same nodes as circuit breaker. ENABLED
    (skip_rate_limiter=False) in _ORCHESTRATION_CALLER.

WHY subgraph nodes use _ORCHESTRATION_CALLER directly instead of
invoke_specialist():
    Subgraph steps have intentionally different prompt structures
    (assessment → risk → recommendation) rather than the standard
    "you are a {specialty} specialist, assess this patient" prompt.
    Using _ORCHESTRATION_CALLER.call(llm.invoke, prompt) gives us the
    full resilience stack (circuit breaker + retry + timeout + rate limiter)
    without forcing the standard specialist prompt.
============================================================
"""

# ============================================================
# STAGE 4.2 — Agent Definitions
# ============================================================

# STAGE 4.2a — Imports and Orchestrator Setup

import sys


from core.config import get_llm
from core.models import PatientCase
from observability.callbacks import build_callback_config
# CONNECTION: orchestration/models.py — format_patient_for_prompt() normalizes
# PatientCase to a standardized prompt string used in all subgraph nodes.
from orchestration.models import format_patient_for_prompt
# CONNECTION: orchestration/orchestrator.py — BaseOrchestrator provides the
# _orchestrator instance with invoke_synthesizer (used in synthesis_node).
# SPECIALIST_SYSTEM_PROMPTS ensures all subgraph steps get consistent system context.
# _ORCHESTRATION_CALLER is the ResilientCaller façade that routes ALL LLM calls
# through the 6-layer resilience stack (circuit breaker → retry → timeout).
from orchestration.orchestrator import BaseOrchestrator, SPECIALIST_SYSTEM_PROMPTS, _ORCHESTRATION_CALLER
from scripts.orchestration.graph_of_subgraphs_orchestration.models import (
    SubgraphInternalState,
    ParentGraphState,
)


# STAGE 4.2b — Subgraph Orchestrator

class SubgraphOrchestrator(BaseOrchestrator):
    """
    Orchestrator for the graph-of-subgraphs pattern.

    Unlike other orchestrators that invoke_specialist() for each specialty,
    this orchestrator runs multi-step subgraphs (assessment → risk → recommendation)
    where each step uses _ORCHESTRATION_CALLER.call(llm.invoke, ...) directly,
    preserving the resilience stack while using subgraph-specific prompts.
    """

    @property
    def pattern_name(self) -> str:
        return "graph_of_subgraphs"

    @property
    def description(self) -> str:
        return "Each specialty runs a multi-step subgraph as an atomic node"


_orchestrator = SubgraphOrchestrator()


# ============================================================
# STAGE 4.2c — Subgraph Node Factories
# ============================================================
# Each specialty subgraph has 3 internal nodes (assessment → risk → recommendation).
# We use factory functions to avoid repeating the same structure 3 times.
#
# ALL nodes now use _ORCHESTRATION_CALLER.call(llm.invoke, ...) instead of raw
# llm.invoke(), routing every call through:
#   rate limiter → circuit breaker → retry → timeout (6-layer resilience stack).

def make_initial_assessment_node(specialty: str):
    """
    Factory: create the initial assessment node for a specialty subgraph.

    Sub-stage 4.2c.1 — Initial Assessment (first step in each subgraph).
    Reads raw patient data, produces key findings and concerns.

    CIRCUIT BREAKER: This node triggers resilience/circuit_breaker via
    _ORCHESTRATION_CALLER.call(llm.invoke, ...) → circuit_breaker.call().
    CONNECTION: resilience/resilient_caller.py — _ORCHESTRATION_CALLER
    applies the full 6-layer stack.
    """

    def initial_assessment_node(state: SubgraphInternalState) -> dict:
        patient = state["patient_case"]
        llm = get_llm()

        system_prompt = SPECIALIST_SYSTEM_PROMPTS.get(specialty, f"You are a {specialty} specialist.")
        patient_text = format_patient_for_prompt(PatientCase(**patient))

        prompt = f"""{system_prompt}

{patient_text}

Provide your INITIAL ASSESSMENT: key findings and concerns.
Keep under 80 words."""

        config = build_callback_config(
            trace_name=f"subgraph_{specialty}_assessment",
            tags=["orchestration", "subgraph", specialty],
        )
        # CONNECTION: resilience/resilient_caller.py — _ORCHESTRATION_CALLER routes
        # this call through: rate limiter → circuit breaker → retry → timeout.
        # Previously this was raw llm.invoke() with no resilience protection.
        response = _ORCHESTRATION_CALLER.call(
            llm.invoke,
            prompt,
            config=config,
            skip_rate_limiter=False,  # RATE LIMITER ENABLED (smooths bursts)
            skip_bulkhead=True,       # Bulkhead skipped (linear subgraph flow)
        )
        print(f"    |   [{specialty}] Assessment: {response.content[:80]}...")
        return {"initial_assessment": response.content}

    initial_assessment_node.__name__ = f"{specialty}_initial_assessment"
    return initial_assessment_node


def make_risk_analysis_node(specialty: str):
    """
    Factory: create the risk analysis node for a specialty subgraph.

    Sub-stage 4.2c.2 — Risk Analysis (second step in each subgraph).
    Reads initial_assessment from subgraph state, produces risk level.

    CIRCUIT BREAKER: This node triggers resilience/circuit_breaker via
    _ORCHESTRATION_CALLER.call(llm.invoke, ...) → circuit_breaker.call().
    CONNECTION: resilience/resilient_caller.py — same shared _ORCHESTRATION_CALLER
    and _ORCHESTRATION_LLM_BREAKER as all other orchestration nodes.
    """

    def risk_analysis_node(state: SubgraphInternalState) -> dict:
        assessment = state.get("initial_assessment", "")
        llm = get_llm()

        prompt = f"""Based on this {specialty} assessment:
{assessment}

Provide a RISK ANALYSIS:
1. Risk level: LOW / MODERATE / HIGH / CRITICAL
2. Key risk factors (2-3 items)
3. Urgency of intervention
Keep under 60 words."""

        config = build_callback_config(
            trace_name=f"subgraph_{specialty}_risk",
            tags=["orchestration", "subgraph", specialty],
        )
        # CONNECTION: resilience/resilient_caller.py — _ORCHESTRATION_CALLER routes
        # this call through the resilience stack. Previously was raw llm.invoke().
        response = _ORCHESTRATION_CALLER.call(
            llm.invoke,
            prompt,
            config=config,
            skip_rate_limiter=False,
            skip_bulkhead=True,
        )
        print(f"    |   [{specialty}] Risk: {response.content[:60]}...")
        return {"risk_analysis": response.content}

    risk_analysis_node.__name__ = f"{specialty}_risk_analysis"
    return risk_analysis_node


def make_recommendation_node(specialty: str):
    """
    Factory: create the recommendation node for a specialty subgraph.

    Sub-stage 4.2c.3 — Recommendation (third/final step in each subgraph).
    Reads assessment + risk from subgraph state, produces final recommendations.

    CIRCUIT BREAKER: This node triggers resilience/circuit_breaker via
    _ORCHESTRATION_CALLER.call(llm.invoke, ...) → circuit_breaker.call().
    CONNECTION: resilience/resilient_caller.py — same shared _ORCHESTRATION_CALLER.
    """

    def recommendation_node(state: SubgraphInternalState) -> dict:
        assessment = state.get("initial_assessment", "")
        risk = state.get("risk_analysis", "")
        llm = get_llm()

        prompt = f"""Based on this {specialty} assessment and risk analysis:

Assessment: {assessment}
Risk: {risk}

Provide your FINAL RECOMMENDATION:
1. Immediate actions needed
2. Medication adjustments
3. Follow-up timeline
Keep under 80 words."""

        config = build_callback_config(
            trace_name=f"subgraph_{specialty}_recommendation",
            tags=["orchestration", "subgraph", specialty],
        )
        # CONNECTION: resilience/resilient_caller.py — _ORCHESTRATION_CALLER routes
        # this call through the resilience stack. Previously was raw llm.invoke().
        response = _ORCHESTRATION_CALLER.call(
            llm.invoke,
            prompt,
            config=config,
            skip_rate_limiter=False,
            skip_bulkhead=True,
        )
        print(f"    |   [{specialty}] Recommendation: {response.content[:80]}...")
        return {"recommendation": response.content}

    recommendation_node.__name__ = f"{specialty}_recommendation"
    return recommendation_node


# ============================================================
# STAGE 4.2d — Parent Graph Wrapper Nodes
# ============================================================

def _make_subgraph_wrapper(specialty: str, compiled_subgraph):
    """
    Wrap a compiled subgraph so it can be invoked as a parent
    graph node. Maps parent state → subgraph input → subgraph
    output → parent state update.

    The parent graph sees each subgraph as a single atomic node.
    Internally, the subgraph runs 3 steps (assessment → risk → recommendation),
    each protected by the resilience stack via _ORCHESTRATION_CALLER.
    """
    result_field = f"{specialty}_result"

    def wrapper_node(state: ParentGraphState) -> dict:
        print(f"    | [Subgraph:{specialty}] Running 3-step workflow...")
        subgraph_input = {
            "patient_case": state["patient_case"],
            "initial_assessment": "",
            "risk_analysis": "",
            "recommendation": "",
        }

        subgraph_result = compiled_subgraph.invoke(subgraph_input)

        return {
            result_field: {
                "assessment": subgraph_result.get("initial_assessment", ""),
                "risk": subgraph_result.get("risk_analysis", ""),
                "recommendation": subgraph_result.get("recommendation", ""),
            }
        }

    wrapper_node.__name__ = f"{specialty}_subgraph_wrapper"
    return wrapper_node


# ============================================================
# STAGE 4.2e — Synthesis Node (Parent Graph)
# ============================================================

def synthesis_node(state: ParentGraphState) -> dict:
    """
    Synthesize all subgraph results into a final report.

    This is the final node in the parent graph. It receives the structured
    outputs from all 3 specialty subgraphs (each containing assessment,
    risk, and recommendation) and synthesizes them into a unified report.

    CIRCUIT BREAKER: This node triggers resilience/circuit_breaker via
    _ORCHESTRATION_CALLER.call(llm.invoke, ...) → circuit_breaker.call().
    CONNECTION: resilience/resilient_caller.py — same shared _ORCHESTRATION_CALLER
    and _ORCHESTRATION_LLM_BREAKER used by all subgraph nodes.
    """
    patient = PatientCase(**state["patient_case"])

    all_findings = []
    for specialty in ["pulmonology", "cardiology", "nephrology"]:
        result = state.get(f"{specialty}_result", {})
        if result:
            all_findings.append(
                f"[{specialty.upper()}]\n"
                f"  Assessment: {result.get('assessment', '')[:200]}\n"
                f"  Risk: {result.get('risk', '')[:100]}\n"
                f"  Recommendation: {result.get('recommendation', '')[:200]}"
            )

    llm = get_llm()
    prompt = f"""Synthesize these multi-step specialist workflows into a unified report:

Patient: {patient.age}y {patient.sex}, {patient.chief_complaint}

{chr(10).join(all_findings)}

Produce:
1) Critical Findings  2) Cross-Specialty Risks  3) Integrated Plan
Keep under 200 words."""

    config = build_callback_config(
        trace_name="subgraph_synthesis",
        tags=["orchestration", "subgraph", "synthesis"],
    )
    # CONNECTION: resilience/resilient_caller.py — _ORCHESTRATION_CALLER routes
    # this synthesis call through the full resilience stack.
    # Previously was raw llm.invoke() — bypassing circuit breaker, retry, timeout.
    response = _ORCHESTRATION_CALLER.call(
        llm.invoke,
        prompt,
        config=config,
        skip_rate_limiter=False,
        skip_bulkhead=True,
    )
    print(f"    | [Synthesis] Report: {len(response.content)} chars")

    return {"final_report": response.content}
