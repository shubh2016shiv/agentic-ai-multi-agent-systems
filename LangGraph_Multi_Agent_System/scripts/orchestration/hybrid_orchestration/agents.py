"""
============================================================
Hybrid Orchestration — Agent Definitions
============================================================
Combines two coordination mechanisms:
    - SUPERVISOR: high-level routing (which department to activate)
    - P2P CLUSTER: within the cardiopulmonary department, pulmonology
      and cardiology agents collaborate via shared findings

This mirrors real hospital structures:
    - Chief Medical Officer decides which department handles the case
    - Within each department, specialists collaborate peer-to-peer

CIRCUIT BREAKER (resilience/circuit_breaker.py):
    DESTINATION — these nodes trigger the circuit breaker via invoke_specialist/
    invoke_synthesizer → ResilientCaller → circuit_breaker.call():
        cardiopulmonary_pulmonology_node, cardiopulmonary_cardiology_node,
        renal_specialist_node, hybrid_synthesis_node.
    NOT triggered: hybrid_supervisor_node (uses llm.invoke directly).

RATE LIMITER (resilience/rate_limiter.py):
    DESTINATION — same nodes as circuit breaker. ENABLED (skip_rate_limiter=False)
    in orchestrator base.
============================================================
"""

# ============================================================
# STAGE 5.2 — Agent Definitions
# ============================================================

import sys
import json


from core.config import get_llm
from core.models import PatientCase
from observability.callbacks import build_callback_config
# CONNECTION: orchestration/ root module — BaseOrchestrator provides invoke_specialist/
# invoke_synthesizer which route all LLM calls through the 6-layer resilience stack
# (circuit breaker → retry → timeout via ResilientCaller in orchestration/orchestrator.py).
from orchestration.orchestrator import BaseOrchestrator
# CONNECTION: orchestration/models.py — OrchestrationResult is the standard agent
# output envelope. format_patient_for_prompt() normalizes PatientCase to a prompt string.
from orchestration.models import OrchestrationResult, format_patient_for_prompt
from scripts.orchestration.hybrid_orchestration.models import HybridState


class HybridOrchestrator(BaseOrchestrator):

    @property
    def pattern_name(self) -> str:
        return "hybrid"

    @property
    def description(self) -> str:
        return "Supervisor routes at department level; P2P within departments"


_orchestrator = HybridOrchestrator()


# ============================================================
# Supervisor Node (high-level routing)
# ============================================================

def hybrid_supervisor_node(state: HybridState) -> dict:
    """
    High-level supervisor that routes to department clusters.

    Routes to:
        - "cardiopulmonary": activates P2P cluster (pulm + cardio)
        - "renal": activates nephrology agent
        - "FINISH": signals all assessments are done

    This operates at the DEPARTMENT level, not individual agents.
    """
    patient = state["patient_case"]
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 4)
    routing_decisions = list(state.get("routing_decisions", []))
    cluster_outputs = state.get("cluster_outputs", {})

    completed_departments = list(cluster_outputs.keys())
    available_departments = [
        dept for dept in ["cardiopulmonary", "renal"]
        if dept not in completed_departments
    ]

    if iteration >= max_iterations or not available_departments:
        print(f"    | [Supervisor] Finishing: {'max iterations' if iteration >= max_iterations else 'all departments done'}")
        return {
            "supervisor_routing": "FINISH",
            "iteration": iteration + 1,
            "routing_decisions": routing_decisions + ["FINISH"],
        }

    llm = get_llm()
    completed_summary = ""
    if cluster_outputs:
        completed_summary = "\n".join(
            f"- {dept}: completed" for dept in completed_departments
        )

    prompt = f"""You are a clinical supervisor routing a patient case to departments.

Patient: {patient.get('age')}y {patient.get('sex')}, {patient.get('chief_complaint')}
History: {', '.join(patient.get('medical_history', []))}

Completed departments: {', '.join(completed_departments) if completed_departments else 'None'}
Available departments: {', '.join(available_departments)}
{f'Status:{chr(10)}{completed_summary}' if completed_summary else ''}

Departments:
  - cardiopulmonary: handles COPD + heart failure (pulm + cardio team)
  - renal: handles CKD and medication adjustments (nephrology)

Which department should assess next? Respond with ONE department name: {' or '.join(available_departments)}"""

    config = build_callback_config(
        trace_name="hybrid_supervisor_route",
        tags=["orchestration", "hybrid", "supervisor"],
    )
    # NOTE: Direct llm.invoke — no circuit breaker. Only specialist/synthesis nodes use it.
    response = llm.invoke(prompt, config=config)
    decision = response.content.strip().lower()

    next_department = available_departments[0]  # default
    for dept in available_departments:
        if dept in decision:
            next_department = dept
            break

    print(f"    | [Supervisor] Route -> {next_department}")

    return {
        "supervisor_routing": next_department,
        "iteration": iteration + 1,
        "routing_decisions": routing_decisions + [next_department],
    }


# ============================================================
# Cardiopulmonary P2P Cluster
# ============================================================
# Within this cluster, pulmonology and cardiology agents
# collaborate sequentially, each building on the other's findings.

def cardiopulmonary_pulmonology_node(state: HybridState) -> dict:
    """
    Pulmonology peer within the cardiopulmonary cluster.
    First peer — works from raw patient data.

    CIRCUIT BREAKER: Triggers resilience/circuit_breaker via invoke_specialist.
    """
    patient = PatientCase(**state["patient_case"])
    result = _orchestrator.invoke_specialist("pulmonology", patient)
    print(f"    |   [Pulm peer] {result.output[:80]}...")

    findings = list(state.get("cardiopulmonary_findings", []))
    findings.append(f"[PULMONOLOGY] {result.output}")

    return {"cardiopulmonary_findings": findings}


def cardiopulmonary_cardiology_node(state: HybridState) -> dict:
    """
    Cardiology peer within the cardiopulmonary cluster.
    Second peer — builds on pulmonology's findings.

    CIRCUIT BREAKER: Triggers resilience/circuit_breaker via invoke_specialist.
    """
    patient = PatientCase(**state["patient_case"])
    prior_findings = state.get("cardiopulmonary_findings", [])
    context = "\n".join(prior_findings) if prior_findings else ""

    result = _orchestrator.invoke_specialist(
        "cardiology", patient, context=context
    )
    print(f"    |   [Cardio peer] {result.output[:80]}...")

    findings = list(prior_findings)
    findings.append(f"[CARDIOLOGY] {result.output}")

    # Store cluster summary
    cluster_outputs = dict(state.get("cluster_outputs", {}))
    cluster_outputs["cardiopulmonary"] = "\n".join(findings)

    return {
        "cardiopulmonary_findings": findings,
        "cluster_outputs": cluster_outputs,
    }


# ============================================================
# Renal Specialist Node
# ============================================================

def renal_specialist_node(state: HybridState) -> dict:
    """
    Nephrology agent — operates as a single specialist.
    Uses any cardiopulmonary findings as upstream context.

    CIRCUIT BREAKER: Triggers resilience/circuit_breaker via invoke_specialist.
    """
    patient = PatientCase(**state["patient_case"])
    cluster_outputs = state.get("cluster_outputs", {})
    context = cluster_outputs.get("cardiopulmonary", "")

    result = _orchestrator.invoke_specialist(
        "nephrology", patient, context=context
    )
    print(f"    | [Nephrology] {result.output[:80]}...")

    updated_cluster_outputs = dict(cluster_outputs)
    updated_cluster_outputs["renal"] = result.output

    return {
        "renal_findings": result.output,
        "cluster_outputs": updated_cluster_outputs,
    }


# ============================================================
# Synthesis Node
# ============================================================

def hybrid_synthesis_node(state: HybridState) -> dict:
    """Synthesize findings from all departments.
    CIRCUIT BREAKER: Triggers resilience/circuit_breaker via invoke_synthesizer."""
    patient = PatientCase(**state["patient_case"])
    cluster_outputs = state.get("cluster_outputs", {})

    results = []
    for dept, output in cluster_outputs.items():
        results.append(OrchestrationResult(
            agent_name=f"{dept}_department",
            specialty=dept,
            output=output,
        ))

    report = _orchestrator.invoke_synthesizer(results, patient)
    print(f"    | [Synthesis] Report: {len(report)} chars")

    return {"final_report": report}
