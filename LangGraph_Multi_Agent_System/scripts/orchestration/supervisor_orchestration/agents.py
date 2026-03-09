"""
============================================================
Supervisor Orchestration — Agent Definitions
============================================================
The supervisor agent and specialist worker agents.

Separation of concern:
    This file defines ONLY the agent behavior (how they
    process state). Graph wiring is in graph.py.

Agents:
    - SupervisorOrchestrator: decides which worker runs next
    - Specialist workers: pulmonology, cardiology, nephrology
      (use BaseOrchestrator.invoke_specialist under the hood)

CIRCUIT BREAKER (resilience/circuit_breaker.py):
    DESTINATION — these nodes trigger the circuit breaker via invoke_specialist/
    invoke_synthesizer → ResilientCaller → circuit_breaker.call():
        pulmonology_worker_node, cardiology_worker_node, nephrology_worker_node,
        report_synthesis_node.
    NOT triggered: supervisor_decide_node (uses llm.invoke directly).

RATE LIMITER (resilience/rate_limiter.py):
    DESTINATION — same nodes as circuit breaker. ENABLED (skip_rate_limiter=False)
    in orchestrator base. Smooths bursts when multiple workers hit LLM API.

TOKEN MANAGER (resilience/token_manager.py):
    DESTINATION — same nodes as circuit breaker. ENABLED (via token_manager param).
    Enforces per-workflow budget (check before call, record after call).
    See TOKEN_BUDGET_GUIDE.md for detailed rationale and integration pattern.
============================================================
"""

# ============================================================
# STAGE 1.2 — Agent Definitions
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
from scripts.orchestration.supervisor_orchestration.models import SupervisorState


# ============================================================
# Supervisor Agent
# ============================================================

class SupervisorOrchestrator(BaseOrchestrator):
    """
    Central coordinator that decides which specialist runs next.

    The supervisor:
        1. Sees the patient case and all completed work
        2. Decides which remaining specialist should run next
        3. Returns "FINISH" when all needed work is done

    This implements the abstract BaseOrchestrator contract.
    """

    @property
    def pattern_name(self) -> str:
        return "supervisor"

    @property
    def description(self) -> str:
        return "Central coordinator dynamically routes to specialist workers"

    def decide_next_worker(self, state: SupervisorState) -> dict:
        """
        Supervisor routing logic — the core of this pattern.

        Args:
            state: current LangGraph state

        Returns:
            dict with next_worker and updated iteration
        """
        completed = state.get("completed_workers", [])
        worker_outputs = state.get("worker_outputs", {})
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 6)
        patient = state["patient_case"]

        available_specialists = ["pulmonology", "cardiology", "nephrology"]
        remaining = [specialist for specialist in available_specialists if specialist not in completed]

        # Guard: force finish if limit reached or no workers remain
        if iteration >= max_iterations or not remaining:
            finish_reason = "max iterations" if iteration >= max_iterations else "all specialists completed"
            print(f"    | [Supervisor] Finishing: {finish_reason}")
            return {"next_worker": "FINISH", "iteration": iteration + 1}

        # Build context for the supervisor LLM
        completed_summary = ""
        if worker_outputs:
            completed_summary = "\n".join(
                f"- {specialty}: {output[:200]}" for specialty, output in worker_outputs.items()
            )

        supervisor_prompt = f"""You are a clinical supervisor coordinating a multi-specialty patient assessment.

Patient: {patient.get('age')}y {patient.get('sex')}, {patient.get('chief_complaint')}
History: {', '.join(patient.get('medical_history', []))}
Labs: {json.dumps(patient.get('lab_results', {}))}

Completed specialists: {', '.join(completed) if completed else 'None yet'}
Remaining specialists: {', '.join(remaining)}

{f'Findings so far:{chr(10)}{completed_summary}' if completed_summary else ''}

Which specialist should assess this patient NEXT?
Consider clinical priority:
  - Acute symptoms first (respiratory distress -> pulmonology)
  - Then cardiac status (BNP elevation -> cardiology)
  - Then renal/medication safety (CKD -> nephrology)

Respond with ONLY the specialty name: {' or '.join(remaining)}
Or respond with "FINISH" if all essential assessments are complete."""

        config = build_callback_config(
            trace_name="supervisor_routing_decision",
            tags=["orchestration", "supervisor", "routing"],
        )
        llm = get_llm()
        # NOTE: Direct llm.invoke — no circuit breaker. Only worker nodes use it.
        response = llm.invoke(supervisor_prompt, config=config)
        decision = response.content.strip().lower()

        # Parse the routing decision
        if "finish" in decision:
            next_worker = "FINISH"
        else:
            next_worker = None
            for specialty in remaining:
                if specialty in decision:
                    next_worker = specialty
                    break
            if next_worker is None:
                next_worker = remaining[0]  # fallback to first remaining

        print(f"    | [Supervisor] Iteration {iteration + 1}: route -> {next_worker}")
        return {"next_worker": next_worker, "iteration": iteration + 1}


# ============================================================
# Specialist Worker Nodes
# ============================================================
# Each worker delegates to BaseOrchestrator.invoke_specialist()
# and updates state with its results.

_orchestrator = SupervisorOrchestrator()


def pulmonology_worker_node(state: SupervisorState) -> dict:
    """Pulmonology specialist — delegates to shared invoke_specialist.
    
    CIRCUIT BREAKER: This node triggers resilience/circuit_breaker via
    invoke_specialist → ResilientCaller → circuit_breaker.call().
    
    TOKEN MANAGER DESTINATION: This node triggers token budget check/record via
    invoke_specialist(token_manager=state["token_manager"]).
    
    SHIELD RATIONALE: Prevents pulm specialist from consuming entire workflow
    budget before cardio/nephro can run. Per-agent cap also applies.
    """
    patient = PatientCase(**state["patient_case"])
    token_manager = state.get("token_manager")
    upstream_context = "\n".join(
        f"{k}: {v[:150]}" for k, v in state.get("worker_outputs", {}).items()
    )

    result = _orchestrator.invoke_specialist(
        "pulmonology",
        patient,
        context=upstream_context,
        token_manager=token_manager,
    )
    print(f"    | [Pulmonology] {result.output[:100]}...")

    completed = list(state.get("completed_workers", []))
    completed.append("pulmonology")
    outputs = dict(state.get("worker_outputs", {}))
    outputs["pulmonology"] = result.output

    return {"completed_workers": completed, "worker_outputs": outputs}


def cardiology_worker_node(state: SupervisorState) -> dict:
    """Cardiology specialist — delegates to shared invoke_specialist.
    
    CIRCUIT BREAKER: This node triggers resilience/circuit_breaker via
    invoke_specialist → ResilientCaller → circuit_breaker.call().
    
    TOKEN MANAGER DESTINATION: This node triggers token budget check/record via
    invoke_specialist(token_manager=state["token_manager"]).
    
    SHIELD RATIONALE: Prevents cardio specialist from consuming entire workflow
    budget before nephro can run. Per-agent cap also applies.
    """
    patient = PatientCase(**state["patient_case"])
    token_manager = state.get("token_manager")
    upstream_context = "\n".join(
        f"{k}: {v[:150]}" for k, v in state.get("worker_outputs", {}).items()
    )

    result = _orchestrator.invoke_specialist(
        "cardiology",
        patient,
        context=upstream_context,
        token_manager=token_manager,
    )
    print(f"    | [Cardiology] {result.output[:100]}...")

    completed = list(state.get("completed_workers", []))
    completed.append("cardiology")
    outputs = dict(state.get("worker_outputs", {}))
    outputs["cardiology"] = result.output

    return {"completed_workers": completed, "worker_outputs": outputs}


def nephrology_worker_node(state: SupervisorState) -> dict:
    """Nephrology specialist — delegates to shared invoke_specialist.
    
    CIRCUIT BREAKER: This node triggers resilience/circuit_breaker via
    invoke_specialist → ResilientCaller → circuit_breaker.call().
    
    TOKEN MANAGER DESTINATION: This node triggers token budget check/record via
    invoke_specialist(token_manager=state["token_manager"]).
    
    SHIELD RATIONALE: Prevents nephro specialist from consuming entire workflow
    budget before synthesis. Per-agent cap also applies.
    """
    patient = PatientCase(**state["patient_case"])
    token_manager = state.get("token_manager")
    upstream_context = "\n".join(
        f"{k}: {v[:150]}" for k, v in state.get("worker_outputs", {}).items()
    )

    result = _orchestrator.invoke_specialist(
        "nephrology",
        patient,
        context=upstream_context,
        token_manager=token_manager,
    )
    print(f"    | [Nephrology] {result.output[:100]}...")

    completed = list(state.get("completed_workers", []))
    completed.append("nephrology")
    outputs = dict(state.get("worker_outputs", {}))
    outputs["nephrology"] = result.output

    return {"completed_workers": completed, "worker_outputs": outputs}


def report_synthesis_node(state: SupervisorState) -> dict:
    """Synthesize all specialist findings into a final report.
    
    CIRCUIT BREAKER: This node triggers resilience/circuit_breaker via
    invoke_synthesizer → ResilientCaller → circuit_breaker.call().
    
    TOKEN MANAGER DESTINATION: This node triggers token budget check/record via
    invoke_synthesizer(token_manager=state["token_manager"]).
    
    SHIELD RATIONALE: Synthesis is the final, most expensive call (aggregates
    all specialist outputs = longest prompt). Failing here saves the most tokens.
    """
    worker_outputs = state.get("worker_outputs", {})
    patient = PatientCase(**state["patient_case"])
    token_manager = state.get("token_manager")

    results = [
        OrchestrationResult(
            agent_name=f"{specialty}_specialist",
            specialty=specialty,
            output=output,
        )
        for specialty, output in worker_outputs.items()
    ]

    report = _orchestrator.invoke_synthesizer(
        results,
        patient,
        token_manager=token_manager,
    )
    print(f"    | [Synthesis] Report generated: {len(report)} chars")

    return {"final_report": report}
