"""
============================================================
Dynamic Router Orchestration — Agent Definitions
============================================================
The router classifies the case and routes to the appropriate
specialist in a ONE-SHOT decision. No loop-back.

Agents:
    - InputClassifier: LLM that classifies the case type
    - Specialist workers: pulmonology, cardiology, nephrology
    - Fallback: general assessment if classification fails

CIRCUIT BREAKER (resilience/circuit_breaker.py):
    DESTINATION — these nodes trigger the circuit breaker via invoke_specialist/
    invoke_synthesizer → ResilientCaller → circuit_breaker.call():
        pulmonology_specialist_node, cardiology_specialist_node,
        nephrology_specialist_node, router_report_node.
    NOT triggered: input_classifier_node (uses llm.invoke directly).

RATE LIMITER (resilience/rate_limiter.py):
    DESTINATION — same nodes as circuit breaker. ENABLED (skip_rate_limiter=False)
    in orchestrator base.
============================================================
"""

# ============================================================
# STAGE 3.2 — Agent Definitions
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
from scripts.orchestration.dynamic_router_orchestration.models import DynamicRouterState, CaseClassification


class DynamicRouterOrchestrator(BaseOrchestrator):
    """
    One-shot routing orchestrator.

    Classifies the input ONCE, routes to the appropriate
    specialist(s), and synthesizes the result. No loops.
    """

    @property
    def pattern_name(self) -> str:
        return "dynamic_router"

    @property
    def description(self) -> str:
        return "One-shot input classification routes to appropriate specialist(s)"


_orchestrator = DynamicRouterOrchestrator()


# ============================================================
# Input Classifier Node
# ============================================================

def input_classifier_node(state: DynamicRouterState) -> dict:
    """
    Classify the patient case to determine routing.

    The classifier analyzes the chief complaint, symptoms,
    and labs to determine:
        - Which specialty is MOST relevant (primary)
        - Whether a second specialty should also assess (secondary)
        - Urgency level for prioritization

    This is a ONE-SHOT decision — the classifier runs once
    and the rest of the graph follows its routing.
    """
    patient = state["patient_case"]
    llm = get_llm()

    classification_prompt = f"""You are a clinical case router. Analyze this patient and classify which specialty should assess them.

Patient: {patient.get('age')}y {patient.get('sex')}
Chief Complaint: {patient.get('chief_complaint')}
Symptoms: {', '.join(patient.get('symptoms', []))}
History: {', '.join(patient.get('medical_history', []))}
Labs: {json.dumps(patient.get('lab_results', {}))}
Vitals: {json.dumps(patient.get('vitals', {}))}

Available specialties: pulmonology, cardiology, nephrology

Respond in this exact format:
PRIMARY: [specialty name]
SECONDARY: [specialty name or NONE]
URGENCY: [emergency/urgent/standard]
REASONING: [one sentence explaining your routing decision]"""

    config = build_callback_config(
        trace_name="dynamic_router_classify",
        tags=["orchestration", "dynamic_router", "classification"],
    )
    # NOTE: Direct llm.invoke — no circuit breaker. Only specialist/report nodes use it.
    response = llm.invoke(classification_prompt, config=config)
    response_text = response.content.strip()

    # Parse the structured response
    primary = "pulmonology"  # default
    secondary = ""
    urgency = "standard"
    reasoning = ""

    for line in response_text.split("\n"):
        line_lower = line.strip().lower()
        if line_lower.startswith("primary:"):
            value = line.split(":", 1)[1].strip().lower()
            if value in ["pulmonology", "cardiology", "nephrology"]:
                primary = value
        elif line_lower.startswith("secondary:"):
            value = line.split(":", 1)[1].strip().lower()
            if value in ["pulmonology", "cardiology", "nephrology"]:
                secondary = value
            elif "none" in value:
                secondary = ""
        elif line_lower.startswith("urgency:"):
            urgency = line.split(":", 1)[1].strip().lower()
        elif line_lower.startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()

    classification = CaseClassification(
        primary_specialty=primary,
        secondary_specialty=secondary,
        urgency=urgency,
        reasoning=reasoning,
    )

    print(f"    | [Classifier] Primary: {primary}, Secondary: {secondary or 'NONE'}, Urgency: {urgency}")
    print(f"    | [Classifier] Reasoning: {reasoning[:100]}")

    return {"classification": classification.model_dump()}


# ============================================================
# Specialist Worker Nodes
# ============================================================

def pulmonology_specialist_node(state: DynamicRouterState) -> dict:
    """Pulmonology specialist routed by the classifier.
    CIRCUIT BREAKER: Triggers resilience/circuit_breaker via invoke_specialist."""
    patient = PatientCase(**state["patient_case"])
    classification = state.get("classification", {})
    is_primary = classification.get("primary_specialty") == "pulmonology"

    result = _orchestrator.invoke_specialist("pulmonology", patient)
    output_field = "primary_output" if is_primary else "secondary_output"
    print(f"    | [Pulmonology] ({'primary' if is_primary else 'secondary'}): {result.output[:100]}...")

    return {output_field: result.output}


def cardiology_specialist_node(state: DynamicRouterState) -> dict:
    """Cardiology specialist routed by the classifier.
    CIRCUIT BREAKER: Triggers resilience/circuit_breaker via invoke_specialist."""
    patient = PatientCase(**state["patient_case"])
    classification = state.get("classification", {})
    is_primary = classification.get("primary_specialty") == "cardiology"

    result = _orchestrator.invoke_specialist("cardiology", patient)
    output_field = "primary_output" if is_primary else "secondary_output"
    print(f"    | [Cardiology] ({'primary' if is_primary else 'secondary'}): {result.output[:100]}...")

    return {output_field: result.output}


def nephrology_specialist_node(state: DynamicRouterState) -> dict:
    """Nephrology specialist routed by the classifier.
    CIRCUIT BREAKER: Triggers resilience/circuit_breaker via invoke_specialist."""
    patient = PatientCase(**state["patient_case"])
    classification = state.get("classification", {})
    is_primary = classification.get("primary_specialty") == "nephrology"

    result = _orchestrator.invoke_specialist("nephrology", patient)
    output_field = "primary_output" if is_primary else "secondary_output"
    print(f"    | [Nephrology] ({'primary' if is_primary else 'secondary'}): {result.output[:100]}...")

    return {output_field: result.output}


# ============================================================
# Report Node
# ============================================================

def router_report_node(state: DynamicRouterState) -> dict:
    """Synthesize primary and optional secondary findings.
    CIRCUIT BREAKER: Triggers resilience/circuit_breaker via invoke_synthesizer."""
    patient = PatientCase(**state["patient_case"])
    classification = state.get("classification", {})
    primary_output = state.get("primary_output", "")
    secondary_output = state.get("secondary_output", "")

    results = []
    if primary_output:
        results.append(OrchestrationResult(
            agent_name=f"{classification.get('primary_specialty', 'primary')}_specialist",
            specialty=classification.get("primary_specialty", "primary"),
            output=primary_output,
        ))
    if secondary_output:
        results.append(OrchestrationResult(
            agent_name=f"{classification.get('secondary_specialty', 'secondary')}_specialist",
            specialty=classification.get("secondary_specialty", "secondary"),
            output=secondary_output,
        ))

    report = _orchestrator.invoke_synthesizer(results, patient)
    print(f"    | [Report] Synthesized: {len(report)} chars")
    return {"final_report": report}
