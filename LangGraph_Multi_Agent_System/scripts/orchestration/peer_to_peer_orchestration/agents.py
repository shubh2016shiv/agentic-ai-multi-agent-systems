"""
============================================================
Peer-to-Peer Orchestration — Agent Definitions
============================================================
Peer agents that communicate via a shared findings list.
There is NO supervisor. Each agent:
    1. Reads ALL prior findings from shared_findings
    2. Produces its own assessment, building on others' work
    3. Appends its finding to shared_findings

This mirrors a clinical huddle where specialists discuss
a case sequentially, each building on what was said before.

CIRCUIT BREAKER (resilience/circuit_breaker.py):
    DESTINATION — these nodes trigger the circuit breaker via invoke_specialist/
    invoke_synthesizer → ResilientCaller → circuit_breaker.call():
        pulmonology_peer_node, cardiology_peer_node, nephrology_peer_node,
        synthesis_node.

RATE LIMITER (resilience/rate_limiter.py):
    DESTINATION — same nodes as circuit breaker. ENABLED (skip_rate_limiter=False)
    in orchestrator base. Smooths bursts when peers hit LLM API.
============================================================
"""

# ============================================================
# STAGE 2.2 — Agent Definitions
# ============================================================

import sys


from core.models import PatientCase
# CONNECTION: orchestration/ root module — BaseOrchestrator provides invoke_specialist/
# invoke_synthesizer which route all LLM calls through the 6-layer resilience stack
# (circuit breaker → retry → timeout via ResilientCaller in orchestration/orchestrator.py).
from orchestration.orchestrator import BaseOrchestrator
# CONNECTION: orchestration/models.py — OrchestrationResult is the standard agent
# output envelope shared by all 5 orchestration patterns.
from orchestration.models import OrchestrationResult
from scripts.orchestration.peer_to_peer_orchestration.models import PeerToPeerState


class PeerToPeerOrchestrator(BaseOrchestrator):
    """
    Decentralized orchestrator — no central coordinator.

    Each peer agent sees findings from ALL previous peers
    and adds its own contribution to the shared pool.
    """

    @property
    def pattern_name(self) -> str:
        return "peer_to_peer"

    @property
    def description(self) -> str:
        return "Decentralized agents communicate via shared findings list"


_orchestrator = PeerToPeerOrchestrator()


def _make_peer_node(specialty: str):
    """
    Factory function: creates a peer node for a given specialty.

    This avoids duplicating the same logic for pulmonology,
    cardiology, and nephrology. Each generated node:
        1. Reads shared_findings for all previous peer outputs
        2. Calls invoke_specialist with that context
        3. Appends its own finding to shared_findings

    CIRCUIT BREAKER: Each peer node triggers resilience/circuit_breaker via
    invoke_specialist → ResilientCaller → circuit_breaker.call().
    """
    def peer_node(state: PeerToPeerState) -> dict:
        patient = PatientCase(**state["patient_case"])
        shared_findings = state.get("shared_findings", [])

        # Build context from all previous peers' findings
        peer_context = ""
        if shared_findings:
            peer_context = "Previous specialist findings:\n" + "\n".join(
                f"  {finding}" for finding in shared_findings
            )
            peer_context += "\n\nBuild on these findings. Add your perspective."

        result = _orchestrator.invoke_specialist(
            specialty, patient, context=peer_context,
        )

        new_finding = f"[{specialty.upper()}] {result.output}"
        print(f"    | [{specialty.title()}] {result.output[:100]}...")

        updated_findings = list(shared_findings) + [new_finding]

        return {
            "shared_findings": updated_findings,
            "current_peer": specialty,
        }

    peer_node.__name__ = f"{specialty}_peer_node"
    peer_node.__doc__ = f"{specialty.title()} peer — reads prior findings and adds its own."
    return peer_node


# Create the three peer nodes
pulmonology_peer_node = _make_peer_node("pulmonology")
cardiology_peer_node = _make_peer_node("cardiology")
nephrology_peer_node = _make_peer_node("nephrology")


def synthesis_node(state: PeerToPeerState) -> dict:
    """
    Final synthesis — compiles ALL peer findings into a report.

    Unlike the supervisor pattern where a coordinator drives
    synthesis, here the shared findings list IS the coordination
    mechanism, and synthesis happens at the end.

    CIRCUIT BREAKER: This node triggers resilience/circuit_breaker via
    invoke_synthesizer → ResilientCaller → circuit_breaker.call().
    """
    patient = PatientCase(**state["patient_case"])
    shared_findings = state.get("shared_findings", [])

    results = []
    for finding in shared_findings:
        # Parse "[SPECIALTY] output" format
        if finding.startswith("[") and "]" in finding:
            bracket_end = finding.index("]")
            specialty = finding[1:bracket_end].lower()
            output = finding[bracket_end + 2:]
        else:
            specialty = "unknown"
            output = finding

        results.append(OrchestrationResult(
            agent_name=f"{specialty}_peer",
            specialty=specialty,
            output=output,
        ))

    report = _orchestrator.invoke_synthesizer(results, patient)
    print(f"    | [Synthesis] Report: {len(report)} chars")

    return {"final_report": report}
