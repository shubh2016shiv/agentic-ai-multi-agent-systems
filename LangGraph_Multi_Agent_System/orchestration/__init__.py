"""
orchestration — Component Implementation Module
================================================
Reusable orchestration building blocks for multi-agent workflows.

════════════════════════════════════════
  WHERE THIS FITS IN THE MAS ARCHITECTURE
════════════════════════════════════════

This is the ROOT component module for all orchestration logic.
Pattern scripts in scripts/orchestration/ import from here,
not the other way around.

    ┌─────────────────────────────────────────────────────────────┐
    │                  MAS Architecture                           │
    │                                                             │
    │  orchestration/          ← ROOT COMPONENT MODULE           │
    │  ├── models.py           ← Shared data contracts           │
    │  │   ├── OrchestrationResult  (standard agent output)      │
    │  │   ├── PatientWorkload      (case + routing metadata)     │
    │  │   ├── SHARED_PATIENT       (demo patient for all 5)      │
    │  │   └── format_patient_for_prompt()                        │
    │  └── orchestrator.py     ← BaseOrchestrator + resilience   │
    │      ├── BaseOrchestrator (ABC: invoke_specialist/synth)    │
    │      ├── SPECIALIST_SYSTEM_PROMPTS  (centralized prompts)  │
    │      ├── _ORCHESTRATION_LLM_BREAKER (shared circuit breaker)│
    │      └── _ORCHESTRATION_CALLER     (ResilientCaller façade) │
    │                                                             │
    │  scripts/orchestration/  ← PATTERN DEMONSTRATIONS          │
    │  ├── supervisor_orchestration/   (STAGE 1.x)               │
    │  ├── peer_to_peer_orchestration/ (STAGE 2.x)               │
    │  ├── dynamic_router_orchestration/ (STAGE 3.x)             │
    │  ├── graph_of_subgraphs_orchestration/ (STAGE 4.x)         │
    │  └── hybrid_orchestration/       (STAGE 5.x)               │
    └─────────────────────────────────────────────────────────────┘

════════════════════════════════════════
  HOW RESILIENCE INTEGRATES WITH ORCHESTRATION
════════════════════════════════════════

Resilience is NOT a separate layer — it is EMBEDDED in orchestration.
The flow for every LLM call in any orchestration pattern:

    Pattern node (e.g. pulmonology_worker_node)
        ↓ calls _orchestrator.invoke_specialist()
    BaseOrchestrator.invoke_specialist()
        ↓ calls _ORCHESTRATION_CALLER.call(llm.invoke, prompt)
    ResilientCaller.call() — 6-layer stack (outer → inner):
        ↓ [1] Token Budget check (fail fast if over budget)
        ↓ [2] Bulkhead (SKIPPED for linear flows)
        ↓ [3] Rate Limiter (ENABLED — smooths request bursts)
        ↓ [4] Circuit Breaker (shared; fail fast if API is down)
        ↓ [5] Retry Handler (auto-retry on transient 429/timeout)
        ↓ [6] Timeout Guard (30s per-call deadline, innermost)
    llm.invoke(prompt)    ← the ACTUAL LLM API call

If a resilience exception is raised, invoke_specialist() maps it
to OrchestrationResult(was_successful=False) so synthesis can skip
it and continue. invoke_synthesizer() re-raises as RuntimeError.

CONNECTION: resilience/resilient_caller.py — ResilientCaller is the façade.
CONNECTION: resilience/circuit_breaker.py  — CircuitBreakerRegistry stores
            the shared _ORCHESTRATION_LLM_BREAKER instance.

════════════════════════════════════════
  LEARNING SEQUENCE
════════════════════════════════════════

To study orchestration patterns in order:

  Step 1: Read this module (orchestration/)
    - models.py: understand data contracts
    - orchestrator.py: understand BaseOrchestrator + resilience integration

  Step 2: Read resilience/ module
    - __init__.py: pattern overview
    - resilient_caller.py: the façade that BaseOrchestrator uses
    - circuit_breaker.py: how fail-fast works in practice

  Step 3: Read pattern scripts in order (STAGE 1→5)
    - STAGE 1.x: supervisor_orchestration/  (centralized routing)
    - STAGE 2.x: peer_to_peer_orchestration/ (decentralized)
    - STAGE 3.x: dynamic_router_orchestration/ (one-shot classification)
    - STAGE 4.x: graph_of_subgraphs_orchestration/ (nested graphs)
    - STAGE 5.x: hybrid_orchestration/ (supervisor + P2P combined)

  Within each STAGE:
    X.1 models.py    → state definition
    X.2 agents.py    → agent/node definitions (imports from orchestration/)
    X.3 graph.py     → LangGraph wiring
    X.4 runner.py    → execution entry point
"""

from orchestration.models import (
    OrchestrationResult,
    PatientWorkload,
    SHARED_PATIENT,
    SHARED_WORKLOAD,
    format_patient_for_prompt,
)
from orchestration.orchestrator import (
    BaseOrchestrator,
    SPECIALIST_SYSTEM_PROMPTS,
)

__all__ = [
    # ── Data contracts ────────────────────────────────────────
    "OrchestrationResult",
    "PatientWorkload",
    "SHARED_PATIENT",
    "SHARED_WORKLOAD",
    "format_patient_for_prompt",
    # ── Base class + prompts ──────────────────────────────────
    "BaseOrchestrator",
    "SPECIALIST_SYSTEM_PROMPTS",
]
