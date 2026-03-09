"""
Orchestration Pattern Scripts — Area 6 of 9
============================================
Root component modules: orchestration/, resilience/
These scripts demonstrate HOW to compose the 5 orchestration patterns using
BaseOrchestrator — the patterns, not the orchestration/resilience implementations.

WHERE THIS FITS IN THE LEARNING SEQUENCE
  Area 6 of 9 — study after memory_management/ (Area 5).
  Orchestration coordinates multiple specialist agents for a shared goal.
  Resilience is embedded inside orchestration (no standalone resilience scripts).
  Prerequisite: understanding of supervisor handoff (scripts/handoff/supervisor.py).

SCRIPTS IN THIS PACKAGE — RECOMMENDED ORDER

Each pattern follows a 4-file layout (read in this order within each pattern):
  X.1  models.py   — state TypedDict definition
  X.2  agents.py   — orchestrator subclass + node functions
  X.3  graph.py    — graph topology and wiring only
  X.4  runner.py   — execution entry point and scenario

Patterns (study in STAGE order):

  STAGE 1: supervisor_orchestration/
     Central supervisor LLM routes to specialist workers dynamically.
     Use when: routing order depends on results; centralized accountability.
     When NOT to use: all agents should run in parallel (use graph_of_subgraphs).

  STAGE 2: peer_to_peer_orchestration/
     No supervisor — agents share findings via a shared_findings list.
     Use when: agents build on each other sequentially without a controller.
     When NOT to use: routing must be adaptive (use supervisor).

  STAGE 3: dynamic_router_orchestration/
     One-shot LLM classification decides which specialist(s) handle the case.
     Use when: input type determines which specialist is needed (one decision).
     When NOT to use: multiple routing decisions are needed (use supervisor).

  STAGE 4: graph_of_subgraphs_orchestration/
     Each specialty runs a multi-step subgraph (assess → risk → recommend).
     Use when: each specialty needs its own internal workflow with multiple steps.
     When NOT to use: single-step specialist calls are sufficient.

  STAGE 5: hybrid_orchestration/
     Supervisor routes at department level; P2P within each department cluster.
     Use when: org structure is two-tiered (department + specialist team).
     When NOT to use: simpler patterns meet your needs (adds graph complexity).

RESILIENCE INTEGRATION
  All LLM calls in every pattern go through _ORCHESTRATION_CALLER (ResilientCaller)
  in orchestration/orchestrator.py. The 6-layer resilience stack (token budget →
  bulkhead → rate limiter → circuit breaker → retry → timeout) is transparent
  to the pattern scripts. See resilience/__init__.py for the full explanation.

ROOT MODULE CONNECTION
  orchestration/orchestrator.py — BaseOrchestrator, SPECIALIST_SYSTEM_PROMPTS,
                                  _ORCHESTRATION_CALLER (ResilientCaller façade)
  orchestration/models.py       — OrchestrationResult, PatientWorkload, SHARED_PATIENT,
                                  format_patient_for_prompt()
  resilience/                   — 6-layer resilience stack (embedded, not standalone)
"""
