"""
scripts/ — MAS Pattern Demonstration Package
=============================================
All scripts in this package are PATTERN DEMONSTRATIONS.
Root modules (guardrails/, hitl/, memory/, orchestration/,
resilience/, core/, tools/, observability/, agents/) are the
COMPONENT IMPLEMENTATIONS that these scripts import from.

Architecture rule:
    root module/    = component implementation (reusable library)
    scripts/<area>/ = pattern demonstrations (how to use it)

MASTER LEARNING PATH (see scripts/README.md for full detail)
  Area 1: tools/                      — tool binding, ToolNode, structured output
  Area 2: handoff/                    — pipeline, routing, supervisor, fan-out
  Area 3: guardrails/                 — input/output validation, confidence, judge
  Area 4: HITL/                       — interrupt/resume, approval, escalation
  Area 5: memory_management/          — scratchpad, RAG, checkpoints, conversation
  Area 6: orchestration/              — 5 LLM orchestration patterns + resilience
  Area 7: MAS_architectures/          — 7 system-level architecture patterns
  Area 8: observability_and_traceability/ — tracing, metrics, evaluation, sessions
  Area 9: resilience/ (embedded in orchestration) — no standalone scripts
"""
