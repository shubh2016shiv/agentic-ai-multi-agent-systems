"""
MAS Architecture Pattern Scripts — Area 7 of 9
================================================
Root component modules: agents/, core/, observability/
These scripts demonstrate HOW to compose the 7 higher-level multi-agent
system architecture patterns using pre-built agent objects.

WHERE THIS FITS IN THE LEARNING SEQUENCE
  Area 7 of 9 — study after orchestration/ (Area 6).
  These are system-level patterns that compose on top of the orchestration
  primitives. They use agent objects from agents/ as their building blocks.
  Prerequisite: orchestration patterns (scripts/orchestration/).

SCRIPTS IN THIS PACKAGE — RECOMMENDED ORDER

  STAGE 1: supervisor_orchestration.py
     Central supervisor LLM routes to specialist agents dynamically.
     Use when: task order is NOT predetermined; centralized control.
     When NOT to use: fixed order is fine (use sequential_pipeline).

  STAGE 2: sequential_pipeline.py
     Fixed sequential flow — no supervisor, no dynamic routing.
     Use when: every case follows the same agent order.
     When NOT to use: routing should adapt to results (use supervisor).

  STAGE 3: parallel_voting.py
     Multiple agents produce independent answers; majority vote wins.
     Use when: you need consensus from multiple perspectives with
     low bias risk; accuracy matters more than speed.
     When NOT to use: reasoning needs to build on prior answers (use debate).

  STAGE 4: adversarial_debate.py
     Two agents argue opposing viewpoints; a judge agent rules.
     Use when: anchoring bias is a real risk; documented rationale
     for both sides is required (complex clinical decisions).
     When NOT to use: straightforward decisions (use pipeline/voting).

  STAGE 5: hierarchical_delegation.py
     Multi-level org structure: L3 specialists → L2 leads → L1 executive.
     Use when: the problem decomposes naturally into management layers
     with different granularity of information at each level.
     When NOT to use: flat team structure (use voting or pipeline).

  STAGE 6: map_reduce_fanout.py
     Parallel sub-tasks (map) aggregated by a reducer (reduce).
     Use when: a large task can be split into independent sub-tasks
     processed in parallel, then merged into a final answer.
     When NOT to use: sub-tasks depend on each other's results.

  STAGE 7: reflection_self_critique.py
     Agent critiques its own output and iterates until quality threshold.
     Use when: output quality is critical and a single pass is insufficient;
     self-correction loop improves accuracy without extra agents.
     When NOT to use: iteration budget is tight (cost of multiple passes).

ROOT MODULE CONNECTION
  agents/          — TriageAgent, DiagnosticAgent, PharmacistAgent (pre-built
                     reusable agent objects; these scripts use them, not define them)
  core/config      — get_llm() centralises LLM instantiation
  core/models      — PatientCase is the canonical domain model
  observability/   — build_callback_config() injects Langfuse tracing into LLM calls
"""
