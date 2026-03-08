"""
Handoff Pattern Scripts — Area 2 of 9
=======================================
Root component modules: core/, tools/, observability/
These scripts demonstrate HOW to wire agents together in LangGraph —
from fixed pipelines to LLM-driven dynamic routing and parallel fan-out.

WHERE THIS FITS IN THE LEARNING SEQUENCE
  Area 2 of 9 — study after tools/ (Area 1).
  These patterns show how agents hand off work to each other.
  Prerequisite: understanding of tool binding (scripts/tools/).

SCRIPTS IN THIS PACKAGE — RECOMMENDED ORDER

  1. linear_pipeline.py
     Fixed-edge pipeline: developer decides order at build time.
     Use when: agent execution order never varies. This is the
     baseline — every other handoff pattern builds on this.
     When NOT to use: if order depends on results (use conditional_routing).

  2. conditional_routing.py
     Python router function decides branching after triage output.
     Use when: routing is deterministic and testable (zero LLM cost).
     When NOT to use: if the LLM should decide routing (use command_handoff).

  3. command_handoff.py
     Agents call "transfer tools" that return Command(goto=, update=).
     The LLM decides which agent runs next.
     Use when: routing logic is too dynamic for a Python function.
     When NOT to use: depth guards are critical — add multihop_depth_guard.

  4. supervisor.py
     One supervisor LLM coordinates multiple worker agents dynamically.
     Use when: you need a central controller that decides worker order
     based on cumulative results. Most structured multi-agent pattern.
     When NOT to use: all workers should always run (use parallel_fanout).

  5. multihop_depth_guard.py
     Adds hop-count limits to Command-based chains.
     Use when: LLM-driven handoffs can chain indefinitely — always add
     a depth guard to prevent runaway loops.

  6. parallel_fanout.py
     Multiple specialist agents run concurrently; results are merged.
     Use when: agents are independent and can run simultaneously
     to reduce total latency.
     When NOT to use: agents need each other's output (use supervisor).

ROOT MODULE CONNECTION
  core/config       — get_llm() centralises LLM instantiation
  core/models       — PatientCase, HandoffContext are canonical models
  core/exceptions   — HandoffLimitReached for depth-guard enforcement
  tools/            — clinical tool functions used by worker agents
  observability/    — build_callback_config() injects Langfuse tracing
"""
