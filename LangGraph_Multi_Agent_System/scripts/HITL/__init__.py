"""
Human-in-the-Loop (HITL) Pattern Scripts — Area 4 of 9
========================================================
Root component module: hitl/
These scripts demonstrate HOW to wire interrupt/resume cycles into
LangGraph graphs — the patterns, not the HITL primitives themselves.

WHERE THIS FITS IN THE LEARNING SEQUENCE
  Area 4 of 9 — study after guardrails/ (Area 3).
  HITL brings a human into the pipeline at critical checkpoints.
  Prerequisite: guardrails patterns (scripts/guardrails/).

SCRIPTS IN THIS PACKAGE — RECOMMENDED ORDER

  1. basic_approval.py
     Core interrupt/resume mechanics: agent output pauses for approval.
     Use when: start here — every other HITL pattern builds on this.
     Root module: hitl/primitives.py → build_approval_payload(), parse_resume_action()
                  hitl/run_cycle.py  → run_hitl_cycle()

  2. tool_call_confirmation.py
     Human confirms before a tool is actually executed.
     Use when: tools have side effects (send email, write DB, call API)
     that must be human-approved before execution.
     Root module: hitl/primitives.py → build_tool_payload()

  3. edit_before_approve.py
     Human can modify the agent's output before approving it.
     Use when: agent output is usually good but occasionally needs
     minor corrections that the human should make in-line.
     Root module: hitl/primitives.py → build_edit_payload()

  4. multi_step_approval.py
     Multiple sequential interrupt checkpoints in one workflow.
     Use when: critical workflows need approval at each milestone
     (e.g., approve assessment, then approve treatment plan).
     Root module: hitl/run_cycle.py → run_multi_interrupt_cycle()

  5. escalation_chain.py
     Junior reviews first; senior review triggered if rejected.
     Use when: you need a tiered review structure where escalation
     to a senior reviewer happens automatically on first rejection.
     Root module: hitl/primitives.py → build_escalation_payload()

ROOT MODULE CONNECTION
  hitl/primitives.py    — InterruptPayload, ResumeAction, build_*_payload(),
                          parse_resume_action()
  hitl/run_cycle.py     — run_hitl_cycle(), run_multi_interrupt_cycle(),
                          display_interrupt_payload()
  hitl/review_nodes.py  — factory functions for reusable review nodes
"""
