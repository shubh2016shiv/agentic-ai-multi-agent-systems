"""
Guardrails Pattern Scripts — Area 3 of 9
==========================================
Root component module: guardrails/
These scripts demonstrate HOW to wire guardrails into LangGraph
graphs with conditional routing — the patterns, not the guardrail logic.

WHERE THIS FITS IN THE LEARNING SEQUENCE
  Area 3 of 9 — study after handoff/ (Area 2).
  Guardrails protect pipelines at entry (input) and exit (output).
  Prerequisite: conditional routing (scripts/handoff/conditional_routing.py).

SCRIPTS IN THIS PACKAGE — RECOMMENDED ORDER

  1. input_validation.py
     Binary pass/fail routing before the agent runs.
     Use when: inputs must be validated (PII, prompt injection, scope)
     before any LLM cost is incurred.
     Root module: guardrails/input_guardrails.py → validate_input()

  2. output_validation.py
     Validate and optionally retry agent output before it leaves.
     Use when: agent output must meet safety/format requirements.
     Root module: guardrails/output_guardrails.py → validate_output()

  3. confidence_gating.py
     LLM reports its own confidence; three-way routing on threshold.
     Use when: borderline responses need human review or escalation.
     Root module: guardrails/confidence_guardrails.py → gate_on_confidence()

  4. llm_as_judge.py
     A second LLM evaluates the first LLM's output semantically.
     Use when: rule-based output checks are insufficient and you need
     semantic evaluation (safety, relevance, completeness).
     Root module: guardrails/llm_judge_guardrails.py → evaluate_with_judge()

  5. layered_validation.py
     Stack input + output guardrails in one pipeline (defence-in-depth).
     Use when: you need both entry and exit protection simultaneously.
     Root modules: guardrails/input_guardrails.py + output_guardrails.py

ROOT MODULE CONNECTION
  guardrails/input_guardrails.py   — validate_input(), detect_pii(), etc.
  guardrails/output_guardrails.py  — validate_output(), check_safety_disclaimers()
  guardrails/confidence_guardrails.py — extract_confidence(), gate_on_confidence()
  guardrails/llm_judge_guardrails.py  — JudgeVerdict, evaluate_with_judge()
"""
