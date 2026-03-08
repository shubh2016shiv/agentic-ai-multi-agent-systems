"""
Tools Pattern Scripts — Area 1 of 9
=====================================
Root component module: tools/
These scripts demonstrate HOW to bind, invoke, and handle tools in
LangGraph — the patterns, not the tool implementations themselves.

WHERE THIS FITS IN THE LEARNING SEQUENCE
  Area 1 of 9 — start here before handoff or orchestration patterns.
  Foundation: you must understand tool binding before you can build
  multi-agent workflows that use tools.

SCRIPTS IN THIS PACKAGE — RECOMMENDED ORDER

  1. tool_binding.py
     Context-scope tools per-agent via bind_tools().
     Use when: you have multiple agents that should each see only
     their relevant tools (triage vs pharmacology vs guidelines).

  2. toolnode_patterns.py
     ToolNode as a graph node vs ToolNode.invoke() inside a node.
     Use when: choosing whether tool execution should be visible
     in the graph topology (Pattern 1) or encapsulated (Pattern 2).

  3. structured_output.py
     Get validated Pydantic models from LLMs instead of free text.
     Use when: downstream code needs reliable structured data from
     the LLM, not a free-form string.

  4. dynamic_tool_selection.py
     Select the tool set at runtime based on patient/input data.
     Use when: the same agent may need different tools depending on
     what the input contains (static binding is not enough).

  5. tool_error_handling.py
     Handle tool failures gracefully with fallback and retry logic.
     Use when: tools can fail (network timeouts, bad input, service
     errors) and the agent should self-correct rather than crash.

ROOT MODULE CONNECTION
  tools/          — clinical tool functions (analyze_symptoms, etc.)
  core/config     — get_llm() centralises LLM instantiation
  core/models     — PatientCase is the canonical domain model
  observability/  — build_callback_config() injects Langfuse tracing
"""
