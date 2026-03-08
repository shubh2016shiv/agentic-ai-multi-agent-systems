"""
Observability & Traceability Pattern Scripts — Area 8 of 9
============================================================
Root component module: observability/
These scripts demonstrate HOW to instrument LangGraph pipelines with
tracing, metrics, evaluation, and session management — the patterns, not
the observability implementation.

WHERE THIS FITS IN THE LEARNING SEQUENCE
  Area 8 of 9 — study after MAS_architectures/ (Area 7).
  Production systems need observability to monitor cost, quality, and reliability.
  Prerequisite: a working pipeline (any from Areas 2-7).

WHAT OBSERVABILITY VS TRACEABILITY MEANS HERE
  Traceability:   which agent made which LLM call, which tools were invoked
  Observability:  token usage per agent, cost per workflow, latency, tool
                  success/failure rates, output quality scores

SCRIPTS IN THIS PACKAGE — RECOMMENDED ORDER

  STAGE 1: trace_hierarchy.py
     Parent/child trace structure — span hierarchy in Langfuse.
     Use when: start here — understand how traces nest before adding metrics.
     Root module: observability/callbacks.py → build_callback_config()

  STAGE 2: agent_metrics_and_cost.py
     Per-agent token usage, latency, and cost tracking with MetricsCollector.
     Use when: you need to know which agent consumes most tokens (cost driver)
     or which is the slowest (latency bottleneck).
     Root module: observability/metrics.py → MetricsCollector

  STAGE 3: trace_scoring_and_evaluation.py
     Rule-based and LLM-based quality scoring of agent outputs.
     Use when: you need automated quality gates on agent output in production.
     Root module: observability/callbacks.py, observability/metrics.py

  STAGE 4: session_based_tracing.py
     Group traces by user session ID for multi-turn conversation analysis.
     Use when: you run multi-turn conversations and need to see the full
     session in Langfuse rather than isolated individual traces.
     Root module: observability/callbacks.py → build_callback_config()

  STAGE 5: observed_clinical_pipeline.py
     Full pipeline with all observability patterns combined (production-style).
     Use when: study this last — it composes trace hierarchy, metrics,
     scoring, and session tracing into a complete instrumented workflow.
     Root modules: observability/callbacks.py, observability/metrics.py

ROOT MODULE CONNECTION
  observability/callbacks.py — build_callback_config() injects Langfuse
                               trace_name, tags, and session_id into every
                               LLM call automatically via LangChain callbacks
  observability/metrics.py   — MetricsCollector tracks token usage, latency,
                               cost, and tool success/failure rates per agent
"""
