# Human-in-the-Loop (HITL) Documentation 

> **A 7-part masterclass on LangGraph's HITL mechanisms.**

This module provides reusable primitives and utilities for introducing human decision points into LangGraph multi-agent pipelines. 

To properly leverage this package, you must understand both the theory of LangGraph Checkpointing and the practical implementation of `interrupt()` and `Command(resume=...)`.

It is highly recommended to read these chapters in order.

### Documentation Index

1. **[Chapter 1: The Big Picture](01_big_picture.md)** — What problem does HITL solve? Our Real-World analogy (Air Traffic Control) and architectural placement.
2. **[Chapter 2: Core Concepts](02_core_concepts.md)** — Checkpointing, memory persistence, Thread IDs, and the rule of idempotency.
3. **[Chapter 3: Deep Dive](03_deep_dive.md)** — A low-level look into `primitives.py`, `review_nodes.py`, and `run_cycle.py`.
4. **[Chapter 4: Design Decisions](04_design_decisions.md)** — Why we standardize resume payloads, why we use node factories, and why WebSockets aren't included.
5. **[Chapter 5: Gotchas & Anti-Patterns](05_gotchas.md)** — The 5 traps every developer falls into when building HITL pipelines.
6. **[Chapter 6: Putting It All Together](06_putting_it_together.md)** — The golden-path implementation pattern, integrations with Guardrails, and checkpointer scaling.
7. **[Chapter 7: Quick Reference](07_quick_reference.md)** — API tables, copy-paste snippets, and graph flowcharts.
