# Orchestration System Documentation 

> **A 7-part masterclass on Multi-Agent Architectures and Silent Resilience.**

This module serves as the foundational root for all multi-agent routing. It does not contain independent runnable scripts; rather, it contains the `BaseOrchestrator` and data schemas (`OrchestrationResult`) that five radically different architectures use to operate.

It is highly recommended to read these chapters in order.

### Documentation Index

1. **[Chapter 1: The Big Picture](01_big_picture.md)** — What problem does this solve? Understand the difference between Specialists and Synthesizers, and the 6-Layer Invisible Resilience Shield.
2. **[Chapter 2: Core Concepts](02_core_concepts.md)** — Data Contracts: Why `OrchestrationResult` prevents the system from crashing, and why all 5 patterns share a hardcoded patient.
3. **[Chapter 3: Deep Dive](03_deep_dive.md)** — A low-level look at `BaseOrchestrator.invoke_specialist` and how building a customized LLM wrapper instantly makes all child patterns robust.
4. **[Chapter 4: Design Decisions](04_design_decisions.md)** — Why we banned `@retry` decorators from LangGraph nodes and centralized them in the Orchestrator, and why the Synthesizer crashes hard.
5. **[Chapter 5: Gotchas & Anti-Patterns](05_gotchas.md)** — The Parallel Node Token Explosion, unstructured dictionary returns, and infinite Peer-to-Peer review loops. 
6. **[Chapter 6: Putting It All Together](06_putting_it_together.md)** — A theoretical breakdown of how the 5 standalone scripts (Supervisor, P2P, Dynamic, Subgraphs, Hybrid) work.
7. **[Chapter 7: Quick Reference](07_quick_reference.md)** — API tables and concept summaries.
