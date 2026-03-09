# Memory System Documentation 

> **A 7-part masterclass on LangGraph's Persistence constraints and RAG Architectures.**

This module provides a robust multi-tier architecture to solve the fact that AI Agents inherently have the memory of a goldfish. 

To properly leverage this package, you must understand the Four Tiers of Memory: LangGraph State, Working Scratchpads, Episodic Summaries, and Persistent Semantic Vector Stores. 

It is highly recommended to read these chapters in order.

### Documentation Index

1. **[Chapter 1: The Big Picture](01_big_picture.md)** — What problem does this solve? Our Real-World analogy (Hospital Staff) and the 4-tier storage architecture.
2. **[Chapter 2: Core Concepts](02_core_concepts.md)** — Episodic vs Semantic memory, how the LangGraph Checkpointer works, and resolving the endless context window via Rolling Summarization.
3. **[Chapter 3: Deep Dive](03_deep_dive.md)** — A low-level look at `working_memory.py`, `conversation_memory.py`, and `long_term_memory.py`.
4. **[Chapter 4: Design Decisions](04_design_decisions.md)** — Why we count messages instead of tokens, and why we block ChromaDB's local model inference in favor of API embeddings.
5. **[Chapter 5: Gotchas & Anti-Patterns](05_gotchas.md)** — Traps to avoid, like confusing Checkpointers with Vector Stores, and destroying your LLM via over-summarization.
6. **[Chapter 6: Putting It All Together](06_putting_it_together.md)** — The golden-path integration pattern showing all 4 tiers running synchronously. 
7. **[Chapter 7: Quick Reference](07_quick_reference.md)** — API tables and copy-paste snippets.
