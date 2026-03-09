# Guardrails Architecture Documentation

Welcome to the Guardrails module documentation for the LangGraph Multi-Agent System. This section comprehensively details the various programmatic strategies employed to enforce safety, security, and reliability around generative AI agents.

## Why Guardrails?

Large Language Models (LLMs) are exceptionally capable reasoning engines, but they are intrinsically non-deterministic. Without oversight, they are susceptible to:
- **Prompt Injections & Jailbreaks**
- **Hallucinations & Confabulations**
- **PII / Data Leakage**
- **Clinical/Domain Misalignment**
- **Missing Statutory Disclaimers**

To transition an agent from an experimental prototype into a production-grade enterprise system, it requires a robust defensive perimeter. We implement these perimeters utilizing LangGraph, enabling explicit state mapping and routing depending on validation outcomes.

## Architectural Patterns

This documentation covers five distinct guardrail methodologies, ranging from simple deterministic gateways to complex, multi-LLM semantic reviewers.

**1. [Input Validation](./01_input_validation.md)**
- **Type:** Deterministic (Pass/Fail)
- **Position:** Pre-Generation
- **Mechanism:** Protects the agent from malicious or irrelevant Prompts (PII, Injections). Halts execution instantly to save tokens.

**2. [Output Validation](./02_output_validation.md)**
- **Type:** Deterministic (Triage: Deliver/Auto-Fix/Block)
- **Position:** Post-Generation
- **Mechanism:** Intercepts LLM hallucinations or compliance violations. Introduces the "Auto-Fix" paradigm to gracefully repair minor errors rather than crashing.

**3. [Confidence Gating](./03_confidence_gating.md)**
- **Type:** Self-Assessment (Dynamic Threshold)
- **Position:** Inter-Generation
- **Mechanism:** Dynamically routes responses based on the LLM's own numerical confidence interval. Elegantly handles edge-case escalation to "Human-in-the-loop" review queues.

**4. [Layered Validation](./04_layered_validation.md)**
- **Type:** Macro-Topology 
- **Position:** End-to-End
- **Mechanism:** Employs "Defense in Depth" by aggressively staging Input checking, Agent Tool-Execution routines, and Output Verification into a unified pipeline.

**5. [LLM-as-Judge](./05_llm_as_judge.md)**
- **Type:** Model-Based (Semantic Review)
- **Position:** Post-Generation (as a subsequent LLM call)
- **Mechanism:** Solves the weakness of rigid Regex rules by deploying a secondary AI to critique the primary Agent's reasoning on qualitative matrices (Safety, Relevance, Completeness) via strict Pydantic parsing.

## How to Read This Documentation

Each pattern document (01–05) focuses specifically on the "What", "Why", "How", and "When". 
They include:
- A textual breakdown of the business logic.
- Low-Level Design (LLD) highlighting LangGraph-specific constructs (`State`, `Nodes`, `Conditional Edges`).
- A Mermaid Visual Execution Flow Diagram.
- Technical implementation insights detailing system trade-offs (e.g., token latency vs. evaluation accuracy). 

Choose the pattern that best matches the specific security tolerance and latency requirement of your production workload.

---

## Learning Path — Understanding the WHY Behind the Code

The documents above (01–05) describe *what* each wiring pattern looks like. The chapters below explain *why* the module is designed this way — the mental models, design tradeoffs, honest limitations, and production considerations that the pattern docs assume you already know.

**Recommended reading order:** Ch 1 → Ch 2 → Ch 3, then any of Ch 4–7 based on your immediate need.

| Chapter | File | What you learn |
|---------|------|----------------|
| Ch 1 — The Big Picture | [06_big_picture.md](./06_big_picture.md) | Airport security analogy, full ASCII architecture diagram, sub-component roles, design philosophy (mechanism vs. policy separation) |
| Ch 2 — Core Concepts | [07_core_concepts.md](./07_core_concepts.md) | 5 foundational concepts: guardrails vs. routing, fail-open vs. fail-closed, content vs. certainty, layered defence, the two-LLM pattern |
| Ch 3 — Deep Dive | [08_deep_dive.md](./08_deep_dive.md) | Every function and class: step-by-step logic, I/O types, implementation strategy, one honest limitation, production upgrade path |
| Ch 4 — Design Decisions | [09_design_decisions.md](./09_design_decisions.md) | 6 non-obvious decisions: check ordering, default confidence value, auto-append vs. block, configurable routing labels, fail-open for judge, structured output |
| Ch 5 — Gotchas & Anti-Patterns | [10_gotchas.md](./10_gotchas.md) | 3 inputs that fool the module, 4 integration mistakes with wrong/right examples, what is genuinely out of scope |
| Ch 6 — Putting It Together | [11_putting_it_together.md](./11_putting_it_together.md) | Annotated golden-path usage, bypass decision guide, component handoff diagram, scaling and async considerations |
| Ch 7 — Quick Reference | [12_quick_reference.md](./12_quick_reference.md) | Function/class table, copy-paste code snippets for every entry point, full ASCII decision flowchart |
