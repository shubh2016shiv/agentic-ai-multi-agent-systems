# Chapter 7 — Quick Reference Card

> **Learning chapter** — Tables and definitions for the `orchestration` package.

---

## 7.1 Data Contracts

| Model | Purpose | Example Fields |
|-------|---------|----------------|
| `OrchestrationResult` | Rigid envelope wrapping an LLM's response. Required so downstream agents can parse outputs. | `agent_name`, `specialty`, `output`, `was_successful` |
| `PatientWorkload` | The input structure passed to every graph. Contains the medical data + routing instructions. | `patient_case`, `required_specialties`, `urgency_level` |
| `SHARED_PATIENT` | A hardcoded test patient (68M COPD/Heart Failure) used uniformly across all 5 architectures to allow isolated performance comparison. | |

---

## 7.2 BaseOrchestrator Methods

| Method | Role | Resilience Integration |
|--------|------|------------------------|
| `invoke_specialist()` | The workhorse. Formats prompts and triggers the LLM. | Fully wrapped. Catches `CircuitBreakerOpen` and returns an `OrchestrationResult` with `was_successful=False` (Graceful Degradation). |
| `invoke_synthesizer()` | The compiler. Combines successful agent outputs into a final report. | Fully wrapped. Re-raises exceptions as a hard `RuntimeError` (Catastrophic Failure). |

---

## 7.3 The 5 Architectures (Scripts)

*Found in `scripts/orchestration/`*

1. **Supervisor:** One LLM coordinates parallel workers.
2. **Peer-to-Peer:** Sequential, decentralized, passing the baton.
3. **Dynamic Router:** An LLM reads the payload and performs conditional subset routing.
4. **Subgraphs:** Hierarchical nested graphs (like hospital departments).
5. **Hybrid:** A combination approach (e.g., parallel deployment of sequential teams).
