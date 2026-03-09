# Chapter 1 — The Big Picture

> **Learning chapter** — this document explains the *why* and *where* of the guardrails package.  
> For wiring patterns (the *how*), see chapters 01–05 in this directory.

---

## 1.1 What Problem Does This Module Solve?

### The Real-World Analogy — Airport Security

Picture an international airport. Passengers board from one side; planes depart from the other. In between, there are layered checkpoints:

1. **Check-in desk** — Is your ticket valid? Do you have the right documents? (Input validation)
2. **Security screening** — No prohibited items. Remove belt, laptop, liquids. (Content filtering)
3. **Gate agent** — Is your gate correct? Is the flight still on time? (Scope / confidence check)
4. **Customs at landing** — A second authority at the destination independently reviews your papers. (LLM-as-judge)

No single checkpoint is perfect. The design *assumes* each one will fail sometimes, which is exactly why there are four of them. If a passenger slips past check-in with a wrong ticket, the gate agent catches it. If they slip past the gate, customs catches it.

The guardrails package applies the same philosophy to an LLM agent:

- **Input guardrails** = check-in desk. Block bad queries before the LLM ever sees them.  
- **Output guardrails** = security screening. Intercept dangerous, incomplete, or non-compliant responses before they reach the user.  
- **Confidence guardrails** = gate agent. The LLM itself says "I'm not sure about this" — route that to human review.  
- **LLM-as-judge guardrails** = customs. A second LLM independently evaluates the first LLM's reasoning.

💡 **Mental model:** Guardrails are not error handlers for code bugs. They are *domain-safety checkpoints* for the semantic content flowing through the system.

---

## 1.2 Where Does This Module Live in the System?

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LangGraph Multi-Agent System                     │
│                                                                     │
│   User / External Caller                                            │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────────────────────────────────────────┐                  │
│   │          GUARDRAIL LAYER (this module)      │                  │
│   │                                             │                  │
│   │   [1] input_guardrails                      │                  │
│   │        • detect_pii()                       │                  │
│   │        • detect_prompt_injection()          │                  │
│   │        • check_medical_scope()              │                  │
│   │        └──> validate_input()  (aggregator) │                  │
│   │                    │                        │                  │
│   │            PASS    │    FAIL                │                  │
│   │                    │──────────────────────► REJECT (no LLM)   │
│   │                    ▼                        │                  │
│   └────────────────────┼────────────────────────┘                  │
│                        │                                            │
│         ▼              ▼                                            │
│   ┌─────────────────────────────────────────────┐                  │
│   │          LLM AGENT LAYER                    │                  │
│   │  (reasoning, tool calls, state updates)     │                  │
│   └─────────────────────────────────────────────┘                  │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────────────────────────────────────────┐                  │
│   │          GUARDRAIL LAYER (continued)        │                  │
│   │                                             │                  │
│   │   [2] output_guardrails                     │                  │
│   │        • check_prohibited_content()         │                  │
│   │        • check_safety_disclaimers()         │                  │
│   │        └──> validate_output()  (aggregator)│                  │
│   │                    │                        │                  │
│   │         ┌──────────┴──────────┐             │                  │
│   │         │          │          │             │                  │
│   │       PASS       FIX        BLOCK           │                  │
│   │         │          │          │             │                  │
│   │         │   auto-append   escalate          │                  │
│   │         │   disclaimer    + replace         │                  │
│   │         │                                   │                  │
│   │   [3] confidence_guardrails (optional)      │                  │
│   │        • extract_confidence()               │                  │
│   │        • gate_on_confidence()               │                  │
│   │                    │                        │                  │
│   │            HIGH    │    LOW                 │                  │
│   │                    │──────────────────────► ESCALATE          │
│   │                    ▼                        │                  │
│   │   [4] llm_judge_guardrails (optional)       │                  │
│   │        • evaluate_with_judge()              │                  │
│   │                    │                        │                  │
│   │         ┌──────────┴──────────┐             │                  │
│   │      APPROVE    REVISE     REJECT            │                  │
│   └─────────────────────────────────────────────┘                  │
│                        │                                            │
│                        ▼                                            │
│               Deliver to User                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**What comes before:** The raw user query or an agent-to-agent handoff message.  
**What comes after:** Either a delivered response, a blocked/escalated response, or a human-review queue entry.

The guardrail layer is *orthogonal* to the graph topology — it can be inserted at any LangGraph node boundary without restructuring the rest of the graph.

---

## 1.3 Sub-Components at a Glance

| File | What it is | What it protects against or enables |
|------|-----------|--------------------------------------|
| `input_guardrails.py` | Pre-LLM content validator | PII leakage, prompt injection attacks, off-topic queries, context overflow |
| `output_guardrails.py` | Post-LLM content validator | Hallucinated dangerous advice, missing disclaimers, incomplete responses |
| `confidence_guardrails.py` | Certainty-threshold router | Overconfident delivery of uncertain responses; enables human escalation |
| `llm_judge_guardrails.py` | Semantic evaluator (second LLM) | Reasoning gaps and nuance missed by keyword/regex checks |
| `__init__.py` | Public API aggregator | Single import surface; documents the full pipeline ordering |

Within each file, there is a further subdivision:

**`input_guardrails.py`**
- `validate_input()` — orchestrates all input checks in order
- `detect_pii()` — regex scan for SSN, phone, email, credit card, DOB
- `detect_prompt_injection()` — regex scan for jailbreak/override patterns
- `check_medical_scope()` — keyword matching to confirm medical domain

**`output_guardrails.py`**
- `validate_output()` — orchestrates all output checks; returns 3-state result
- `check_prohibited_content()` — regex scan for dangerous medical advice patterns
- `check_safety_disclaimers()` — checks for required medical disclaimer language
- `add_human_review_flag()` — prepends a HITL review flag to the output

**`confidence_guardrails.py`**
- `extract_confidence()` — parses "Confidence: 0.87" or "Confidence: 85%" from LLM text
- `gate_on_confidence()` — threshold comparison returning a routing label
- `check_confidence()` — extract + gate in one convenience call

**`llm_judge_guardrails.py`**
- `JudgeVerdict` — Pydantic model: safety + relevance + completeness + verdict
- `JUDGE_SYSTEM_PROMPT` — rubric that drives the judge LLM
- `evaluate_with_judge()` — invokes the second LLM with structured output
- `default_approve_verdict()` — safe fallback when the judge LLM fails

---

## 1.4 Design Philosophy

### Why is the guardrail layer separate from the LLM agent layer?

Three distinct reasons drive this separation:

**1. Timing.** Input guardrails must run *before* the LLM call to save tokens and prevent injection. Output guardrails must run *after* the LLM call to catch hallucinations. Confidence and judge guardrails run *after* the output check to evaluate quality. This creates a natural sequencing boundary that would collapse if merged into the agent node.

**2. Cost model.** Deterministic checks (regex, keyword matching) cost effectively zero — microseconds, no API calls. LLM judge checks cost tokens and add latency. Keeping them separate lets you choose which layers to activate per request, per deployment context (e.g., skip the judge for low-stakes queries; use full stack for high-stakes clinical decisions).

**3. Responsibility.** The agent layer handles *reasoning* — transforming inputs into outputs. The guardrail layer handles *policy* — enforcing what is safe to accept and deliver. Mixing them creates a component that is harder to test, harder to audit, and harder to update independently. When a new regulatory requirement arrives (e.g., "all outputs must contain an FDA disclaimer"), you change one file in `guardrails/`, not the agent.

⚠️ **Why this matters:** If guardrails were merged into the agent node, you would have no way to enforce the same policy across multiple agents in a multi-agent system. The separate layer enforces policy *uniformly*, regardless of how many agents are in the graph.

### What would break if the guardrails were merged with the agent?

- **No early rejection.** The LLM would process every query including injections and off-topic requests, consuming tokens and potentially leaking system prompt information.
- **No auto-fix.** The three-way output routing (deliver/fix/block) requires a layer *after* the LLM that can inspect and modify the response without the LLM re-running.
- **No composability.** You could not add or remove guardrail layers per agent without modifying each agent's code.
- **No independent testability.** Guardrail logic would be buried inside agent node functions, untestable in isolation.

### What principle drives the module's boundaries?

**Separation of mechanism from policy.** The LLM agent embodies mechanism (how to reason about clinical data). The guardrails embody policy (what is safe to accept, process, and deliver). These are fundamentally different concerns with different rates of change. Policy changes with regulations, deployment context, and risk tolerance. Mechanism changes with model capability and domain knowledge.
