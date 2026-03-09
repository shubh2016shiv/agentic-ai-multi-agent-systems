# Chapter 4 — Design Decisions

> **Learning chapter** — An examination of the non-obvious choices made when designing the `orchestrator.py` package, and the trade-offs involved.

---

## 4.1 Why embed Resilience at the BaseOrchestrator root?

*The typical LangGraph approach:*
When a developer writes a node in `peer_to_peer_orchestration/agents.py`, they usually put the LLM retry logic inside that exact node function using a library like `tenacity`.
```python
@retry(stop=stop_after_attempt(3))
def cardiology_node(state):
    llm.invoke(prompt)
```

*Our Design Decision:* **We banned resilience logic from the graph nodes.**

1. **DRY Principle (Don't Repeat Yourself):** If you use the typical approach, you have to write `@retry(stop=stop_after_attempt(3))` in literally every single node across all 5 architectures. That's 50+ identical decorator lines.
2. **Global State Awareness:** If Pulmonology fails 5 times and the server crashes, Cardiology in a different sub-graph shouldn't attempt the same failed server. By lifting the CircuitBreaker OUT of the subgraphs and putting it inside `BaseOrchestrator.invoke_specialist`, we create a single, unified global pool of resilience. If one agent trips the breaker, all agents instantly benefit from the fast-failure. 
3. **Focus on Orchestration:** When students read the 5 pattern scripts, they don't see ugly retry blocks cluttering up the code. They only see pure, raw LangGraph routing logic.

---

## 4.2 Why centralize `SPECIALIST_SYSTEM_PROMPTS`?

In `orchestrator.py`, we hardcode the system prompts for Nephrology, Cardiology, and Pulmonology.

*The alternative:* Define the prompt inside the `nephrology_node` in the `scripts/orchestration/supervisor/agents.py` file.

*Our Design Decision:* **Enforce uniform testing variables.**

The entire point of the `scripts/orchestration/` folder is to teach you the difference between Supervisor routing vs Peer-to-Peer routing vs Subgraph routing.

If the Supervisor's Nephrologist had a completely different system prompt than the Peer-to-Peer's Nephrologist, you wouldn't know if the output change was caused by the *Graph Architecture* or simply because the prompt string changed. By centralizing the string in the root class, we guarantee that all 5 architectures are fighting on a perfectly level playing field.

---

## 4.3 Why skip the Bulkhead Pattern in `_ORCHESTRATION_CALLER`?

```python
response = _ORCHESTRATION_CALLER.call(
    llm.invoke,
    prompt,
    config=config,
    skip_rate_limiter=False,  
    skip_bulkhead=True,  # WHY IS THIS TRUE?
)
```

A **Bulkhead** isolates concurrent workloads (e.g., if you have 100 API slots, you give 50 to Agent A and 50 to Agent B so Agent A can't monopolize the queue). 

*Our Design Decision:* **LangGraph edges implicitly handle concurrency limits.**

Most of these orchestration patterns are linear or run a finite, knowable number of parallel branches (e.g., exactly 3 specialists in parallel). The bulkheading layer is massive overkill, adding memory usage and locking latency without providing any safety that LangGraph's internal thread limits don't already handle.

---

## 4.4 Why does the Synthesizer raise an error instead of returning `False`?

When `invoke_specialist()` fails, it returns `was_successful=False`.
When `invoke_synthesizer()` fails, it raises `RuntimeError`.

*Our Design Decision:* **Graceful Degradation vs Catastrophic Failure.**

If Pulmonology fails, it's unfortunate, but Cardiology and Nephrology probably succeeded. The Synthesizer can still read the two successful envelopes and generate a 66% accurate medical report. That is **Graceful Degradation**, which is critical in healthcare and enterprise scale tools.

If the Synthesizer fails... the entire graph returns nothing. The user clicked "Submit" and got a blank screen. There is no fallback. The application must crash hard (raise) so the upstream API server knows to return an `HTTP 500 Internal Server Error` to the React frontend.
