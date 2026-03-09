# Chapter 4 — Design Decisions

> **Learning chapter** — an examination of the non-obvious choices made when designing the `hitl` package, and the trade-offs involved.

---

## 4.1 Why standardise the Resume Action?

*The problem:* An approval node requires a boolean. An edit node requires a string. A tool confirmation requires a boolean. A tiered escalation requires a string ("escalate" vs "reject").

*The native LangGraph approach:* Just let the human send whatever arbitrary Python object they want in the `Command(resume=...)` call and let the review node figure it out.

*Our Design Decision:* **Force everything through `parse_resume_action()`**

The UI developers shouldn't have to look up the source code for each LangGraph node to figure out what data type to send. By standardizing on a single dictionary shape (`{"action": str, "content": str, "reason": str}`), we decouple the backend from the frontend. 

The backend nodes treat everything as a dictionary. 
- Boolean `True` becomes `{"action": "approve"}`.
- String `"escalate"` becomes `{"action": "escalate"}`.

This makes conditional routing extremely robust. Every downstream edge can just check `state["human_decision"]["action"] == "approve"`.

---

## 4.2 Why use Factories instead of Inline Nodes?

*The problem:* If you look at the scripts in `scripts/HITL/`, every pattern uses inline nodes:
```python
def review(state):
    interrupt(...)
    # parsing logic
workflow.add_node("review", review)
```

*Our Design Decision:* **Use Factories (`create_approval_node()`) in production, but Inline Nodes in scripts.**

In a single educational script, inline nodes are great because you can read top-to-bottom and see exactly what `interrupt()` is doing. 

In a production clinical application with 15 different agents, writing out the `build_payload -> interrupt -> parse_resume` boilerplate 15 times is a recipe for a copy-paste error. What happens if 1 of the 15 nodes forgets to parse the boolean out of the dict? The whole graph crashes.

The factories in `hitl/review_nodes.py` guarantee that every single review node parses data identically.

---

## 4.3 Why not use LangGraph's Webhook / Polling features directly?

*The problem:* When `interrupt()` fires, how does the frontend UI know that the graph is paused? Does LangGraph broadcast a WebSocket message?

*The native LangGraph approach:* No, LangGraph is fundamentally a backend library. It simply suspends the thread. 

*Our Design Decision:* **Leave the notification layer entirely out of this package.**

The `hitl` module provides the LangGraph primitives, but it intentionally does *not* include logic for WebSockets, polling, or email notifications. 

Why? Because the notification architecture is completely dependent on your deployment environment (FastAPI, LangServe, AWS Lambda). 

If you are using LangGraph Studio or LangGraph Cloud, its HTTP API natively handles Webhooks when `__interrupt__` fires. If you are building a custom FastAPI app, your application layer should check the `result` dict for `__interrupt__` and push a WebSocket event to your React frontend. Putting that logic inside the `hitl` package would couple it to a specific web framework.

---

## 4.4 Why store the "action" and the "output" in different state keys?

*The problem:* When a human edits an agent's response, we need to save the new text and we need to log that it was edited.

*Our Design Decision:* **Separation of mutation and status.**

In `create_edit_node()`:
```python
return {
    "action_key": "edit",            # Passed down the conditional edge
    "output_key": edited_text,       # Replaces the agent's response
    "status_key": "edited",          # Saved forever in the State DB
}
```

We do not overwrite the agent's original response text. We create a completely new key in the state (`final_output`). 
By keeping the agent's draft and the human's edit in separate state variables, we preserve a perfect audit trail. You can query the Checkpointer DB later and see exactly what the LLM hallucinated versus what the human actually delivered safely to the patient.
