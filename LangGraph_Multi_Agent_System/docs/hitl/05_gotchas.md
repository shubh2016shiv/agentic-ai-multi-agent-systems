# Chapter 5 — Gotchas, Edge Cases & Anti-Patterns

> **Learning chapter** — The traps and pitfalls of implementing Human-in-the-Loop workflows in LangGraph.

---

## Gotcha 1: The "Double Execution" Bug
**The Scenario:** You place an `llm.invoke()` call and an `interrupt()` call in the exact same LangGraph node.
**The Failure:** When the human resumes the graph, the node restarts from line 1. The `llm.invoke()` runs a second time.
**The Fix:** Never put side effects in the same node as an `interrupt`. Make your review nodes 100% idempotent. They should only read from the state dictionary, format the payload, and interrupt.

## Gotcha 2: Missing the Checkpointer
**The Scenario:** You build a complex HITL workflow, but when you run `workflow.invoke()` it throws a confusing error about `super_step`.
**The Failure:** LangGraph relies on the checkpointer to save the state when `interrupt()` fires. You forgot to pass it during compilation.
**The Fix:** Always compile your graph with a checkpointer: `app = workflow.compile(checkpointer=MemorySaver())`.

## Gotcha 3: The Thread ID Collision
**The Scenario:** Human A pauses a graph to review a case. Human B resumes the graph with their decision.
**The Failure:** Human B accidentally passed Human A's `thread_id` into their `Command(resume=...)` config. LangGraph applies B's decision to A's patient case.
**The Fix:** Your API backend must have strict session management ensuring that the `thread_id` associated with a specific interrupt payload cannot be spoofed or accessed by unauthorized users. 

## Gotcha 4: Forgetting the `parse_resume_action()` Default
**The Scenario:** You build a custom inline node, call `interrupt()`, and assume the resume value is always a boolean `True`.
**The Failure:** A rogue API caller or a bug in your frontend UI sends `"yes"` instead of `True`. Your node crashes with a `TypeError` because it tries to do `if decision is True`.
**The Fix:** Always run human inputs through `parse_resume_action(resume, default_action="reject")`. It guarantees that if garbage data is sent, the system fails closed (rejects the action).

## Gotcha 5: Misunderstanding `Command(resume=...)` vs `State` Updates
**The Scenario:** A human reviews the agent's output and wants to append a new document to the state dictionary before the graph resumes. They attempt to do this by passing a state dictionary into the resume command: `Command(resume={"messages": ["New Doc"]})`.
**The Failure:** `Command(resume=...)` does **not** update the state. It *only* passes data back to the `interrupt()` function's return variable.
**The Fix:** The node that called `interrupt()` must receive the resume value, parse it, and then explicitly return it as a state update:
```python
def review_node(state):
    human_doc = interrupt("Please provide a doc")
    # This actually updates the state
    return {"messages": [human_doc]}
```

## Anti-Pattern 1: The Infinite Interrupt Loop
**The Scenario:** A developer puts an `interrupt()` inside a node that is part of a cyclic graph (e.g. a tool-calling `while` loop), hoping to get human approval for every tool.
**The Reality:** The human approves Tool 1. The loop restarts. The node hits the interrupt again. The graph pauses. The UI is locked until the human approves Tool 2. 
**The Fix:** Be incredibly careful putting `interrupt()` inside cyclic logic. Usually, it is better to have the agent plan *all* required tool calls, pause once, and have the human approve the entire batch using `Pattern B (tool_call_confirmation)`.
