# Chapter 2 — Core Concepts

> **Learning chapter** — this document explains the abstract theories and system mechanics that make LangGraph's HITL feature work under the hood. 

---

## 2.1 State Persistence and the Checkpointer

The single most important concept in Human-in-the-Loop workflows is **memory persistence**.

When an agent needs human review, the human is not sitting there waiting for it to finish. The human might take hours to review the payload. This means the Python process running the graph *must exit*.

Without a Checkpointer, LangGraph lives entirely in RAM. Calling `interrupt()` without a Checkpointer will simply throw an exception. 

### Why the Checkpointer matters
To use HITL, you must mount a checkpointer (like `MemorySaver` or `SqliteSaver`) during compilation:
```python
workflow.compile(checkpointer=MemorySaver())
```

When a node calls `interrupt(payload)`:
1. LangGraph takes everything in the current `State` (messages, variables, lists) and runs it through Pydantic/Pickle.
2. It writes that binary blob to the Checkpointer's database.
3. It throws an `__interrupt__` signal back to the main thread.
4. The graph execution officially dies. All RAM is freed.

When you call `graph.invoke(Command(resume=...))`:
1. LangGraph queries the Checkpointer for the saved blob.
2. It hydrates the `State` back into RAM.
3. It **re-enters the exact node** that called `interrupt()`.

---

## 2.2 Threads and Thread IDs

How does the Checkpointer know *which* frozen graph to wake up? **Thread IDs.**

In LangGraph, every single execution must be uniquely identified if you want to checkpoint it. 

If User A and User B both ask the clinical agent a question exactly same time, they both spawn a new thread. If User A's thread hits an `interrupt()`, the system must be sure that User A's resume command doesn't accidentally resume User B's thread.

You enforce this via the `config` dictionary:

```python
# Starting the graph
config = {"configurable": {"thread_id": "user_a_session_123"}}
graph.invoke(initial_state, config=config)

# ... Hours later ...

# Waking up the exact same graph
config = {"configurable": {"thread_id": "user_a_session_123"}}
graph.invoke(Command(resume={"action": "approve"}), config=config)
```

If the `thread_id` does not match, LangGraph will either throw an error (saying this thread is not paused) or it will start a brand new execution entirely.

---

## 2.3 The Idempotency Rule of Interrupted Nodes

As mentioned in the Big Picture: **When a paused graph is resumed, the node that called `interrupt()` restarts from line 1.**

Consider this node:
```python
def dangerous_node(state):
    # This happens BEFORE the pause
    charge_customer_credit_card(100)
    
    # Pause and wait for human to approve the receipt
    interrupt("Confirm receipt sent?")
    
    # This happens AFTER the pause
    send_receipt_email()
```

When `invoke()` is run the first time:
1. The customer is charged $100.
2. The graph pauses.

When `Command(resume=...)` is run the second time:
1. The node restarts at line 1.
2. **The customer is charged $100 AGAIN.**
3. The `interrupt()` function sees the `Command` and immediately returns the resume payload.
4. The receipt email is sent.

### The Fix: Isolate the Interrupt
Never put side effects in the same node as an `interrupt()`. 

```python
# Node 1
def charge_node(state):
    charge_customer_credit_card(100)
    return {}

# Node 2
def review_node(state):
    # This node ONLY reads state and interrupts. 100% idempotent.
    decision = interrupt("Confirm receipt?")
    return {"decision": decision}

# Node 3
def email_node(state):
    send_receipt_email()
    return {}
```

---

## 2.4 Resume Values and Standardized Payloads

The `interrupt()` function takes exactly one argument: the payload. 
The `Command(resume=...)` object takes exactly one argument: the resume value.

### The Payload (Node → Human)
This maps to what the human *sees*. If we don't standardize this, our frontend UI team will hate us. 
That's why `primitives.py` defines `InterruptPayload`:
```python
class InterruptPayload(TypedDict, total=False):
    question: str
    content: str
    options: dict | list
```
Instead of manually typing strings into `interrupt()`, you use a builder:
```python
payload = build_approval_payload(response="Take 25mg Aspirin")
interrupt(payload)
```

### The Resume Value (Human → Node)
This maps to what the human *did*. 
If Pattern A expects a boolean (`True`), but Pattern B expects a string (`"approve"`), the backend routing logic becomes a confusing mess of type checking.

Instead, ALL resume values are instantly piped through a parser:
```python
# The UI sent a boolean True:
raw_resume = interrupt(...) 

# We parse it into a standardized schema
parsed = parse_resume_action(raw_resume)
# Returns {"action": "approve", "content": "", "reason": ""}
```
This forces all human interactions down into a predictible dictionary that the rest of the LangGraph node can easily route against.
