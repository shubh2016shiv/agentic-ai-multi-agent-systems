# Chapter 6 — Putting It All Together

> **Learning chapter** — The recommended golden path for integrating `hitl` with LangGraph schemas, deploying to production, and connecting with `guardrails`.

---

## 6.1 The Recommended Usage Pattern (The Golden Path)

The most robust way to implement HITL in a production MAS is to use the `review_nodes` factories and a strong `TypedDict` state.

### 1. Define the State
Your state must accommodate both the agent's output and the human's decision.
```python
class PatientCaseState(TypedDict):
    input_symptoms: str
    agent_prescription: str
    
    # HITL fields
    human_decision: str       # "approve", "reject", "edit"
    final_prescription: str   # The actual text delivered to the patient
```

### 2. Form the Nodes
Use the `hitl` package factories to generate a node that enforces separation of mutation and status.
```python
from hitl.review_nodes import create_edit_node

# Generate the node
physician_review = create_edit_node(
    state_key="agent_prescription",       # What the human reviews
    output_key="final_prescription",      # Where the human's text is saved
    action_key="human_decision",          # Where the action command is saved
    question="Review prescription. You may edit the dosage.",
)
```

### 3. Compile the Graph
Add the agent node, the generated review node, and a node to deliver the final text.
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(PatientCaseState)

builder.add_node("agent", clinical_agent_node)
builder.add_node("review", physician_review)
builder.add_node("delivery", deliver_to_patient_node)

builder.add_edge("agent", "review")

# Conditional routing based on the human's action
builder.add_conditional_edges(
    "review",
    lambda state: "delivery" if state["human_decision"] in ["approve", "edit"] else END
)

builder.add_edge("delivery", END)

# CRITICAL: HITL requires a checkpointer
memory = MemorySaver()
app = builder.compile(checkpointer=memory)
```

### 4. Execute the Graph
Because LangGraph executes asynchronously from the human's perspective, your backend application logic looks like this:

```python
thread = {"configurable": {"thread_id": "patient_123"}}

# Start the graph. It will pause when it hits the review node.
app.invoke({"input_symptoms": "Headache"}, thread)

# --- Human logs into UI hours later ---

# The UI sends a REST payload {"action": "edit", "content": "Take Advil"}
# Your backend wakes up the graph:
app.invoke(Command(resume={"action": "edit", "content": "Take Advil"}), thread)
```

---

## 6.2 How HITL connects to Guardrails

In an enterprise-grade medical MAS, `guardrails` and `hitl` work together in a pipeline.

1. **Input Guardrails** run first (blocks SQL injection, PII).
2. The Agent runs.
3. **Output Guardrails** run second (blocks toxicity, checks length).
4. **Confidence Guardrails** run third.
    - If the agent's confidence < 0.70, it sets a routing flag `"needs_review"`.
5. **HITL** is triggered conditionally by the Confidence Guardrails.

```python
# The routing function that connects Guardrails to HITL
def route_after_agent(state):
    # Did the guardrail component flag this?
    if state.get("confidence_needs_review", False):
        return "human_review_node"  # Route to HITL
    return "delivery_node"          # Skip HITL
```

---

## 6.3 Scaling & Performance Considerations

**The Checkpointer Database:** When a graph is paused via `interrupt()`, the state is serialized into the checkpointer. In production, this means `SqliteSaver` or `PostgresSaver`. If your `state` dictionary contains massive base64 images or gigabytes of context documents, LangGraph must write all of that to Postgres and then read it all back into RAM upon `resume()`. **Keep your state as lightweight as possible.**

**State Management for Review Queues:** In LangGraph, to build a "Review Queue" UI where doctors can see all pending interrupts, your backend must query the checkpointer database using the Search API.
```python
# Pseudo-code to find all paused threads
pending_threads = app.checkpointer.search(filter={"status": "interrupted"})
```
You build your UI table off this query. When the doctor clicks "Approve", your backend runs `app.invoke(Command(resume="approve"), config={"thread_id": thread["thread_id"]})`.
