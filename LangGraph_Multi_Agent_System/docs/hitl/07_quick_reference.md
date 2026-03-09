# Chapter 7 — Quick Reference Card

> **Learning chapter** — Tables, snippets, and flowcharts for the `hitl` package.

---

## 7.1 Function / Class Reference Table

### `hitl.primitives`
| Name | Type | Input | Output | Purpose |
|------|------|-------|--------|---------|
| `InterruptPayload` | TypedDict | — | — | Standardized schema for interrupt payloads (question, content, options). |
| `ResumeAction` | Literal | — | — | "approve", "reject", "edit", "escalate", "execute", "skip" |
| `parse_resume_action` | function | `resume_value`, `default` | `dict` | Normalizes booleans, strings, and dicts into `{"action": "...", "content": "..."}`. |
| `build_{x}_payload` | function | `response`, `question` | `InterruptPayload` | Formats a raw AI response into a UI-ready structure. |

### `hitl.review_nodes`
| Name | Type | Key Capabilities | Purpose |
|------|------|------------------|---------|
| `create_approval_node` | factory func | Parses bools, sets `approve_key` | Quick Yes/No approvals. |
| `create_edit_node` | factory func | Parses dicts, sets `output_key` and `status_key` | Approvals with human-injected text rewrites. |
| `create_tool_confirmation_node`| factory func | Reads `proposed_tool_call` from state | Preventing agents from executing destructive tools. |
| `create_escalation_node` | factory func | Accepts `reviewer_role` flag | Tiered pipelines (Junior reviews, then Senior reviews). |

### `hitl.run_cycle` (For Testing Only)
| Name | Type | Input | Output | Purpose |
|------|------|-------|--------|---------|
| `run_hitl_cycle` | function | `graph`, `thread`, `state`, `resume_val` | `dict` | Invokes the graph, auto-resumes the paused thread with `resume_val`, and returns the final state. |

---

## 7.2 Copy-Paste Cheat Sheet

### Basic Approve/Reject Node
```python
from hitl.review_nodes import create_approval_node

# Adds an `is_approved: bool` key to the state dictionary
node = create_approval_node(
    state_key="agent_prescription",
    question="Approve this prescription?",
    approve_key="is_approved"
)
workflow.add_node("review", node)
```

### Basic Edit Node
```python
from hitl.review_nodes import create_edit_node

# Destroys nothing. Saves human edit to `human_prescription` 
# and saves action to `human_action` (edit/approve/reject).
node = create_edit_node(
    state_key="agent_prescription",
    output_key="human_prescription",
    action_key="human_action"  
)
workflow.add_node("review", node)
```

### Routing Logic
```python
# The conditional edge function
def route_after_review(state):
    action = state.get("human_action")
    if action in ["approve", "edit"]:
        return "send_to_patient"
    else: # "reject" or default case
        return "abort"

workflow.add_conditional_edges("review", route_after_review)
```

### Waking up a paused graph (Application Backend)
```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "patient_001"}}
graph.invoke(Command(resume={"action": "edit", "content": "Advil 200mg"}), config)
```

---

## 7.3 Graph Lifecycle Flowchart

```text
 ┌──────────────┐         ┌───────────────┐
 │ app.invoke() ├────────►│  Agent Node   │───┐
 └──────────────┘         └───────────────┘   │
                                              ▼
 ┌──────────────┐         ┌───────────────────────────────┐
 │ Human Viewer │         │       Review Node             │
 │              │         │ 1. state = get_state()        │
 │ [Approve UI] │◄────────┤ 2. interrupt(state)           │
 │ [Edit UI]    │ (PAUSED)│ 3. ─── LangGraph Halts ────   │
 └──────────────┘         └───────────────┬───────────────┘
        │                                 │
        ▼                                 │
  Command(resume=...)                     │
        │                                 ▼
 ┌──────┴───────┐         ┌───────────────────────────────┐
 │ app.invoke() │────────►│       Review Node             │
 └──────────────┘         │ 1. state = get_state()        │
                          │ 2. decision = interrupt(...)  │
                          │ 3. return {"action": decision}│
                          └───────────────┬───────────────┘
                                          │
                                          ▼
                               ┌─────────────────┐
                               │ Delivery Node   │
                               └─────────────────┘
```
