# Chapter 3 — Deep Dive

> **Learning chapter** — a granular breakdown of the code that powers the HITL package.

---

## 3.1 `primitives.py`: The Standardization Layer

This file contains the foundational types and functions shared across every single HITL node. 

### `InterruptPayload` TypedDict
A `total=False` TypedDict defining the canonical shape of an interrupt payload.
```python
class InterruptPayload(TypedDict, total=False):
    question: str
    content: str
    options: dict | list
    # ... other fields
```

### Payload Builders
These functions take basic strings/dicts and shape them into an `InterruptPayload`.
- `build_approval_payload()`: For binary Yes/No decisions.
- `build_edit_payload()`: For Yes/No/Edit decisions.
- `build_tool_payload()`: For Yes/Skip tool executions.
- `build_escalation_payload()`: For tiered reviewer chains (e.g. Junior vs Senior).

**Why they exist:** To guarantee the UI receiving the interrupt payload never crashes due to a missing `"options"` key or a misspelled `"question"` string. 

### `parse_resume_action()`
The most critical utility function. It acts as an adapter flattening out chaotic user inputs.
```python
def parse_resume_action(resume_value: Any, default_action: str = "reject") -> dict:
```
- If the UI sent `True` → returns `{"action": "approve", ...}`
- If the UI sent `"escalate"` → returns `{"action": "escalate", ...}`
- If the UI sent `{"action": "edit", "content": "hello"}` → returns `{"action": "edit", "content": "hello", ...}`

This ensures that the downstream nodes can always reliably check `parsed["action"] == "approve"` without risking an `AttributeError` or `TypeError`.

---

## 3.2 `review_nodes.py`: The Factory Layer

This file contains functions that *generate LangGraph nodes*. 

Because an interrupt node requires doing the identical `build_payload -> interrupt -> parse_resume` boilerplate every time, we wrapped the boilerplate in a factory pattern.

### `create_approval_node()`
Creates a simple Approve/Reject node.

```python
def create_approval_node(
    state_key: str = "agent_response",
    question: str = "Do you approve this recommendation?",
    approve_key: str = "decision",
) -> Callable[[dict], dict]:
```
1. It looks up the text stored in `state[state_key]`.
2. It builds an `approval_payload`.
3. It `interrupt()`s the graph.
4. It parses the resume action into a boolean.
5. It returns `{approve_key: boolean_decision}` back to the graph state.

### `create_edit_node()`
Creates an Approve/Reject/Edit node.

```python
def create_edit_node(
    state_key: str = "agent_response",
    output_key: str = "final_output",
    status_key: str = "status",
) -> Callable[[dict], dict]:
```
1. Builds an `edit_payload`.
2. If the user edits the content, it intercepts the new string.
3. It updates `state[output_key]` with the *human's* text, completely destroying the *agent's* original text.
4. It updates `state[status_key]` to `"edited"` for downstream audit logging.

### `create_tool_confirmation_node()`
Creates a hook for approving high-stakes tool calls.

This relies on LangChain's standard tool call format. It reads the `proposed_tool_call` key, unpacks the `name` and `args` variables, and passes them to the human. The node returns `execute` or `skip`, which a router downstream will use to either fire the `ToolNode` or skip it.

---

## 3.3 `run_cycle.py`: The Test Harness Wrapper

If you want to manually test an interrupt node in a Jupyter Notebook or a generic script, you have to write this nightmare:

```python
config = {"configurable": {"thread_id": "1"}}
r = graph.invoke(state, config)
if "__interrupt__" in r:
    payload = r["__interrupt__"][0].value
    print(payload)
    command = Command(resume="approve")
    final = graph.invoke(command, config)
```

`run_cycle.py` wraps that exact boilerplate.

### `run_hitl_cycle()`
Automates a single-interrupt test. You give it the graph, the starting state, and the hardcoded resume action. It prints the interrupt payload sequentially and outputs the final state.

### `run_multi_interrupt_cycle()`
Automates a multi-step interrupt graph (like Escalation). You provide a `List` of resume values, and it loops the `invoke -> interrupt -> resume` dance until the graph naturally finishes.
