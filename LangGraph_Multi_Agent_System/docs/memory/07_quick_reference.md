# Chapter 7 — Quick Reference Card

> **Learning chapter** — Tables and snippets for the `memory` package.

---

## 7.1 Component Reference Table

### `WorkingMemory` (Tier 2)
| Name | Concept | Purpose |
|------|---------|---------|
| `get/set` | Standard Dict | Read/Write global workflow states. |
| `append_to` | Accumulator | Adding new diagnosis notes to an array seamlessly. |
| `get/set_scratch` | Agent Local | Private notepad for a single node. Hidden from others. |
| `to_context_string`| Serializer | Formats the memory cleanly so it can be injected into the LLM `SystemMessage`. |

### `ConversationMemory` (Tier 3)
| Name | Concept | Purpose |
|------|---------|---------|
| `maybe_summarise(msgs, old_summary)` | Rolling Window | Checks if length > `summarise_after`. If yes, tells the LLM to rewrite the history into a single paragraph. |
| `DEFAULT_SUMMARISE_PROMPT` | Instruction | The prompt telling the LLM to discard greetings and keep medical facts. |

### `LongTermMemory` (Tier 4)
| Name | Concept | Purpose |
|------|---------|---------|
| `add_documents(docs)` | Ingestion | Hashes strings, embeds via `.env` configured provider, writes to SQLite vector-store. |
| `search(query, k=3)` | Retrieval | Semantic similarity search returning an array of `{content: str, metadata: dict}`. |

### `checkpoints_helpers` 
| Name | Concept | Purpose |
|------|---------|---------|
| `build_checkpointed_graph()`| Factory | Allows instant swapping between `memory`, `sqlite`, and `postgres` checkpointers via a single argument. |
| `inspect_checkpoint(thread)` | Debugger | Spits out exactly what the graph database recorded on the last turn without re-triggering the LLM. |

---

## 7.2 Copy-Paste Cheat Sheet

### Swapping Checkpointers for Production
```python
from memory.checkpoint_helpers import build_checkpointed_graph

# Local dev (dies on reboot)
app = build_checkpointed_graph(workflow, checkpointer_type="memory")

# Local testing (survives reboot)
app = build_checkpointed_graph(workflow, checkpointer_type="sqlite", db_path="data.db")

# Production (survives nuclear bombs)
app = build_checkpointed_graph(workflow, checkpointer_type="postgres", connection_string="postgresql://...")
```

### Retrieving RAG Context 
```python
from memory.long_term_memory import LongTermMemory

db = LongTermMemory(collection_name="clinical_guidelines")

# Agent asked for hypertension protocol
results = db.search("Hypertension", k=2)

for r in results:
    print(r["content"]) # "The guideline dictates..."
```

### Auditing a frozen thread
```python
from memory.checkpoint_helpers import inspect_checkpoint

# Oh no, patient_001's thread got stuck. What did the LLM output?
history = inspect_checkpoint(app, thread_id="patient_001")
print(history["messages"][-1].content)
```
