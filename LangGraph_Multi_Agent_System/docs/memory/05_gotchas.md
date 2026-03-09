# Chapter 5 — Gotchas, Edge Cases & Anti-Patterns

> **Learning chapter** — The traps and pitfalls of implementing Memory architectures in LangGraph.

---

## Gotcha 1: Confusing VectorDB Persistence with Checkpointer Persistence

**The Scenario:** You install ChromaDB, index your medical documents, and ask the agent a question. Then you restart the Python script and ask a follow-up ("What was the first drug you told me about?"). The agent has no idea.
**The Failure:** "But I installed long-term memory!"
**The Fix:** ChromaDB (Long-Term Memory) only stores *documents*. It does NOT store the *conversation history*. To make the agent remember yesterday's chat, you must mount a `MemorySaver`/`SqliteSaver` in your graph compilation.  

## Gotcha 2: Returning the wrong structure from the Summarization Node

**The Scenario:** When `should_summarise()` triggers, your LangGraph node returns the newly compressed system prompt. 
```python
def compress_node(state):
    msgs, summary = cm.maybe_summarise(state["messages"])
    return {"messages": msgs}
```
**The Failure:** LangGraph's default reducer for messages is *append*. If you return the 3 compressed messages, LangGraph appends them to the end of the existing 10 messages. Now your state has 13 messages, totally defeating the compression.
**The Fix:** You must pass the LangChain `RemoveMessage` objects to explicitly command LangGraph to delete the old messages from the Checkpoint database. 

## Gotcha 3: Forgetting the `thread_id` 

**The Scenario:** You build a patient-check-in bot. User A starts chatting. When User B opens the app on a different device, they immediately see User A's un-summarized conversation history. 
**The Failure:** You hardcoded `config = {"configurable": {"thread_id": "1"}}` in your backend server. All users across the country are reading from the exact same Checkpointer row in SQLite.
**The Fix:** The `thread_id` must be dynamically tied to the User's authentication session (e.g., their JWT token or UUID). 

## Gotcha 4: Adding `WorkingMemory` to the TypedDict

**The Scenario:** You love the `WorkingMemory` class so much that you try to put it directly into the LangGraph state.
```python
class State(TypedDict):
    memory: WorkingMemory
```
**The Failure:** When LangGraph attempts to serialize the `State` to the Checkpointer (Sqlite), it uses `pickle`/JSON. It will completely choke on the custom Python object, crashing the application.
**The Fix:** `WorkingMemory` should wrap the state *inside* the node, or you should restrict the `TypedDict` exclusively to primitive types (`str`, `int`, `dict`, `list`). 

## Anti-Pattern 1: Summarizing every single turn

**The Scenario:** Because LLM context windows are expensive, you decide to summarize the conversation after every single human message.
**The Reality:** The conversation turns into a game of Telephone. 
Turn 1: "Take Tylenol." -> Summary: "Recommended Tylenol."
Turn 2: "What dosage?" -> Summary: "Patient asked for dosage of recommended painkiller." 
By Turn 3, the exact scientific name of the drug has been completely hallucinated away by the summarizer LLM because of repeated compression passes.
**The Fix:** Use a rolling window. Let the LLM keep the last 4 exact, unedited messages in its context window (the exact strings) so it can pull verbatim medical dosages out of the immediate history. Only compress messages that have scrolled completely out of view.
