# Chapter 3 — Deep Dive

> **Learning chapter** — a granular breakdown of the code that powers the Memory package.

---

## 3.1 `working_memory.py`: The `WorkingMemory` Class

This class is a wrapper around a standard Python dictionary. It is instantiated inside the LangGraph workflow process and is wiped clean as soon as the workflow ends.

### Key Methods
- `append_to(key, value)`: Creates a list if it doesn't exist, and appends to it. Instead of writing 4 lines of boilerplate in every LangGraph node, agents can seamlessly build up `memory.append_to("findings", "Lesion on left arm")`.
- `set_scratch(agent, key, value)` & `get_scratch(agent)`: Provides a private partition within the memory. If Agent A has temporary reasoning notes that would confuse Agent B, they get written to the localized scratchpad.
- `to_context_string(max_length)`: Iterates through the entire dictionary and formats it into a human-readable string. This is invaluable because you can dynamically inject the entire state of the working memory straight into a downstream agent's System Prompt.

---

## 3.2 `conversation_memory.py`: The `ConversationMemory` Class

This stateless utility object provides the rolling summarization logic required for long-lived chat sessions.

### `should_summarise(messages: list) -> bool`
Checks if the length of the message array exceeds the configured `summarise_after` limit (default 6). Note that we count *messages*, not tokens, to prioritize simplicity and determinism over exact maximum-token density.

### `window_messages(messages: list) -> tuple`
Splits the history array into an `old_messages` array (to be summarized and deleted) and a `recent_messages` array (to be preserved exactly as the user typed them).

### `summarise_history(messages, old_summary, llm) -> tuple[list, str]`
The heavy lifter:
1. It looks at any existing `old_summary`.
2. It formats the `old_messages` into a transcript.
3. It prompts the `llm` to merge the old summary and the transcript into a new paragraph.
4. It prepends the new summary as a `SystemMessage` in front of the `recent_messages` array.
5. It returns the newly compressed message list back to the LangGraph state.

---

## 3.3 `long_term_memory.py`: The `LongTermMemory` Class

A wrapper over `langchain-chroma`, providing a seamless semantic search layer. 

### Why wrap ChromaDB in our own class?
ChromaDB exposes a complex API. We wrap it to hide:
1. The injection of the embedding model configured in `.env` (`get_embeddings()`).
2. The automatic generation of document IDs via MD5 hashing.
3. The translation of similarity score (0-1) into distance.

### `add_documents(documents, metadatas=None, ids=None)`
Takes an array of raw strings, hashes them to create unique IDs (preventing duplication), embeds them via the LLM provider API, and writes them to the persistent local `.sqlite3` file inside `chroma_db/`.

### `search(query, k=3, where=None)`
Performs an embedding search on the `query` string, returning the Top-K most semantically identical documents. Crucially, it returns an array of dicts formatted perfectly for LLM context injection rather than exposing raw ChromaDB artifacts.

---

## 3.4 `checkpoint_helpers.py`: The DB Wiring

LangGraph requires a Checkpointer (like Sqlite) to save history. Passing the checkpointer correctly varies between environments.

### `build_checkpointed_graph(workflow, type, **kwargs)`
A massive quality-of-life wrapper. In development, you don't want to spin up Postgres. You just want in-memory fast testing. In production, you *need* Postgres. 

This helper function centralizes the initialization logic.
```python
# Magic! Defaults to MemorySaver
graph = build_checkpointed_graph(workflow)

# Swap to Sqlite silently under the hood
graph = build_checkpointed_graph(workflow, checkpointer_type="sqlite")
```

### `inspect_checkpoint(graph, thread_id)`
A critical debugging tool. It reads the database directly and prints out the exact variables stored in the graph's `State` for that user, without re-triggering the LLM. Extremely useful for identifying where a multi-agent system stalled.
