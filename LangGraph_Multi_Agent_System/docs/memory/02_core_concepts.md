# Chapter 2 — Core Concepts

> **Learning chapter** — an explanation of the abstract mechanics underlying conversational persistence and semantic retrieval.

---

## 2.1 The Two Types of AI Memory

Humans have different systems for remembering what happened 10 minutes ago versus what they learned in medical school 10 years ago. AI Memory works exactly the same way.

### Episodic Memory (Conversation)
This is "what happened during this specific episode/session." 
- **Scope:** Transient. It belongs solely to User A and Session 1.
- **Mechanism:** List of LangChain `Message` objects stored in LangGraph state.
- **Problem it solves:** Answering follow-up questions ("What was the second medication you mentioned?").
- **Implementation:** `ConversationMemory` + `MemorySaver`.

### Semantic Memory (Knowledge Base)
This is "facts about the world that are always true."
- **Scope:** Permanent. It applies to all users and all sessions.
- **Mechanism:** Text documents converted to floats (Embeddings) and stored in a vector database.
- **Problem it solves:** Hallucinations. Grounding responses in truth ("According to the 2024 Asthma Guidelines...").
- **Implementation:** `LongTermMemory` + `ChromaDB`.

---

## 2.2 Re-hydrating State (The Checkpointer)

How does LangGraph actually remember a conversation if the Python script exits after returning an HTTP response?

**It doesn't. The Checkpointer does.**

Every time `app.invoke()` finishes a node, LangGraph takes the entire `State` dictionary and serialize-dumps it into a database (SQLite, Postgres). 

In order to resume a conversation, your frontend application MUST send back a `thread_id` associated with that user's session:
```python
# The server wakes up and looks for thread "user_123"
config = {"configurable": {"thread_id": "user_123"}}

# LangGraph queries SQLite, rebuilds the state dictionary into RAM, 
# and processes the user's new message.
app.invoke({"messages": [new_msg]}, config)
```
If you forget the `thread_id` or send a new one, LangGraph creates a brand new blank state. **The Thread ID is the anchor for episodic memory.**

---

## 2.3 Rolling Summarization Mechanics

As a conversation hits 10, 20, or 50 turns, the array of `Message` objects in the LangGraph state grows linearly. 

If your LLM has an 8,000 token limit, passing a 9,000 token list of messages will result in a hard crash. Even if you have a 128,000 token limit, passing 20,000 tokens of "Hello" / "Please wait a moment" / "Okay" pleasantries is incredibly expensive and degrades the LLM's reasoning capability.

`ConversationMemory` solves this by introducing a **sliding window**.

```text
Turn 1-6: [M1, M2, M3, M4, M5, M6]  <-- Normal History

Turn 7: Window limit Reached!

  1. We split the history:
     Old Window: [M1, M2, M3, M4]
     New Window: [M5, M6, M7]

  2. We pass the Old Window to the LLM and ask: "Summarize this."
     Result: "Patient has a history of X and we discussed Y."

  3. We rewrite the graph state:
     [SystemMessage("Summary: Patient has X Discussed Y"), M5, M6, M7]
```
The token count drops dramatically, but the critical medical context survives.

---

## 2.4 Retrieval-Augmented Generation (RAG)

When you ask an LLM, "What is the dosage for Drug X?", it checks its pre-trained weights. If it doesn't know, it guesses (hallucinates).

RAG intercepts the query *before* it reaches the LLM. 
1. **The Vector Store**: You ingest 1,000 PDF pages of medical guidelines into ChromaDB.
2. **The Embedding**: The text is converted into numbers (vectors) representing semantic meaning.
3. **The Search**: The user asks "Drug X dosage". The database finds the 3 paragraphs mathematically closest to that query.
4. **The Augmentation**: You silently inject those 3 paragraphs into the LLM's prompt:
   - *"Answer the user's query using ONLY the following context: [Paragraph 1, 2, 3]."*

This is what `LongTermMemory` executes behind the scenes. It creates a firewall against hallucination.
