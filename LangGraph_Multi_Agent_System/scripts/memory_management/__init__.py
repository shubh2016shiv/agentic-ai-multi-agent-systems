"""
Memory Management Pattern Scripts — Area 5 of 9
================================================
Root component module: memory/
These scripts demonstrate HOW to integrate the four memory tiers into
LangGraph graphs — the patterns, not the memory implementations.

WHERE THIS FITS IN THE LEARNING SEQUENCE
  Area 5 of 9 — study after HITL/ (Area 4).
  Memory enables agents to remember across turns and across sessions.
  Prerequisite: LangGraph state and checkpointers (scripts/handoff/).

SCRIPTS IN THIS PACKAGE — RECOMMENDED ORDER

  1. working_memory_scratchpad.py
     Shared key-value scratchpad flowing through LangGraph state.
     Use when: agents in a pipeline need to read each other's findings
     (upstream context). No LLM calls — pure memory mechanics.
     Root module: memory/working_memory.py → WorkingMemory (Tier 2)

  2. semantic_retrieval.py
     Retrieve relevant documents from a ChromaDB vector store (RAG).
     Use when: agents need to query a knowledge base at runtime
     (clinical guidelines, drug databases, policy documents).
     Root module: memory/long_term_memory.py → LongTermMemory (Tier 4)

  3. checkpoint_persistence.py
     Compile graphs with memory/sqlite/postgres/redis checkpointers.
     Use when: workflow state must survive restarts, or you need
     multi-turn conversations with the same thread_id.
     Root module: memory/checkpoint_helpers.py → build_checkpointed_graph(),
                  inspect_checkpoint()

  4. conversation_memory.py
     Multi-turn conversation with rolling summarisation.
     Use when: long conversations would exceed context window — summarise
     old messages while keeping recent ones intact.
     Root module: memory/conversation_memory.py → ConversationMemory (Tier 3)

  5. shared_memory_multi_agent.py
     All four memory tiers combined in one multi-agent workflow.
     Use when: you need scratchpad + RAG + persistence + summarisation
     working together. Study this last — it assumes the others.
     Root modules: memory/working_memory.py, memory/long_term_memory.py,
                   memory/checkpoint_helpers.py, memory/conversation_memory.py

ROOT MODULE CONNECTION
  memory/working_memory.py     — WorkingMemory (Tier 2: in-process scratchpad)
  memory/conversation_memory.py — ConversationMemory (Tier 3: rolling summarisation)
  memory/long_term_memory.py   — LongTermMemory (Tier 4: ChromaDB RAG)
  memory/checkpoint_helpers.py — build_checkpointed_graph(), inspect_checkpoint()
"""
