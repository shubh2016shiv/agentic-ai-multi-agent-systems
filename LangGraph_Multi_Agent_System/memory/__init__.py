"""
Memory Package
================
Multi-tier memory system for multi-agent pipelines. Each tier serves a
different temporal scope and storage backend.

Where This Fits in the MAS Architecture
-----------------------------------------
Memory is the persistence layer of an agent system. Without memory,
each agent call starts from zero — the agent has no context about the
current session, previous conversations, or domain knowledge.

The four-tier model (from MAS reference architecture):

    ┌─────────────────────────────────────────────────────────┐
    │ Tier 1: LangGraph State (TypedDict)                     │
    │   Scope: ONE node execution → next node                 │
    │   Storage: In-memory Python dict                        │
    │   LangGraph feature: state passed to/from nodes         │
    └───────────────────────────┬─────────────────────────────┘
                                │
    ┌───────────────────────────▼─────────────────────────────┐
    │ Tier 2: Working Memory (working_memory.py)              │
    │   Scope: ONE workflow execution                         │
    │   Storage: WorkingMemory dict (wraps Tier 1)            │
    │   API: get(), set(), append_to(), to_context_string()   │
    └───────────────────────────┬─────────────────────────────┘
                                │
    ┌───────────────────────────▼─────────────────────────────┐
    │ Tier 3: Conversation Memory (conversation_memory.py)    │
    │   Scope: ONE session (multiple turns)                   │
    │   Storage: LangGraph MemorySaver + add_messages         │
    │   Feature: rolling summarisation for context window     │
    └───────────────────────────┬─────────────────────────────┘
                                │
    ┌───────────────────────────▼─────────────────────────────┐
    │ Tier 4: Long-Term Memory (long_term_memory.py)          │
    │   Scope: PERMANENT (survives restarts)                  │
    │   Storage: ChromaDB vector store                        │
    │   Feature: semantic RAG retrieval                       │
    └─────────────────────────────────────────────────────────┘

Checkpoint helpers (checkpoint_helpers.py):
    Not a memory tier — these are utilities for the LangGraph
    checkpointer that powers Tier 3 persistence. They also enable
    HITL interrupt/resume and fault recovery.

Pattern scripts (scripts/memory_management/) demonstrate each tier:
    Pattern 1 — working_memory_scratchpad.py : Tier 2 (WorkingMemory)
    Pattern 2 — checkpoint_persistence.py   : Checkpointing mechanics
    Pattern 3 — semantic_retrieval.py       : Tier 4 (LongTermMemory RAG)
    Pattern 4 — conversation_memory.py      : Tier 3 (ConversationMemory)
    Pattern 5 — shared_memory_multi_agent.py : All tiers combined
"""

# Tier 2: Per-workflow scratchpad
from memory.working_memory import WorkingMemory

# Tier 3: Per-session conversation history with summarisation
from memory.conversation_memory import ConversationMemory

# Tier 4: Persistent RAG via ChromaDB
from memory.long_term_memory import LongTermMemory

# Checkpointer utilities — graph compilation and state inspection
from memory.checkpoint_helpers import (
    build_checkpointed_graph,
    inspect_checkpoint,
    list_thread_ids,
)

__all__ = [
    # Memory tiers
    "WorkingMemory",
    "ConversationMemory",
    "LongTermMemory",
    # Checkpoint helpers
    "build_checkpointed_graph",
    "inspect_checkpoint",
    "list_thread_ids",
]
