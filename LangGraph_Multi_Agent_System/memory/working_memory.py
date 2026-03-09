"""
Working Memory
================
Short-lived, in-graph memory that persists for the duration of a single
workflow execution. Stored in LangGraph's state (TypedDict) and passed
between nodes.

Working memory is the agent's "scratchpad" — it holds the current patient
case, intermediate reasoning results, and handoff context as they flow
through the agent pipeline.

Where This Fits in the MAS Architecture
-----------------------------------------
The multi-tier memory architecture (from MAS reference architecture):

    Working Memory (this module)
        Scope: ONE workflow execution
        Storage: In-process Python dict (no persistence)
        Purpose: Carry intermediate state between nodes
        Example: triage result → pharmacist → report

    Short-Term / Conversation Memory (memory/conversation_memory.py)
        Scope: ONE conversation session (multiple turns)
        Storage: LangGraph MemorySaver (in-process)
        Purpose: Multi-turn dialogue history with summarisation
        Example: patient follows up on previous advice

    Long-Term Memory (memory/long_term_memory.py)
        Scope: Persistent across ALL sessions
        Storage: ChromaDB vector store
        Purpose: RAG retrieval of medical guidelines, drug info
        Example: retrieve COPD guidelines for current patient

Working memory vs LangGraph state (TypedDict):
    LangGraph state IS working memory for most pipelines.
    WorkingMemory is useful when you need:
        - Agent-local scratch space (not shared with other agents)
        - append_to() for accumulating lists (reasoning traces)
        - to_context_string() to serialise memory into LLM prompts
        - A clean API that hides the raw dict from node logic

Pattern scripts:
    scripts/memory_management/working_memory_scratchpad.py  — Pattern 1
    scripts/memory_management/shared_memory_multi_agent.py  — Pattern 5

Key Concepts:
    - Scoped: Each workflow execution has its own working memory
    - Typed: Pydantic models enforce data structure
    - Ephemeral: Cleared at the end of each workflow
    - Shared: All agents in the graph can read/write to it

Usage:
    from memory.working_memory import WorkingMemory

    memory = WorkingMemory()
    memory.set("patient_case", patient_case)
    memory.append_to("reasoning_trace", "Step 1: Analyzed symptoms")
    case = memory.get("patient_case")
"""

import logging
from typing import Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class WorkingMemory:
    """
    In-memory key-value store for workflow-scoped data.

    This is a simple, typed wrapper around a dictionary that provides
    convenient methods for the patterns used in multi-agent workflows:
        - get/set for single values
        - append for accumulating lists (reasoning traces, findings)
        - scratch for temporary agent-local data

    Args:
        initial_data: Optional dictionary of initial key-value pairs.
    """

    def __init__(self, initial_data: dict[str, Any] | None = None):
        self._store: dict[str, Any] = dict(initial_data or {})
        self._scratch: dict[str, dict[str, Any]] = defaultdict(dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from working memory."""
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in working memory."""
        self._store[key] = value
        logger.debug(f"WorkingMemory.set: {key} = {type(value).__name__}")

    def append_to(self, key: str, value: Any) -> None:
        """
        Append a value to a list in working memory.

        If the key doesn't exist, creates a new list. This is commonly
        used for accumulating reasoning traces, findings, and recommendations
        as data flows through the agent pipeline.
        """
        if key not in self._store:
            self._store[key] = []
        if isinstance(self._store[key], list):
            self._store[key].append(value)
        else:
            raise TypeError(f"Cannot append to non-list key '{key}' (type: {type(self._store[key]).__name__})")

    def get_scratch(self, agent_name: str) -> dict[str, Any]:
        """
        Get the scratch space for a specific agent.

        Each agent has its own private scratch area within working memory
        for temporary data that shouldn't be visible to other agents.
        """
        return self._scratch[agent_name]

    def set_scratch(self, agent_name: str, key: str, value: Any) -> None:
        """Set a value in an agent's private scratch space."""
        self._scratch[agent_name][key] = value

    def get_all(self) -> dict[str, Any]:
        """Return the entire working memory contents."""
        return dict(self._store)

    def keys(self) -> list[str]:
        """List all keys in working memory."""
        return list(self._store.keys())

    def clear(self) -> None:
        """Clear all working memory (end of workflow)."""
        self._store.clear()
        self._scratch.clear()
        logger.debug("WorkingMemory cleared")

    def to_context_string(self, max_length: int = 2000) -> str:
        """
        Serialize working memory to a string for LLM context injection.

        This converts the current memory state into a human-readable
        format that can be included in agent prompts, allowing agents
        to "see" what previous agents have recorded.

        Args:
            max_length: Maximum string length (for token management).

        Returns:
            Formatted string representation of memory contents.
        """
        lines = ["=== Working Memory ==="]
        for key, value in self._store.items():
            if isinstance(value, list):
                lines.append(f"\n{key}:")
                for item in value[-5:]:  # Last 5 items to avoid overflow
                    lines.append(f"  - {str(item)[:200]}")
            elif isinstance(value, dict):
                lines.append(f"\n{key}: {str(value)[:300]}")
            else:
                lines.append(f"\n{key}: {str(value)[:300]}")

        result = "\n".join(lines)
        if len(result) > max_length:
            result = result[:max_length] + "\n... [truncated]"
        return result
