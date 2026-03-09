"""
Checkpoint Helpers
===================
Utilities for working with LangGraph checkpointers — the persistence
layer that saves graph state after every node execution.

Where This Fits in the MAS Architecture
-----------------------------------------
A checkpointer is the bridge between LangGraph's in-memory execution
and durable state storage. It enables four critical capabilities:

    1. HITL (Human-in-the-Loop):
       interrupt() REQUIRES a checkpointer. Without one, there is nowhere
       to save state when the graph pauses for human review.

    2. Multi-turn conversations:
       MemorySaver persists conversation state across invoke() calls
       keyed by thread_id. Each turn continues from the saved state.

    3. Fault recovery:
       If a node fails mid-pipeline, the graph can restart from the
       last checkpoint instead of from scratch.

    4. Auditing and debugging:
       graph.get_state(config) returns the full saved state. Inspect
       what each agent produced at each step without re-running.

Checkpointer types — all use the SAME API (drop-in replacements):

    MemorySaver     → in-process dict (dev/testing, lost on restart)
    SqliteSaver     → file-based SQLite (single-machine persistence)
    PostgresSaver   → database-backed (multi-machine production)
    RedisSaver      → sub-millisecond in-memory (distributed systems)

    The graph code stays EXACTLY the same when you swap checkpointers.
    Only the instantiation line changes.

Pattern script:
    scripts/memory_management/checkpoint_persistence.py  — Pattern 2

Usage:
    from memory.checkpoint_helpers import build_checkpointed_graph, inspect_checkpoint

    graph = build_checkpointed_graph(workflow)
    result = graph.invoke(state, {"configurable": {"thread_id": "demo-001"}})

    checkpoint = inspect_checkpoint(graph, thread_id="demo-001")
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# Graph Compilation with Checkpointer
# ============================================================

def build_checkpointed_graph(
    workflow: Any,
    checkpointer: Any = None,
    checkpointer_type: str = "memory",
    **checkpointer_kwargs: Any,
) -> Any:
    """
    Compile a StateGraph workflow with a checkpointer attached.

    Concept — Why compile with a checkpointer:
        graph.compile() without a checkpointer produces a graph where
        state exists ONLY during invoke(). Once invoke() returns, state
        is discarded. interrupt() will raise an error.

        graph.compile(checkpointer=...) produces a graph where state is
        saved after every node. thread_id is the key — different thread
        IDs are isolated, independent executions.

    Checkpointer selection (checkpointer_type):
        "memory"    → MemorySaver() — default, in-process
        "sqlite"    → SqliteSaver   — requires db_path kwarg
        "postgres"  → PostgresSaver — requires connection_string kwarg
        "redis"     → RedisSaver    — requires connection_string kwarg

    Args:
        workflow: An uncomplied StateGraph (before .compile()).
        checkpointer: An already-instantiated checkpointer. If provided,
            checkpointer_type is ignored.
        checkpointer_type: Which checkpointer to create. Default "memory".
        **checkpointer_kwargs: Passed to the checkpointer constructor.
            For "sqlite": db_path="checkpoints.db"
            For "postgres": connection_string="postgresql://..."
            For "redis": connection_string="redis://localhost:6379"

    Returns:
        Compiled CompiledStateGraph with the checkpointer attached.

    Example:
        # Development (in-memory, zero config):
        graph = build_checkpointed_graph(workflow)

        # Production (SQLite persistence):
        graph = build_checkpointed_graph(
            workflow,
            checkpointer_type="sqlite",
            db_path="data/checkpoints.db",
        )
    """
    if checkpointer is not None:
        # Use the provided checkpointer directly
        return workflow.compile(checkpointer=checkpointer)

    cp = _create_checkpointer(checkpointer_type, **checkpointer_kwargs)
    return workflow.compile(checkpointer=cp)


def _create_checkpointer(
    checkpointer_type: str,
    **kwargs: Any,
) -> Any:
    """
    Instantiate a checkpointer by type name.

    Concept — The checkpointer is the only thing that changes between
    development and production. The graph topology, node logic, and
    state schemas are identical. This function centralises the
    "which checkpointer do I use?" decision in one place.

    Args:
        checkpointer_type: "memory", "sqlite", "postgres", or "redis"
        **kwargs: Constructor arguments for the selected checkpointer.

    Returns:
        An instantiated checkpointer ready to pass to workflow.compile().

    Raises:
        ValueError: If checkpointer_type is not recognised.
        ImportError: If the required package for the checkpointer is not installed.
    """
    if checkpointer_type == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        logger.debug("Using MemorySaver (in-memory, dev/testing only)")
        return MemorySaver()

    elif checkpointer_type == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            db_path = kwargs.get("db_path", "checkpoints.db")
            logger.info(f"Using SqliteSaver (path: {db_path})")
            return SqliteSaver.from_conn_string(db_path)
        except ImportError:
            raise ImportError(
                "SqliteSaver requires: pip install langgraph-checkpoint-sqlite"
            )

    elif checkpointer_type == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            conn_str = kwargs.get("connection_string", kwargs.get("db_url"))
            if not conn_str:
                raise ValueError("postgres checkpointer requires 'connection_string' kwarg")
            logger.info("Using PostgresSaver (multi-machine persistence)")
            return PostgresSaver.from_conn_string(conn_str)
        except ImportError:
            raise ImportError(
                "PostgresSaver requires: pip install langgraph-checkpoint-postgres"
            )

    elif checkpointer_type == "redis":
        try:
            from langgraph_checkpoint_redis import RedisSaver
            conn_str = kwargs.get("connection_string", kwargs.get("redis_url", "redis://localhost:6379"))
            logger.info(f"Using RedisSaver (url: {conn_str})")
            return RedisSaver.from_conn_string(conn_str)
        except ImportError:
            raise ImportError(
                "RedisSaver requires: pip install langgraph-checkpoint-redis"
            )

    else:
        raise ValueError(
            f"Unknown checkpointer_type: '{checkpointer_type}'. "
            f"Valid options: 'memory', 'sqlite', 'postgres', 'redis'"
        )


# ============================================================
# Checkpoint Inspection
# ============================================================

def inspect_checkpoint(
    graph: Any,
    thread_id: str,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Retrieve and optionally display the saved state for a thread.

    Concept — graph.get_state() for auditing:
        After graph.invoke(), the checkpointer holds the full state
        keyed by thread_id. graph.get_state(config) returns a
        StateSnapshot with:
            .values   — the actual state dict
            .next     — next nodes to execute (empty if completed)
            .config   — the config used for this snapshot
            .metadata — execution metadata

        This is invaluable for:
            Debugging: "What did the agent produce at step 2?"
            Testing:   Assertions on saved state without re-running
            Auditing:  Full audit trail of every agent's output
            HITL:      Read the paused state before resuming

    Args:
        graph: A compiled graph with a checkpointer.
        thread_id: The thread ID to inspect.
        verbose: If True, print a summary of the checkpoint.

    Returns:
        The state values dict (snapshot.values).
        Returns empty dict if no checkpoint found for thread_id.

    Example:
        graph = build_checkpointed_graph(workflow)
        result = graph.invoke(state, {"configurable": {"thread_id": "t1"}})
        saved = inspect_checkpoint(graph, thread_id="t1")
        assert saved["assessment"] == result["assessment"]
    """
    config = {"configurable": {"thread_id": thread_id}}

    try:
        snapshot = graph.get_state(config)
        values = snapshot.values

        if verbose:
            print(f"\n    Checkpoint for thread_id='{thread_id}':")
            print(f"      Keys: {list(values.keys())}")
            for key, val in values.items():
                if isinstance(val, list):
                    print(f"      {key}: list({len(val)} items)")
                elif isinstance(val, str) and len(val) > 100:
                    print(f"      {key}: '{val[:100]}...'")
                else:
                    print(f"      {key}: {repr(val)[:100]}")

            next_nodes = snapshot.next
            if next_nodes:
                print(f"      next_nodes: {next_nodes}")
            else:
                print(f"      status: COMPLETED (no pending nodes)")

        logger.debug(f"Checkpoint retrieved for thread_id='{thread_id}'")
        return values

    except Exception as e:
        logger.warning(f"Could not retrieve checkpoint for thread_id='{thread_id}': {e}")
        if verbose:
            print(f"\n    No checkpoint found for thread_id='{thread_id}'")
            print(f"    (Graph may not have a checkpointer, or thread_id not found)")
        return {}


# ============================================================
# Thread Isolation Check
# ============================================================

def list_thread_ids(graph: Any) -> list[str]:
    """
    List all thread IDs that have saved checkpoints.

    Concept — Thread isolation:
        Each thread_id is a completely independent execution. Two
        invocations with the same thread_id share state; two with
        different thread_ids are isolated. This is the foundation
        of multi-user, multi-session graph applications.

        In a web application:
            thread_id = session_id  (one per user session)
            graph     = shared      (one compiled graph, many threads)

    Args:
        graph: A compiled graph with a MemorySaver checkpointer.

    Returns:
        List of thread ID strings that have saved checkpoints.
        Returns empty list if the checkpointer doesn't support listing
        or if no threads have been used.
    """
    try:
        checkpointer = graph.checkpointer
        if checkpointer is None:
            return []
        # MemorySaver stores threads in its .storage dict
        if hasattr(checkpointer, "storage"):
            return list(checkpointer.storage.keys())
        return []
    except Exception:
        return []
