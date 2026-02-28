"""
Langfuse Callback Handler for LangChain
==========================================
Integrates Langfuse with LangChain's callback system to automatically
capture all LLM calls, tool invocations, and chain executions.

This is the PRIMARY integration point between Langfuse and the
multi-agent system. By passing this callback handler to LLM calls,
every interaction is automatically traced without modifying agent code.

Langfuse v3 changes:
  - CallbackHandler takes no constructor args; configured via env vars.
  - Trace metadata (user_id, session_id, tags) passed via
    config["metadata"] with "langfuse_" prefixed keys.
  - Use get_client() for the singleton Langfuse client.

Usage:
    from observability.callbacks import build_callback_config

    config = build_callback_config(trace_name="clinical_workflow")
    result = llm.invoke(prompt, config=config)
"""

import logging
from typing import Any

from core.config import settings

logger = logging.getLogger(__name__)


def _init_langfuse_singleton() -> bool:
    """
    Initialize the Langfuse singleton client (Langfuse v3 pattern).

    In Langfuse v3 the SDK uses a module-level singleton. This function
    initializes it once with our configured keys so that the CallbackHandler
    (which reads from the singleton) works correctly.

    Returns True if initialization succeeded, False otherwise.
    """
    if not settings.langfuse_public_key or settings.langfuse_public_key.startswith("your-"):
        return False

    try:
        from langfuse import Langfuse

        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        return True
    except ImportError:
        logger.warning("langfuse package not installed. Tracing unavailable.")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse singleton: {e}")
        return False


def get_langfuse_callback_handler(
    trace_name: str = "langgraph_workflow",
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """
    Create a Langfuse CallbackHandler for LangChain integration (v3).

    In Langfuse v3, CallbackHandler takes no constructor arguments. Keys
    are configured via the singleton initialized at startup (or via env
    vars LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL).

    Trace metadata such as trace_name, user_id, session_id, and tags are
    injected into the LangChain config dict via build_callback_config().

    Args:
        trace_name: Name for the trace in Langfuse UI (passed via metadata).
        user_id: Doctor/user identifier for filtering.
        session_id: Session grouping identifier.
        tags: Labels for trace categorization.
        metadata: Additional context for the trace.

    Returns:
        CallbackHandler instance, or None if Langfuse is unavailable.
    """
    if not settings.langfuse_public_key or settings.langfuse_public_key.startswith("your-"):
        logger.debug("Langfuse not configured — returning None callback handler")
        return None

    try:
        from langfuse.langchain import CallbackHandler

        if not _init_langfuse_singleton():
            return None

        handler = CallbackHandler()
        logger.debug(f"Created Langfuse v3 callback handler (trace: {trace_name})")
        return handler

    except ImportError:
        logger.warning("langfuse package not installed. Callback handler unavailable.")
        return None
    except Exception as e:
        logger.error(f"Failed to create Langfuse callback handler: {e}")
        return None


def build_callback_config(
    trace_name: str = "langgraph_workflow",
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    additional_callbacks: list | None = None,
) -> dict:
    """
    Build a LangChain config dict with Langfuse callback (Langfuse v3).

    In Langfuse v3, trace metadata is passed through config["metadata"]
    using "langfuse_" prefixed keys rather than as CallbackHandler
    constructor arguments.

    Args:
        trace_name: Name for the Langfuse trace.
        user_id: User identifier.
        session_id: Session identifier.
        tags: Trace tags.
        additional_callbacks: Other callback handlers to include.

    Returns:
        Config dict compatible with LangChain/LangGraph:
        {"callbacks": [...], "metadata": {"langfuse_trace_name": ..., ...}}

    Example:
        config = build_callback_config(trace_name="triage_flow")
        result = graph.invoke(state, config=config)
    """
    callbacks = list(additional_callbacks or [])

    handler = get_langfuse_callback_handler(
        trace_name=trace_name,
        user_id=user_id,
        session_id=session_id,
        tags=tags,
    )

    if handler is not None:
        callbacks.append(handler)

    config: dict[str, Any] = {}

    if callbacks:
        config["callbacks"] = callbacks

    # Langfuse v3: inject trace attributes via metadata with langfuse_ prefix
    langfuse_metadata: dict[str, Any] = {}
    if trace_name:
        langfuse_metadata["langfuse_trace_name"] = trace_name
    if user_id:
        langfuse_metadata["langfuse_user_id"] = user_id
    if session_id:
        langfuse_metadata["langfuse_session_id"] = session_id
    if tags:
        langfuse_metadata["langfuse_tags"] = tags

    if langfuse_metadata:
        config["metadata"] = langfuse_metadata

    return config
