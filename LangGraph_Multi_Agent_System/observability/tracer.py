"""
Langfuse Tracer
=================
Initializes and manages the Langfuse client for distributed tracing
of multi-agent workflows. Provides helper functions for creating
traces, spans, and generations.

Langfuse Concepts (mapped to MAS):
    - Trace: Represents an entire user request / workflow execution
    - Span: A sub-operation within a trace (e.g., one agent's execution)
    - Generation: An LLM call with prompt, completion, and usage stats

Langfuse v3 changes:
    - Langfuse() is initialized once as a singleton; access via get_client()
    - Traces and spans created with context managers (start_as_current_observation)
    - NoOpTrace / NoOpSpan still used when Langfuse is not configured

Design Decisions:
    - Graceful degradation: if Langfuse is not configured, tracing is
      silently disabled (never blocks the core system)
    - Session grouping: traces from the same user session are linked

Usage:
    from observability.tracer import get_langfuse_client, create_trace

    trace = create_trace(name="clinical_workflow", user_id="doctor-1")
    span = trace.span(name="triage_agent", input={"symptoms": [...]})
    span.end(output={"urgency": "urgent"})
"""

import logging
from typing import Any

from core.config import settings
from core.exceptions import ObservabilityError

logger = logging.getLogger(__name__)

# Module-level flag so we only initialize the Langfuse singleton once
_langfuse_initialized: bool = False


def get_langfuse_client():
    """
    Get or initialize the Langfuse v3 singleton client.

    In Langfuse v3, the client is a module-level singleton accessed via
    get_client(). We initialize it here on first call so the rest of the
    system can use get_client() directly without worrying about setup order.

    Returns None if Langfuse is not configured (keys are empty or placeholders),
    allowing the system to run without observability in development.

    Returns:
        Langfuse client instance or None if not configured.
    """
    global _langfuse_initialized

    if not settings.langfuse_public_key or settings.langfuse_public_key.startswith("your-"):
        logger.warning(
            "Langfuse is not configured (missing or placeholder keys). "
            "Observability tracing is DISABLED. Set LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY in .env to enable."
        )
        return None

    try:
        from langfuse import Langfuse, get_client

        if not _langfuse_initialized:
            Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            _langfuse_initialized = True
            logger.info(f"Langfuse v3 client initialized → {settings.langfuse_host}")

        return get_client()

    except ImportError:
        logger.warning("langfuse package not installed. Run: pip install langfuse")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {e}")
        return None


def create_trace(
    name: str,
    user_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
):
    """
    Create a new Langfuse trace for a workflow execution (Langfuse v3).

    In Langfuse v3, traces are created with start_as_current_observation()
    context manager. For backward compatibility this function returns a
    NoOpTrace when the context manager API is not used, and logs the
    trace creation for debugging.

    A trace represents a complete user request flowing through the
    multi-agent system. All agent calls, tool invocations, and handoffs
    within this workflow will be nested under this trace.

    Args:
        name: Human-readable name (e.g., "clinical_decision_support").
        user_id: Identifier for the requesting user/doctor.
        session_id: Groups multiple traces into a session.
        metadata: Additional key-value pairs for filtering in the UI.
        tags: Labels for categorization (e.g., ["cardiology", "urgent"]).

    Returns:
        Langfuse client with trace context, or a NoOpTrace if Langfuse
        is unavailable.
    """
    client = get_langfuse_client()

    if client is None:
        return NoOpTrace()

    try:
        # In Langfuse v3, use start_as_current_observation for trace context.
        # For simple script-level usage, we update_current_observation with
        # the trace attributes so they appear correctly in the dashboard.
        obs = client.start_as_current_observation(
            name=name,
            as_type="trace",
            metadata=metadata or {},
            tags=tags or [],
        )
        if user_id:
            client.update_current_trace(user_id=user_id)
        if session_id:
            client.update_current_trace(session_id=session_id)

        logger.debug(f"Created trace: {name}")
        return obs

    except Exception as e:
        logger.error(f"Failed to create Langfuse trace: {e}")
        return NoOpTrace()


class NoOpTrace:
    """
    A no-op implementation of the Langfuse trace interface.

    Used when Langfuse is not configured, ensuring that tracing code
    never needs conditional checks. All methods accept any args and
    return NoOp objects — the Null Object pattern.

    This is critical for the principle: "Observability should observe,
    not interfere." The core system runs identically with or without
    Langfuse configured.
    """

    id: str = "noop-trace"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def span(self, **kwargs) -> "NoOpSpan":
        return NoOpSpan()

    def generation(self, **kwargs) -> "NoOpSpan":
        return NoOpSpan()

    def update(self, **kwargs) -> "NoOpTrace":
        return self

    def end(self, **kwargs) -> None:
        pass

    def score(self, **kwargs) -> None:
        pass


class NoOpSpan:
    """No-op span that does nothing — see NoOpTrace."""

    id: str = "noop-span"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def span(self, **kwargs) -> "NoOpSpan":
        return NoOpSpan()

    def generation(self, **kwargs) -> "NoOpSpan":
        return NoOpSpan()

    def update(self, **kwargs) -> "NoOpSpan":
        return self

    def end(self, **kwargs) -> None:
        pass

    def score(self, **kwargs) -> None:
        pass
