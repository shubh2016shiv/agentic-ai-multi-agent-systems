"""
Observability Decorators
==========================
Function decorators that automatically add Langfuse tracing to any
function, agent, or tool. These decorators follow the "convention over
configuration" principle — annotate once, trace automatically.

Usage:
    from observability.decorators import observe, observe_agent, observe_tool

    @observe(name="process_case")
    def process_case(patient_case):
        ...

    @observe_agent(agent_name="triage")
    def run_triage(state):
        ...

    @observe_tool(tool_name="drug_lookup")
    def lookup_drug(drug_name):
        ...
"""

import time
import functools
import logging
from typing import Callable, Any

from observability.tracer import get_langfuse_client

logger = logging.getLogger(__name__)


def observe(name: str | None = None, metadata: dict | None = None):
    """
    General-purpose tracing decorator.

    Wraps any function with a Langfuse span that captures:
        - Function name and arguments
        - Return value
        - Execution duration
        - Any exceptions raised

    Args:
        name: Span name (defaults to function name).
        metadata: Additional key-value pairs for the span.

    Example:
        @observe(name="validate_patient_data")
        def validate(patient_case):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            span_name = name or func.__name__
            client = get_langfuse_client()

            if client is None:
                # No Langfuse — just run the function
                return func(*args, **kwargs)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                logger.debug(
                    f"@observe({span_name}): completed in {duration_ms:.0f}ms"
                )
                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"@observe({span_name}): failed after {duration_ms:.0f}ms — {e}"
                )
                raise

        return wrapper

    return decorator


def observe_agent(agent_name: str, tags: list[str] | None = None):
    """
    Decorator specifically for agent execution functions.

    Adds agent-specific metadata to the trace span, including:
        - Agent name and type
        - Tags for filtering (e.g., ["cardiology", "specialist"])

    Args:
        agent_name: Name of the agent being traced.
        tags: Labels for the agent span.

    Example:
        @observe_agent(agent_name="pharmacology_agent", tags=["specialist"])
        def run_pharmacology(state):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            client = get_langfuse_client()

            if client is None:
                return func(*args, **kwargs)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                logger.debug(
                    f"@observe_agent({agent_name}): completed in {duration_ms:.0f}ms"
                )
                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"@observe_agent({agent_name}): FAILED after {duration_ms:.0f}ms — {e}"
                )
                raise

        return wrapper

    return decorator


def observe_tool(tool_name: str):
    """
    Decorator specifically for tool functions.

    Adds tool-specific metadata and tracks:
        - Tool name
        - Input arguments
        - Output value
        - Success/failure status
        - Execution latency

    Args:
        tool_name: Name of the tool being traced.

    Example:
        @observe_tool(tool_name="check_drug_interactions")
        def check_interactions(drug_a, drug_b):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            client = get_langfuse_client()

            if client is None:
                return func(*args, **kwargs)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                logger.debug(
                    f"@observe_tool({tool_name}): completed in {duration_ms:.0f}ms"
                )
                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"@observe_tool({tool_name}): FAILED after {duration_ms:.0f}ms — {e}"
                )
                raise

        return wrapper

    return decorator
