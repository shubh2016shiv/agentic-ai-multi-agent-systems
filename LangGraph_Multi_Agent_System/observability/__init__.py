"""
Observability Package (Langfuse)
==================================
Provides end-to-end tracing for the multi-agent system using Langfuse.
All LLM calls, tool invocations, and agent handoffs are automatically
traced for debugging, cost tracking, and performance analysis.
"""

from observability.tracer import get_langfuse_client, create_trace
from observability.callbacks import get_langfuse_callback_handler
from observability.decorators import observe, observe_agent, observe_tool

__all__ = [
    "get_langfuse_client",
    "create_trace",
    "get_langfuse_callback_handler",
    "observe",
    "observe_agent",
    "observe_tool",
]
