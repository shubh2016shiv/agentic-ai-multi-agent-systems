"""
Structured logging configuration using structlog.

Provides correlation ID tracking via contextvars for distributed tracing
across pipeline steps. Supports both JSON (production) and console (dev)
output formats.
"""

import os
import logging
from contextvars import ContextVar
from typing import Any, Dict

import structlog
from structlog.types import EventDict, WrappedLogger


_correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")
_doc_id_var: ContextVar[str] = ContextVar("doc_id", default="")
_step_var: ContextVar[str] = ContextVar("step", default="")


def add_correlation_id(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """
    Processor that adds correlation_id, doc_id, and step from contextvars.
    
    These values persist across log calls within the same execution context,
    enabling distributed tracing without manual parameter passing.
    """
    correlation_id = _correlation_id_var.get()
    doc_id = _doc_id_var.get()
    step = _step_var.get()
    
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    if doc_id:
        event_dict["doc_id"] = doc_id
    if step:
        event_dict["step"] = step
    
    return event_dict


def configure_structlog(log_format: str = "console") -> None:
    """
    Configure structlog for the pipeline.
    
    Args:
        log_format: "json" for production (machine-readable), 
                    "console" for development (human-readable)
    """
    processors = [
        structlog.contextvars.merge_contextvars,
        add_correlation_id,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
    )


def bind_pipeline_context(
    correlation_id: str,
    doc_id: str = "",
    step: str = ""
) -> None:
    """
    Bind pipeline context variables for subsequent log calls.
    
    Args:
        correlation_id: Unique ID for this pipeline run (typically a UUID)
        doc_id: SHA-256 hash of the document being processed
        step: Pipeline step name (e.g., "parse", "chunk", "embed", "write")
    """
    _correlation_id_var.set(correlation_id)
    _doc_id_var.set(doc_id)
    _step_var.set(step)
    
    structlog.contextvars.bind_contextvars(
        correlation_id=correlation_id,
        doc_id=doc_id,
        step=step,
    )


def clear_pipeline_context() -> None:
    """Clear all pipeline context variables."""
    _correlation_id_var.set("")
    _doc_id_var.set("")
    _step_var.set("")
    structlog.contextvars.clear_contextvars()


log_format = os.getenv("LOG_FORMAT", "console")
configure_structlog(log_format)
