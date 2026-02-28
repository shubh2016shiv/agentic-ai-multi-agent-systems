"""
Custom Exception Hierarchy
===========================
A well-designed exception hierarchy is essential for enterprise multi-agent
systems. Each exception type maps to a specific failure mode, enabling:

    1. Targeted error handling (catch only what you can handle)
    2. Clear error reporting (each exception carries context)
    3. Resilience pattern integration (circuit breakers catch specific types)
    4. Observability (Langfuse traces include exception metadata)

Hierarchy:
    MASBaseException
    ├── GuardrailTripped          — Input/output/tool guardrail violations
    ├── TokenBudgetExceeded       — Agent or workflow exceeded token limit
    ├── CircuitBreakerOpen        — LLM service is unhealthy, calls blocked
    ├── HandoffLimitReached       — Too many agent-to-agent handoffs (loop prevention)
    ├── AgentExecutionError       — Agent failed during reasoning or tool execution
    ├── DocumentProcessingError   — Docling/document ingestion failures
    └── ObservabilityError        — Non-fatal: Langfuse tracing failures
"""


class MASBaseException(Exception):
    """
    Base exception for all Multi-Agent System errors.

    All custom exceptions inherit from this class, allowing callers to
    catch all MAS-related errors with a single except clause when needed.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary with structured error context.
    """

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} [{detail_str}]"
        return self.message


class GuardrailTripped(MASBaseException):
    """
    Raised when an input, output, or tool guardrail blocks a request.

    This is NOT an error — it is the guardrail working as designed.
    The system should respond gracefully with a user-friendly message
    explaining why the request was blocked.

    Example:
        GuardrailTripped(
            "PII detected in patient query",
            details={"guardrail": "input_pii_filter", "field": "chief_complaint"}
        )
    """

    pass


class TokenBudgetExceeded(MASBaseException):
    """
    Raised when an agent call or workflow exceeds the configured token budget.

    This prevents runaway costs from long reasoning chains or excessive
    tool calling. The token_manager in the resilience module tracks usage
    and raises this when limits are hit.

    Example:
        TokenBudgetExceeded(
            "Workflow token budget exceeded",
            details={"used": 35000, "limit": 32000, "workflow_id": "wf-123"}
        )
    """

    pass


class CircuitBreakerOpen(MASBaseException):
    """
    Raised when the circuit breaker is in OPEN state, blocking calls to
    an unhealthy service (typically the LLM API).

    The circuit breaker opens after N consecutive failures (configurable)
    and automatically attempts recovery after a timeout period. During
    the open period, this exception is raised immediately without
    attempting the call — this is the "fail fast" pattern.

    Example:
        CircuitBreakerOpen(
            "OpenAI API circuit breaker is open",
            details={"fail_count": 5, "reset_timeout": 60}
        )
    """

    pass


class HandoffLimitReached(MASBaseException):
    """
    Raised when the maximum number of agent-to-agent handoffs is exceeded.

    This is the primary defense against the "Infinite Handoff Loop"
    anti-pattern (Chapter 9). If agents keep passing work to each other
    without resolution, this exception forces escalation to a human
    operator or a fallback response.

    Example:
        HandoffLimitReached(
            "Maximum handoff depth (3) exceeded",
            details={"depth": 4, "chain": ["Triage", "Cardiology", "Nephrology", "Cardiology"]}
        )
    """

    pass


class AgentExecutionError(MASBaseException):
    """
    Raised when an agent fails during its reasoning or tool execution phase.

    This wraps underlying errors (LLM API errors, tool failures, parsing
    errors) to provide consistent error handling at the orchestration layer.

    Example:
        AgentExecutionError(
            "DiagnosisAgent failed during symptom analysis",
            details={"agent": "DiagnosisAgent", "step": "tool_execution", "tool": "analyze_symptoms"}
        )
    """

    pass


class DocumentProcessingError(MASBaseException):
    """
    Raised when document processing (Docling PDF/Excel ingestion) fails.

    This covers failures in reading, parsing, chunking, or extracting
    structured data from medical guidelines or drug databases.

    Example:
        DocumentProcessingError(
            "Failed to process PDF",
            details={"file": "COPD GOLD 2024.pdf", "reason": "corrupted file"}
        )
    """

    pass


class ObservabilityError(MASBaseException):
    """
    Raised when Langfuse tracing encounters an error.

    This is intentionally NON-FATAL. Observability failures should NEVER
    prevent the core system from functioning. The system logs the error
    and continues processing. This follows the principle that monitoring
    should observe, not interfere.

    Example:
        ObservabilityError(
            "Failed to send trace to Langfuse",
            details={"trace_id": "tr-456", "reason": "connection timeout"}
        )
    """

    pass
