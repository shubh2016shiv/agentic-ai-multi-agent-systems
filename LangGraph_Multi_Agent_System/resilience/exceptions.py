"""
Resilience Exception Hierarchy
================================
A single, well-typed exception hierarchy for every failure mode the
resilience module can produce.

WHY A DEDICATED HIERARCHY?
    Scattered, generic exceptions (ValueError, RuntimeError) make it
    impossible for callers — especially LangGraph node wrappers — to
    distinguish *what* failed and *how to respond*. A typed hierarchy
    lets the orchestrator make surgical decisions:

        except CircuitBreakerOpen:
            return cached_fallback()      # service is down, use cache
        except TokenBudgetExceeded:
            return partial_result()       # out of budget, return what we have
        except RateLimitExhausted:
            raise                         # propagate up, scheduler will retry

SYSTEM DESIGN NOTE — "Let it fail fast, fail clearly":
    In a multi-agent pipeline Agent A → Agent B → Agent C, a silent
    failure or a generic Exception in Agent A will corrupt Agent B's
    input and cascade through the whole pipeline before anything is
    caught. A typed exception with structured details allows the
    LangGraph supervisor node to catch it at the boundary, log it
    with full context, and decide whether to retry, fallback, or abort.

Exception Tree:
    ResilienceError (base)
    ├── CircuitBreakerOpen      — service is known-unhealthy, fail fast
    ├── RateLimitExhausted      — quota hit, caller should back off
    ├── TimeoutExceeded         — deadline passed, abort
    ├── AllFallbacksFailed      — every provider in the chain is down
    ├── BulkheadFull            — resource pool exhausted, shed load
    └── TokenBudgetExceeded     — workflow/agent token quota exceeded
"""

from __future__ import annotations

from typing import Any


class ResilienceError(Exception):
    """
    Base class for all resilience-layer failures.

    Carries a `details` dict so callers can log structured context
    without parsing string messages.

    Attributes:
        message:  Human-readable description.
        details:  Structured key/value context (agent name, counts, etc.).
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details})"


class CircuitBreakerOpen(ResilienceError):
    """
    Raised when the circuit breaker is in the OPEN state.

    This means the downstream service has exceeded its failure threshold
    and is considered unhealthy. Calls are rejected immediately (fail fast)
    to avoid adding latency and wasting resources on calls that will fail.

    Caller guidance:
        - Return a cached / fallback response immediately.
        - Do NOT retry — the circuit will reset itself after `reset_timeout`.

    Details keys:
        breaker_name  (str)  — which breaker tripped
        fail_count    (int)  — consecutive failures that caused the trip
        state         (str)  — current state string ("open", "half-open")
    """


class RateLimitExhausted(ResilienceError):
    """
    Raised when the rate limiter's quota is exceeded AND blocking is disabled.

    By default the rate limiter *blocks* (sleeps) until the window resets.
    Set `block=False` on the RateLimiter to raise this instead.

    Caller guidance:
        - Exponential-backoff the calling agent and retry after `retry_after` seconds.

    Details keys:
        limiter_name  (str)   — which limiter was hit
        limit         (int)   — max_calls per period
        period        (float) — the time window in seconds
        retry_after   (float) — seconds until the next slot opens
    """


class TimeoutExceeded(ResilienceError):
    """
    Raised when an LLM call or tool call exceeds its configured deadline.

    Timeouts are critical in multi-agent pipelines. Without them, one
    slow/hung agent blocks the entire downstream chain indefinitely.

    Caller guidance:
        - Record the partial state (if any) in LangGraph state.
        - Decide whether to retry, fall back, or abort the workflow.

    Details keys:
        agent_name    (str)   — which agent timed out
        timeout_secs  (float) — the deadline that was exceeded
        elapsed_secs  (float) — how long the call actually ran
    """


class AllFallbacksFailed(ResilienceError):
    """
    Raised by FallbackChain when every provider in the chain has failed.

    This is the "last resort" exception — it means the primary provider
    AND all backup providers are unavailable at this moment.

    Caller guidance:
        - Return a safe, static error response to the end user.
        - Page on-call: this indicates a multi-provider outage.

    Details keys:
        providers_tried  (list[str]) — ordered list of providers attempted
        last_error       (str)       — error message from the last provider
    """


class BulkheadFull(ResilienceError):
    """
    Raised when the bulkhead's resource pool is at capacity.

    The bulkhead pattern prevents one agent from starving others by
    capping the number of concurrent calls per pool. When the pool is
    full, new callers are rejected rather than queued indefinitely.

    Caller guidance:
        - Treat as a temporary backpressure signal.
        - Retry with a short jitter delay (50–200ms).

    Details keys:
        pool_name       (str) — which bulkhead pool is full
        max_concurrent  (int) — the pool's capacity limit
        current_active  (int) — calls currently in flight
    """


class TokenBudgetExceeded(ResilienceError):
    """
    Raised when an agent or the overall workflow exceeds its token budget.

    Token budgets prevent runaway costs. A single reasoning loop can
    iterate 10+ times; without a budget, one workflow can cost $5–10.

    Caller guidance:
        - If per-agent budget: return a truncated but valid response.
        - If workflow budget: abort the workflow and return partial results.

    Details keys:
        scope           (str) — "agent" or "workflow"
        agent_name      (str) — which agent triggered the check
        used            (int) — tokens consumed so far
        limit           (int) — the budget ceiling
        estimated_add   (int) — tokens the next call would have consumed
    """
