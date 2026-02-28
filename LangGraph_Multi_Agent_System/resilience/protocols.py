"""
Resilience Protocols (Interfaces)
====================================
Abstract contracts for every resilience component, using Python's
structural subtyping via `typing.Protocol`.

════════════════════════════════════════════════════════════════════════════════
  *** WHERE AND HOW PROTOCOLS ARE USED IN MAS ***
════════════════════════════════════════════════════════════════════════════════

**PURPOSE**: Define interfaces (contracts) for resilience components WITHOUT
requiring explicit inheritance. Any class implementing the required methods
automatically satisfies the protocol (structural typing).

**PRIMARY CONSUMER**: resilience/resilient_caller.py (ResilientCaller)
    - ResilientCaller composes all resilience components via these protocols
    - Instantiates concrete implementations: CircuitBreaker, RateLimiter,
      RetryHandler, TimeoutGuard, Bulkhead, TokenManager
    - Type hints use protocols for dependency injection flexibility

**WHERE PROTOCOLS ARE USED**:
    1. resilience/resilient_caller.py — ResilientCaller.__init__()
       - Accepts CircuitBreaker via CircuitBreakerProtocol
       - Creates RateLimiter (RateLimiterProtocol)
       - Creates RetryHandler (RetryHandlerProtocol)
       - Creates TimeoutGuard (TimeoutGuardProtocol)
       - Creates Bulkhead (BulkheadProtocol)
       - Accepts TokenManager via TokenTrackerProtocol

    2. scripts/orchestration/_base/orchestrator.py (indirect)
       - Uses ResilientCaller which internally composes all protocols
       - All specialist/synthesis calls flow through this stack

    3. resilience/langgraph_integration_example.py (reference)
       - Demonstrates per-node protocol composition pattern

**WHY PROTOCOLS INSTEAD OF ABSTRACT BASE CLASSES?**
    ABCs require explicit inheritance (`class MyBreaker(CircuitBreakerProtocol)`).
    Protocols use structural typing ("duck typing with type hints") — any class
    that implements the required methods satisfies the interface automatically.

    This matters for resilience components because:
    1. You can wrap third-party libraries (pybreaker, resilience4j bindings)
       without touching their source code.
    2. Test doubles (stubs, mocks) satisfy the protocol with zero boilerplate.
    3. Alternative implementations (Redis-backed, distributed) can slot in
       without changing any calling code.

SOLID — INTERFACE SEGREGATION PRINCIPLE (ISP):
    Each protocol is small and focused on ONE capability. Callers only
    depend on the interface they actually use.

    Bad (fat interface):
        class ResilienceManager:
            def check_circuit(self): ...
            def acquire_rate_limit(self): ...
            def record_tokens(self): ...

    Good (segregated interfaces — this file):
        class CircuitBreakerProtocol: ...   # callers that need circuit checking
        class RateLimiterProtocol:   ...   # callers that need throttling
        class TokenTrackerProtocol:  ...   # callers that need budget tracking

SOLID — DEPENDENCY INVERSION PRINCIPLE (DIP):
    High-level modules (ResilientCaller, LangGraph nodes) depend on these
    abstractions, NOT on concrete implementations. This means you can swap
    `PyBreakerCircuitBreaker` for `InMemoryCircuitBreaker` in tests without
    changing the caller.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CircuitBreakerProtocol(Protocol):
    """
    Contract for a circuit breaker.

    A circuit breaker wraps a callable and monitors it for failures.
    When failures exceed a threshold, it "opens" and rejects all
    calls immediately (fail fast) until the service recovers.

    Implementors: CircuitBreaker (circuit_breaker.py)
    """

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute `func` through the circuit breaker.

        Raises:
            CircuitBreakerOpen: if the breaker is currently OPEN.
        """
        ...

    @property
    def is_open(self) -> bool:
        """True if the breaker is OPEN (rejecting calls)."""
        ...

    @property
    def name(self) -> str:
        """Unique identifier for this breaker (used in logs and metrics)."""
        ...


@runtime_checkable
class RateLimiterProtocol(Protocol):
    """
    Contract for a rate limiter.

    A rate limiter controls the frequency of calls to protect against
    quota exhaustion. It can either block the caller until a slot is
    available, or raise RateLimitExhausted to signal backpressure.

    Implementors: RateLimiter (rate_limiter.py)
    """

    def acquire(self) -> None:
        """
        Acquire permission to make one call.

        Blocks (or raises RateLimitExhausted, depending on config)
        if the rate limit is currently exceeded.
        """
        ...


@runtime_checkable
class RetryHandlerProtocol(Protocol):
    """
    Contract for a retry handler.

    Wraps a callable to automatically retry on transient failures,
    using exponential backoff and jitter to avoid thundering-herd.

    Implementors: RetryHandler (retry_handler.py)
    """

    def call_with_retry(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute `func`, retrying on configured transient exceptions.

        Raises:
            The last exception if all retries are exhausted.
        """
        ...


@runtime_checkable
class TimeoutGuardProtocol(Protocol):
    """
    Contract for a timeout/deadline enforcer.

    Ensures that any single LLM call returns within a configured
    deadline. Without timeouts, one hung agent blocks the entire
    downstream pipeline indefinitely.

    Implementors: TimeoutGuard (timeout_guard.py)
    """

    def call_with_timeout(
        self,
        func: Callable,
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute `func`, raising TimeoutExceeded if it takes too long.

        Args:
            func:    The callable to execute.
            timeout: Override deadline in seconds. Uses default if None.

        Raises:
            TimeoutExceeded: if the call exceeds the deadline.
        """
        ...


@runtime_checkable
class FallbackChainProtocol(Protocol):
    """
    Contract for a provider fallback chain.

    Tries providers (LLM callers) in priority order. If the primary
    fails (error or timeout), it silently falls through to the next.
    This is the "provider failover" pattern.

    Implementors: FallbackChain (fallback_chain.py)
    """

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """
        Try each provider in order, returning the first successful result.

        Raises:
            AllFallbacksFailed: if every provider in the chain fails.
        """
        ...


@runtime_checkable
class BulkheadProtocol(Protocol):
    """
    Contract for a bulkhead resource pool.

    Isolates a group of agents into a bounded concurrency pool so that
    a burst from one agent type cannot exhaust resources for others.

    Implementors: Bulkhead (bulkhead.py)
    """

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute `func` within the bulkhead's resource pool.

        Raises:
            BulkheadFull: if the pool is at capacity and the queue is full.
        """
        ...


@runtime_checkable
class TokenTrackerProtocol(Protocol):
    """
    Contract for a token usage tracker.

    Tracks input/output token consumption per agent and per workflow.
    Raises TokenBudgetExceeded before a call that would breach the budget.

    Implementors: TokenManager (token_manager.py)
    """

    def check_budget(self, agent_name: str, estimated_tokens: int = 0) -> None:
        """
        Assert that the budget has not been exceeded.

        Raises:
            TokenBudgetExceeded: if the budget would be breached.
        """
        ...

    def record_usage(self, agent_name: str, tokens_in: int, tokens_out: int) -> dict:
        """Record actual token usage after a completed call."""
        ...

    def get_remaining_budget(self) -> int:
        """Return the remaining token budget for this workflow."""
        ...
