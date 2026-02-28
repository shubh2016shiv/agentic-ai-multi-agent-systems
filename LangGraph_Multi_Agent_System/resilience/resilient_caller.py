"""
ResilientCaller — Unified Façade
===================================
Composes all resilience patterns into a single, clean entry point.

════════════════════════════════════════
  DESIGN PATTERN: FAÇADE
════════════════════════════════════════
Without this façade, every LangGraph node would need to:
  1. Import 5+ resilience classes.
  2. Know the correct ORDER to apply them.
  3. Manually wire them together.
  4. Handle errors from each one.

This is repetitive, error-prone, and violates DRY.

The Façade pattern provides ONE class with ONE method (`call`).
Callers know nothing about the internal composition.

════════════════════════════════════════
  CALL EXECUTION ORDER
════════════════════════════════════════
Patterns are applied in this order (inner-most = closest to the real call):

    ┌─ Token Budget Check ──────────────────────────────────────────┐
    │  ┌─ Bulkhead (concurrency pool) ───────────────────────────┐  │
    │  │  ┌─ Rate Limiter (quota throttle) ──────────────────┐   │  │
    │  │  │  ┌─ Circuit Breaker (fail-fast) ──────────────┐  │   │  │
    │  │  │  │  ┌─ Retry (backoff on transient errors) ─┐ │  │   │  │
    │  │  │  │  │  ┌─ Timeout (deadline enforcement) ─┐ │ │  │   │  │
    │  │  │  │  │  │                                  │ │ │  │   │  │
    │  │  │  │  │  │       LLM / Tool Call            │ │ │  │   │  │
    │  │  │  │  │  └──────────────────────────────────┘ │ │  │   │  │
    │  │  │  │  └───────────────────────────────────────┘ │  │   │  │
    │  │  │  └────────────────────────────────────────────┘  │   │  │
    │  │  └──────────────────────────────────────────────────┘   │  │
    │  └─────────────────────────────────────────────────────────┘  │
    └───────────────────────────────────────────────────────────────┘
    After call: Record token usage

WHY THIS ORDER?
    - Token budget check is outermost: fail fast before acquiring ANY resource
    - Bulkhead is next: don't rate-limit or wait in line if the pool is full
    - Rate limiter throttles before hitting the circuit breaker
    - Circuit breaker is inside rate limiter: if the service is known-down,
      don't consume a rate-limit slot
    - Retry wraps the actual call so failed attempts count toward the breaker
    - Timeout is innermost: the deadline applies to each individual attempt

════════════════════════════════════════
  SELECTIVE PATTERN APPLICATION
════════════════════════════════════════
Not every pattern is needed for every call. Use the `skip_*` flags
to opt out of individual patterns for specific scenarios:

    # Quick health-check: no retry, no timeout
    caller.call(
        health_check_func,
        skip_retry=True,
        skip_timeout=True,
    )

    # Batch job: no bulkhead (it has its own dedicated queue)
    caller.call(
        batch_summarise_func,
        prompt,
        skip_bulkhead=True,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .bulkhead import Bulkhead  # BULKHEAD SOURCE: used in __init__ and call() layer 2
from .circuit_breaker import CircuitBreaker
from .config import ResilienceConfig
from .exceptions import ResilienceError
from .rate_limiter import RateLimiter
from .retry_handler import RetryHandler  # RETRY HANDLER SOURCE: used below in __init__ and _call_inner
from .timeout_guard import TimeoutGuard
from .token_manager import TokenManager

logger = logging.getLogger(__name__)


class ResilientCaller:
    """
    Composes all resilience patterns into a single reusable caller.

    Instantiate once per agent (or once per LLM service) and reuse
    across all calls. All state (circuit-breaker failure count, rate-limiter
    window, bulkhead active count) is preserved across calls.

    Args:
        config:          Top-level ResilienceConfig (uses defaults if not provided).
        agent_name:      Used in logs and exception details.
        circuit_breaker: Optional pre-configured CircuitBreaker instance.
                         Useful for sharing one breaker across multiple callers
                         that call the same downstream service.
        token_manager:   Optional shared TokenManager for workflow-level budget
                         tracking across multiple agents.

    Example — minimal setup (all defaults):
        caller = ResilientCaller(agent_name="triage_agent")
        result = caller.call(llm.invoke, prompt)

    Example — production setup with shared circuit breaker and token budget:
        from resilience import (
            ResilientCaller, ResilienceConfig,
            CircuitBreakerRegistry, TokenManager,
            CircuitBreakerConfig, TokenBudgetConfig,
        )

        shared_breaker = CircuitBreakerRegistry.get_or_create(
            "openai_api",
            CircuitBreakerConfig(fail_max=3),
        )
        workflow_budget = TokenManager(TokenBudgetConfig(max_tokens_per_workflow=16_000))

        caller = ResilientCaller(
            config=ResilienceConfig(),
            agent_name="pharmacology_agent",
            circuit_breaker=shared_breaker,
            token_manager=workflow_budget,
        )

        result = caller.call(llm.invoke, prompt)

    Example — LangGraph node integration:
        def pharmacology_node(state: GraphState) -> dict:
            caller = state["resilient_caller"]  # injected via state
            try:
                result = caller.call(llm.invoke, state["prompt"])
                return {"response": result}
            except CircuitBreakerOpen:
                return {"response": CACHED_FALLBACK, "used_fallback": True}
            except TokenBudgetExceeded:
                return {"response": state.get("partial_response"), "budget_exceeded": True}
    """

    def __init__(
        self,
        config: ResilienceConfig | None = None,
        agent_name: str = "unknown_agent",
        circuit_breaker: CircuitBreaker | None = None,
        token_manager: TokenManager | None = None,
    ) -> None:
        cfg = config or ResilienceConfig()
        self._agent_name = agent_name

        # Construct each component (or use the provided shared instances)
        self._circuit_breaker = circuit_breaker or CircuitBreaker(
            name=agent_name,
            config=cfg.circuit_breaker,
        )
        self._rate_limiter = RateLimiter(name=agent_name, config=cfg.rate_limiter)
        # RETRY HANDLER: One instance per ResilientCaller. Wraps timeout layer; retries only
        # on TRANSIENT_EXCEPTIONS (ConnectionError, TimeoutError, RateLimitError, etc.).
        # HOW: call_with_retry(protected_call) in _call_inner (layer 5). DESTINATION: every
        # node that uses this caller (orchestrator invoke_specialist/invoke_synthesizer → all
        # specialist and synthesis nodes in supervisor, peer_to_peer, dynamic_router, hybrid).
        self._retry_handler = RetryHandler(config=cfg.retry)
        self._timeout_guard = TimeoutGuard(config=cfg.timeout, agent_name=agent_name)
        # BULKHEAD: One instance per ResilientCaller. Applied in call() as layer 2 (after
        # token check). When skip_bulkhead=False, call runs inside self._bulkhead.acquire().
        # In MAS orchestration, orchestrator passes skip_bulkhead=True — bulkhead not used.
        self._bulkhead = Bulkhead(name=agent_name, config=cfg.bulkhead)
        self._token_manager = token_manager  # Optional — None means no budget tracking

    def call(
        self,
        func: Callable,
        *args: Any,
        timeout: float | None = None,
        estimated_tokens: int = 0,
        skip_circuit_breaker: bool = False,
        skip_rate_limiter: bool = False,
        skip_retry: bool = False,
        skip_timeout: bool = False,
        skip_bulkhead: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Execute `func` through the full resilience stack.

        The patterns are applied in a fixed order (see module docstring).
        Each `skip_*` flag removes one layer for that specific call.

        Args:
            func:                  The callable to execute (e.g., llm.invoke).
            *args:                 Positional args forwarded to `func`.
            timeout:               Per-call timeout override in seconds.
            estimated_tokens:      Pre-call token estimate for budget checking.
            skip_circuit_breaker:  Skip circuit breaker protection.
            skip_rate_limiter:     Skip rate limiting.
            skip_retry:            Skip automatic retry on transient errors.
            skip_timeout:          Skip deadline enforcement.
            skip_bulkhead:         Skip concurrency pool isolation.
            **kwargs:              Keyword args forwarded to `func`.

        Returns:
            The return value of `func`.

        Raises:
            CircuitBreakerOpen:   Service is known-unhealthy.
            RateLimitExhausted:   Quota exceeded (only in non-blocking mode).
            TimeoutExceeded:      Deadline was breached.
            TokenBudgetExceeded:  Token budget would be exceeded.
            BulkheadFull:         Concurrency pool is at capacity.
            Any exception raised by `func` after retries are exhausted.
        """
        # ── Layer 1: Token budget pre-check ─────────────────────────────────
        if self._token_manager and estimated_tokens:
            self._token_manager.check_budget(self._agent_name, estimated_tokens)

        # ── Layer 2: Bulkhead ────────────────────────────────────────────────
        # BULKHEAD USAGE: If skip_bulkhead=True (default in orchestrator), we skip the
        # pool and call _call_inner directly. If False, we acquire a slot from
        # self._bulkhead and run _call_inner inside it; BulkheadFull raised when
        # pool+queue are full. MAS: scripts/orchestration/_base/orchestrator.py
        # passes skip_bulkhead=True for invoke_specialist and invoke_synthesizer,
        # so no orchestration node currently uses the bulkhead.
        if skip_bulkhead:
            result = self._call_inner(func, *args, timeout=timeout,
                                      skip_circuit_breaker=skip_circuit_breaker,
                                      skip_rate_limiter=skip_rate_limiter,
                                      skip_retry=skip_retry,
                                      skip_timeout=skip_timeout, **kwargs)
        else:
            with self._bulkhead.acquire():
                result = self._call_inner(func, *args, timeout=timeout,
                                          skip_circuit_breaker=skip_circuit_breaker,
                                          skip_rate_limiter=skip_rate_limiter,
                                          skip_retry=skip_retry,
                                          skip_timeout=skip_timeout, **kwargs)

        return result

    def _call_inner(
        self,
        func: Callable,
        *args: Any,
        timeout: float | None,
        skip_circuit_breaker: bool,
        skip_rate_limiter: bool,
        skip_retry: bool,
        skip_timeout: bool,
        **kwargs: Any,
    ) -> Any:
        """
        Inner execution layers: rate limiter → circuit breaker → retry → timeout.

        Split from `call()` to keep bulkhead context-manager logic clean.
        """
        # ── Layer 3: Rate limiter ────────────────────────────────────────────
        if not skip_rate_limiter:
            self._rate_limiter.acquire()

        # ── Build the innermost callable ────────────────────────────────────
        # The actual call, optionally wrapped in a timeout.
        def _timed_call() -> Any:
            if skip_timeout:
                return func(*args, **kwargs)
            return self._timeout_guard.call_with_timeout(func, *args, timeout=timeout, **kwargs)

        # ── Layer 4: Circuit breaker ─────────────────────────────────────────
        if skip_circuit_breaker:
            protected_call = _timed_call
        else:
            def protected_call() -> Any:  # type: ignore[no-redef]
                return self._circuit_breaker.call(_timed_call)

        # ── Layer 5: Retry ───────────────────────────────────────────────────
        # RETRY HANDLER USAGE: Wraps the circuit-breaker–protected call. If protected_call()
        # raises a transient exception (ConnectionError, TimeoutError, RateLimitError, etc.),
        # RetryHandler retries with exponential backoff + jitter up to max_retries. After
        # exhaustion or on permanent errors, the exception propagates. This is the ONLY place
        # in MAS where retry_handler is invoked — all orchestration LLM calls flow through here.
        if skip_retry:
            return protected_call()
        else:
            return self._retry_handler.call_with_retry(protected_call)

    # ── Convenience accessors ────────────────────────────────────────────────

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Direct access to the circuit breaker (e.g., to check `is_open`)."""
        return self._circuit_breaker

    @property
    def token_manager(self) -> TokenManager | None:
        """Direct access to the token manager (e.g., to call `record_usage`)."""
        return self._token_manager
