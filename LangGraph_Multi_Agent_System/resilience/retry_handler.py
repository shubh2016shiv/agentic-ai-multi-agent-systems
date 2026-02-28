"""
Retry Handler — Exponential Backoff with Jitter
=================================================
Automatically retries transient LLM/tool call failures using exponential
backoff and random jitter.

════════════════════════════════════════════════════════════════════════════════
  *** WHERE AND HOW RETRY HANDLER IS USED IN MAS ***
════════════════════════════════════════════════════════════════════════════════

**SINGLE USAGE SITE (SOURCE)**:
    resilience/resilient_caller.py
    - Line 81:  from .retry_handler import RetryHandler
    - Line 159: self._retry_handler = RetryHandler(config=cfg.retry)  # one per ResilientCaller
    - Line 265: return self._retry_handler.call_with_retry(protected_call)  # layer 5 in call stack

**HOW IT IS USED**:
    1. ResilientCaller.__init__() builds one RetryHandler per caller from config.retry.
    2. On each call(), _call_inner() builds a protected_call (circuit breaker wraps timeout).
    3. If skip_retry is False, the actual execution is:
           self._retry_handler.call_with_retry(protected_call)
       So the inner call (timeout → circuit breaker → LLM) is wrapped in retry.
    4. RetryHandler retries only on TRANSIENT_EXCEPTIONS (ConnectionError, TimeoutError,
       RateLimitError, APIConnectionError, etc.). Permanent errors (auth, invalid request)
       are not retried and propagate immediately.

**WHERE IT PROTECTS (DESTINATION)**:
    RetryHandler is NOT called from any script directly. All MAS usage flows through
    ResilientCaller. Therefore the same DESTINATION nodes as circuit breaker / timeout:
    - scripts/orchestration/_base/orchestrator.py: invoke_specialist(), invoke_synthesizer()
    - supervisor_orchestration/agents.py: pulmonology_worker_node, cardiology_worker_node,
      nephrology_worker_node, report_synthesis_node
    - peer_to_peer_orchestration/agents.py: all peer nodes + report node
    - dynamic_router_orchestration/agents.py: specialist workers + report node
    - hybrid_orchestration/agents.py: cardiopulmonary_*, renal_specialist_node, hybrid_synthesis_node

**ORDER IN STACK**: Token check → Bulkhead → Rate limiter → Circuit breaker → **Retry** → Timeout → LLM

════════════════════════════════════════
  THE BACKOFF FORMULA
════════════════════════════════════════

    wait = min(initial_wait × 2^(attempt−1) + random(0, jitter), max_wait)

    With defaults (initial=1s, jitter=1s, max=30s):
      Attempt 1 (initial):  0s   (executed immediately)
      Attempt 2 (retry 1):  ~1–2s
      Attempt 3 (retry 2):  ~2–3s
      Attempt 4 (retry 3):  ~4–5s

════════════════════════════════════════
  WHY JITTER IS MANDATORY
════════════════════════════════════════
Without jitter, all agents that hit a rate limit at t=0 will retry
simultaneously at t=1, again at t=2, and so on — the "thundering herd."
Each retry wave hits the already-overloaded API in synchronized bursts.

Jitter desynchronizes retries: agent A waits 1.2s, agent B waits 1.8s,
agent C waits 1.4s — spreading the load across the recovery window.

════════════════════════════════════════
  CRITICAL: TRANSIENT VS PERMANENT ERRORS
════════════════════════════════════════
Only TRANSIENT errors should be retried. Retrying a permanent error
wastes quota, adds latency, and fills logs with noise.

  RETRY these (transient):
    - ConnectionError    → network blip, will self-heal
    - TimeoutError       → server was briefly overloaded
    - RateLimitError     → quota window, wait and try again
    - APIConnectionError → upstream connectivity issue

  DO NOT RETRY these (permanent):
    - AuthenticationError  → bad API key — retrying won't fix it
    - InvalidRequestError  → malformed prompt — retrying won't fix it
    - ContentPolicyError   → policy violation — retrying won't fix it

The `TRANSIENT_EXCEPTIONS` tuple is the single source of truth.
When adding a new LLM provider, add its transient exception types here.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .config import RetryConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Transient exception registry
#
# WHY CENTRALIZED HERE:
#   All retry logic in this codebase references this tuple. When you add a
#   new LLM provider (e.g., Gemini, Cohere), you add its transient exceptions
#   here — in ONE place — and every retry site picks them up automatically.
#   This is the Open/Closed Principle: open for extension (add to this list),
#   closed for modification (retry logic itself never changes).
# ──────────────────────────────────────────────────────────────────────────────

TRANSIENT_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# Extend with provider-specific exceptions if the SDKs are installed.
# The try/except ensures this module is importable even without the SDKs.
try:
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    TRANSIENT_EXCEPTIONS = TRANSIENT_EXCEPTIONS + (
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
    )
except ImportError:
    pass

try:
    from anthropic import APIConnectionError as AnthropicConnectionError
    from anthropic import APITimeoutError as AnthropicTimeoutError

    TRANSIENT_EXCEPTIONS = TRANSIENT_EXCEPTIONS + (
        AnthropicConnectionError,
        AnthropicTimeoutError,
    )
except ImportError:
    pass


class RetryHandler:
    """
    Wraps calls with automatic retry on transient failures.

    Responsibilities (SRP — this class does exactly one thing):
        Execute a callable, catch transient exceptions, wait with
        exponential backoff + jitter, and retry up to `config.max_retries`
        times. Re-raise after all retries are exhausted.

    Args:
        config: RetryConfig controlling backoff timing and attempt counts.
        retryable_exceptions: Tuple of exception types to retry on.
            Defaults to the module-level TRANSIENT_EXCEPTIONS.

    Example — wrap a one-off call:
        handler = RetryHandler(RetryConfig())
        result  = handler.call_with_retry(llm.invoke, prompt)

    Example — decorate a function:
        handler = RetryHandler(RetryConfig(max_retries=2))

        @handler.as_decorator
        def call_llm(prompt):
            return llm.invoke(prompt)
    """

    def __init__(
        self,
        config: RetryConfig,
        retryable_exceptions: tuple[type[Exception], ...] | None = None,
    ) -> None:
        self._config = config
        self._retryable = retryable_exceptions or TRANSIENT_EXCEPTIONS

        # Build the tenacity retry decorator once at construction time.
        # Reusing the same decorator instance is more efficient than
        # creating it on every call.
        self._retry_decorator = retry(
            stop=stop_after_attempt(config.max_retries + 1),  # +1: first attempt is not a "retry"
            wait=wait_exponential_jitter(
                initial=config.initial_wait,
                max=config.max_wait,
                jitter=config.jitter,
            ),
            retry=retry_if_exception_type(self._retryable),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.DEBUG),
            reraise=True,   # Re-raise the last exception — don't swallow it
        )

    def call_with_retry(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute `func(*args, **kwargs)` with retry on transient failures.

        *** MAS USAGE: This method is the only place RetryHandler is invoked in the
        multi-agent system. Called from resilience/resilient_caller.py _call_inner()
        as layer 5: self._retry_handler.call_with_retry(protected_call). ***

        Args:
            func:     Callable to execute (e.g., llm.invoke, tool.run).
            *args:    Positional args forwarded to `func`.
            **kwargs: Keyword args forwarded to `func`.

        Returns:
            Whatever `func` returns on success.

        Raises:
            The last transient exception if all retry attempts are exhausted.
            Any non-transient exception immediately (no retry attempted).
        """
        @self._retry_decorator
        def _execute() -> Any:
            return func(*args, **kwargs)

        return _execute()

    def as_decorator(self, func: Callable) -> Callable:
        """
        Wrap `func` so every call is automatically retried.

        Usage:
            handler = RetryHandler(RetryConfig())

            @handler.as_decorator
            def call_llm(prompt):
                return llm.invoke(prompt)
        """
        import functools

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.call_with_retry(func, *args, **kwargs)

        return wrapper
