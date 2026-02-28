"""
Rate Limiter — Sliding-Window Token Bucket
============================================
Controls the frequency of LLM API calls to prevent quota exhaustion.

════════════════════════════════════════
  *** WHEN AND WHERE THE RATE LIMITER IS TRIGGERED ***
════════════════════════════════════════

TRIGGER — Rate limiter acquires a slot before each call:
    Condition: rate_limiter.acquire() is invoked (unless skip_rate_limiter=True).
    Effect (blocking mode, default): If quota exhausted, sleep until slot opens.
    Effect (non-blocking mode): If quota exhausted, raise RateLimitExhausted.

WHERE IT IS INVOKED — SOURCE ↔ DESTINATION MAPPING:
    *** SOURCE: rate_limiter.acquire() in resilient_caller._call_inner()
    *** DESTINATION: Same MAS nodes that trigger the circuit breaker (when
    rate limiter is enabled via skip_rate_limiter=False). ***

    Currently ENABLED (skip_rate_limiter=False) in orchestrator base:
        supervisor_orchestration, peer_to_peer, dynamic_router, hybrid —
        all specialist and synthesis nodes.

    Call chain (SOURCE): orchestrator.invoke_specialist/invoke_synthesizer
        → ResilientCaller.call(llm.invoke, ..., skip_rate_limiter=False)
        → _call_inner() → rate_limiter.acquire()

    DESTINATION: Same as circuit_breaker — see resilience/circuit_breaker.py
    module docstring for the full table.

════════════════════════════════════════
  HOW THE SLIDING WINDOW WORKS
════════════════════════════════════════

  Time ─────────────────────────────────────────────────►
          │← ─ ─ ─ ─ ─  period (60s)  ─ ─ ─ ─ ─ →│
          ┌───┬───┬───┬───┬───┐
  calls:  │t1 │t2 │t3 │...│tn │  ◄── stored timestamps
          └───┴───┴───┴───┴───┘
                              ▲
                         current call?

  1. Drop all timestamps older than `now - period`  (the window slides)
  2. If `len(remaining) < max_calls`, admit the call and record timestamp
  3. Otherwise, the *oldest* timestamp tells us when the next slot opens:
       sleep_time = period - (now - oldest_timestamp)

  This gives a fair "rolling quota" — not a hard-reset counter that
  resets at the top of every minute (which allows a burst at :59
  followed by another burst at :01).

════════════════════════════════════════
  WHY THIS MATTERS IN A MAS
════════════════════════════════════════
Consider a voting pattern: 5 specialist agents each call the LLM
simultaneously to produce a recommendation. Without rate limiting,
all 5 calls hit the API at once — a burst of 5 RPM in one second.

At 20 RPM (typical tier), a burst of 5 simultaneous calls can trigger
a 429 in a pipeline that technically stays *under* the quota overall.
The rate limiter smooths the burst, spacing calls automatically.

════════════════════════════════════════
  THREAD-SAFETY
════════════════════════════════════════
The `_lock` (threading.Lock) ensures that two agents running in
separate threads cannot simultaneously check-and-record a timestamp,
which would allow both to slip through when only one slot remained.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from .config import RateLimiterConfig
from .exceptions import RateLimitExhausted

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Thread-safe sliding-window rate limiter.

    Controls how many calls can be made to a service within a time window.
    Can operate in two modes depending on `config.block`:

    Blocking mode (default, block=True):
        - If the limit is reached, `acquire()` sleeps until a slot opens.
        - Use when you want simple throttling with no extra error handling.

    Non-blocking mode (block=False):
        - If the limit is reached, `acquire()` raises RateLimitExhausted.
        - Use in async systems or when you want the caller to decide
          whether to wait, retry, or shed load.

    Args:
        name:   Identifier used in log messages (e.g., "openai_api").
        config: RateLimiterConfig with max_calls, period, and block settings.

    Example — blocking (wraps a function):
        limiter = RateLimiter("openai", RateLimiterConfig(max_calls=10))
        limiter.acquire()
        result = llm.invoke(prompt)

    Example — as a decorator:
        limiter = RateLimiter("openai", RateLimiterConfig(max_calls=10))

        @limiter.as_decorator
        def call_llm(prompt):
            return llm.invoke(prompt)
    """

    def __init__(self, name: str, config: RateLimiterConfig) -> None:
        self._name = name
        self._max_calls = config.max_calls
        self._period = config.period
        self._block = config.block
        self._lock = threading.Lock()
        self._call_times: list[float] = []

    def acquire(self) -> None:
        """
        Acquire permission to make one call.

        *** TRIGGER: This is the actual entry point where rate limiting occurs.
        Called from resilient_caller._call_inner() (SOURCE) when skip_rate_limiter=False.
        DESTINATION: orchestration specialist/synthesis nodes (same as circuit breaker). ***

        Blocking mode: sleeps until a slot is available.
        Non-blocking mode: raises RateLimitExhausted if no slot is available.

        This method is the single entry point — `call()` and the decorator
        both use it. Always call this before making your LLM/tool call.
        """
        with self._lock:
            now = time.monotonic()

            # Step 1: Prune expired timestamps (slide the window)
            self._call_times = [t for t in self._call_times if now - t < self._period]

            # Step 2: Check if under the limit
            if len(self._call_times) < self._max_calls:
                self._call_times.append(time.monotonic())
                return

            # Step 3: Limit reached — block or raise
            oldest = self._call_times[0]
            sleep_for = self._period - (now - oldest)

            if not self._block:
                raise RateLimitExhausted(
                    f"Rate limiter '{self._name}' quota exhausted "
                    f"({self._max_calls} calls per {self._period}s).",
                    details={
                        "limiter_name": self._name,
                        "limit": self._max_calls,
                        "period": self._period,
                        "retry_after": round(sleep_for, 2),
                    },
                )

            if sleep_for > 0:
                logger.warning(
                    "Rate limit reached — throttling caller",
                    extra={
                        "limiter": self._name,
                        "sleep_secs": round(sleep_for, 2),
                        "limit": self._max_calls,
                        "period": self._period,
                    },
                )
                time.sleep(sleep_for)

            # After sleeping, re-prune and record
            now = time.monotonic()
            self._call_times = [t for t in self._call_times if now - t < self._period]
            self._call_times.append(time.monotonic())

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Acquire a rate-limit slot, then execute `func`.

        Convenience wrapper so you don't need to call `acquire()` manually.

        Args:
            func:     The callable to rate-limit (e.g., llm.invoke).
            *args:    Positional args forwarded to `func`.
            **kwargs: Keyword args forwarded to `func`.

        Returns:
            Whatever `func` returns.
        """
        self.acquire()
        return func(*args, **kwargs)

    def as_decorator(self, func: Callable) -> Callable:
        """
        Return a version of `func` that acquires a rate-limit slot before each call.

        Usage:
            @limiter.as_decorator
            def call_llm(prompt):
                return llm.invoke(prompt)
        """
        import functools

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.acquire()
            return func(*args, **kwargs)

        return wrapper

    @property
    def name(self) -> str:
        return self._name
