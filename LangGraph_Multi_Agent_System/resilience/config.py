"""
Resilience Configuration
==========================
A single, typed configuration object for the entire resilience layer.

WHY A DATACLASS INSTEAD OF SCATTERED SETTINGS LOOKUPS?
    The original code called `settings.circuit_breaker_fail_max`,
    `settings.rate_limit_calls_per_minute`, etc., in several different
    files. This has two problems:

    1. Coupling: every resilience class depends on the global `settings`
       object. You can't instantiate a CircuitBreaker in a unit test
       without mocking the entire settings module.

    2. Discovery: there is no single place where a new developer can
       see all the tunable knobs. They have to grep the codebase.

    A `ResilienceConfig` dataclass solves both. It:
    - Is plain data (no I/O, no global state) — trivially testable.
    - Has sensible defaults, so tests and quick scripts need zero config.
    - Can be built from environment variables, a YAML file, or Pydantic
      settings by a single factory function (`from_settings`).

TUNING GUIDANCE (what each knob does and why the default exists):
    See individual field docstrings below.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """
    Configuration for one circuit breaker instance.

    fail_max:
        How many consecutive failures before the breaker opens.
        Default 5 — low enough to catch real outages quickly, high
        enough to tolerate transient flakes.

    reset_timeout:
        Seconds the breaker stays OPEN before moving to HALF-OPEN to
        test the service. Default 60s — one minute is usually enough
        for an LLM provider to recover from a rate-limit window.
    """

    fail_max: int = 5
    reset_timeout: int = 60


@dataclass(frozen=True)
class RateLimiterConfig:
    """
    Configuration for the rate limiter.

    max_calls:
        Maximum API calls allowed per `period` seconds.
        Default 60 (1 per second average over a minute) — matches
        typical LLM provider free-tier limits.

    period:
        Time window in seconds. Default 60 (one minute).

    block:
        If True (default), sleep until a slot opens.
        If False, raise RateLimitExhausted immediately — useful for
        async systems where blocking is unacceptable.
    """

    max_calls: int = 60
    period: float = 60.0
    block: bool = True


@dataclass(frozen=True)
class RetryConfig:
    """
    Configuration for retry-with-backoff.

    max_retries:
        Retry attempts after the initial failure. Default 3 means
        up to 4 total attempts. Don't go above 5 — exponential
        backoff means attempt 5 could wait 16+ seconds.

    initial_wait:
        Base wait time in seconds. Default 1.0s.

    max_wait:
        Cap on backoff to prevent very long waits. Default 30s.

    jitter:
        Random seconds added to each wait. Prevents the "thundering
        herd" problem where all agents retry simultaneously and slam
        the API at the same moment.
    """

    max_retries: int = 3
    initial_wait: float = 1.0
    max_wait: float = 30.0
    jitter: float = 1.0


@dataclass(frozen=True)
class TimeoutConfig:
    """
    Configuration for deadline enforcement.

    default_timeout:
        Seconds to wait for a single LLM call. Default 30s.
        Most LLM providers time out server-side at 60s; using 30s
        leaves room to retry before the user-facing SLO is breached.
    """

    default_timeout: float = 30.0


@dataclass(frozen=True)
class BulkheadConfig:
    """
    Configuration for the bulkhead resource pool.

    max_concurrent:
        Max simultaneous in-flight LLM calls per pool. Default 10.
        A single OpenAI org typically has 500 RPM; with 10 agents each
        making calls at 5s average latency, you're at ~120 RPM — safe.

    max_queue:
        Calls waiting for a slot before BulkheadFull is raised.
        Default 20. Set to 0 to disable queuing (shed load immediately).
    """

    max_concurrent: int = 10
    max_queue: int = 20


@dataclass(frozen=True)
class TokenBudgetConfig:
    """
    Configuration for token budget enforcement.

    max_tokens_per_workflow:
        Hard cap on total tokens consumed across all agents in one
        workflow execution. Default 32,000 (~$0.03–0.10 depending
        on model). Adjust based on your acceptable cost ceiling.

    max_tokens_per_agent:
        Per-agent call cap. Prevents one runaway agent from consuming
        the entire workflow budget. Default 4,096.
    """

    max_tokens_per_workflow: int = 32_000
    max_tokens_per_agent: int = 4_096


@dataclass(frozen=True)
class ResilienceConfig:
    """
    Top-level configuration container for the entire resilience layer.

    Compose and pass a single `ResilienceConfig` to `ResilientCaller`.
    Each field has safe defaults — you only need to override what differs
    from the standard settings.

    Example — default config (good for development):
        config = ResilienceConfig()

    Example — conservative production config for a medical MAS:
        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(fail_max=3, reset_timeout=120),
            rate_limiter=RateLimiterConfig(max_calls=30, block=True),
            retry=RetryConfig(max_retries=2, max_wait=10.0),
            timeout=TimeoutConfig(default_timeout=20.0),
            token_budget=TokenBudgetConfig(max_tokens_per_workflow=16_000),
        )

    Example — loading from environment / Pydantic settings:
        config = ResilienceConfig.from_settings(settings)
    """

    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    rate_limiter: RateLimiterConfig = field(default_factory=RateLimiterConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)
    bulkhead: BulkheadConfig = field(default_factory=BulkheadConfig)
    token_budget: TokenBudgetConfig = field(default_factory=TokenBudgetConfig)

    @classmethod
    def from_settings(cls, settings: object) -> "ResilienceConfig":
        """
        Build a ResilienceConfig from an existing Pydantic/Django settings object.

        Reads well-known attribute names with graceful fallback to defaults
        if an attribute is absent. This means you can migrate incrementally —
        add settings attributes one by one without breaking anything.

        Args:
            settings: Any object with optional attributes matching the
                      field names listed below.

        Returns:
            ResilienceConfig populated from the settings object.
        """

        def _get(attr: str, default):
            return getattr(settings, attr, default)

        return cls(
            circuit_breaker=CircuitBreakerConfig(
                fail_max=_get("circuit_breaker_fail_max", 5),
                reset_timeout=_get("circuit_breaker_reset_timeout", 60),
            ),
            rate_limiter=RateLimiterConfig(
                max_calls=_get("rate_limit_calls_per_minute", 60),
                period=_get("rate_limit_period", 60.0),
                block=_get("rate_limit_block", True),
            ),
            retry=RetryConfig(
                max_retries=_get("retry_max_attempts", 3),
                initial_wait=_get("retry_initial_wait", 1.0),
                max_wait=_get("retry_max_wait", 30.0),
            ),
            timeout=TimeoutConfig(
                default_timeout=_get("llm_call_timeout", 30.0),
            ),
            bulkhead=BulkheadConfig(
                max_concurrent=_get("bulkhead_max_concurrent", 10),
                max_queue=_get("bulkhead_max_queue", 20),
            ),
            token_budget=TokenBudgetConfig(
                max_tokens_per_workflow=_get("max_tokens_per_workflow", 32_000),
                max_tokens_per_agent=_get("max_tokens_per_agent_call", 4_096),
            ),
        )
