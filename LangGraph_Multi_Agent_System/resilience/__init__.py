"""
resilience — Enterprise Resilience Layer for LangGraph MAS
============================================================
A composable set of patterns that protect LLM-backed multi-agent
systems from cascading failures, quota exhaustion, and runaway costs.

WHERE TO USE WHAT: See RESILIENCE_MAS_MAPPING.md for pattern-to-MAS
mapping, rationale, and configuration. Orchestration base (orchestrator.py)
is the primary integration point; handoff/MAS_architectures may adopt
the same pattern or a future shared helper.

════════════════════════════════════════
  QUICK START — TYPICAL USAGE
════════════════════════════════════════

1. The simplest way — let ResilientCaller handle everything:

    from resilience import ResilientCaller

    caller = ResilientCaller(agent_name="triage_agent")
    result = caller.call(llm.invoke, prompt)

2. Production setup — custom config + shared circuit breaker:

    from resilience import (
        ResilientCaller,
        ResilienceConfig, CircuitBreakerConfig, TokenBudgetConfig,
        CircuitBreakerRegistry,
        TokenManager,
    )

    # One circuit breaker shared by all agents hitting the same API
    breaker = CircuitBreakerRegistry.get_or_create(
        "openai_api",
        CircuitBreakerConfig(fail_max=3, reset_timeout=120),
    )

    # One token manager per workflow execution
    budget = TokenManager(TokenBudgetConfig(max_tokens_per_workflow=16_000))

    caller = ResilientCaller(
        config=ResilienceConfig(),
        agent_name="pharmacology_agent",
        circuit_breaker=breaker,
        token_manager=budget,
    )

    result = caller.call(llm.invoke, prompt)

3. Multi-provider failover:

    from resilience import FallbackChain, Provider

    chain = FallbackChain([
        Provider("gpt4o",   openai_llm.invoke,    weight=1),
        Provider("claude",  anthropic_llm.invoke,  weight=2),
        Provider("gemini",  gemini_llm.invoke,     weight=3),
    ])

    result = chain.call(prompt)

════════════════════════════════════════
  PATTERNS PROVIDED
════════════════════════════════════════
Pattern              Class                 Purpose
─────────────────────────────────────────────────────────────────────
Circuit Breaker      CircuitBreaker        Fail fast on known-down service
Rate Limiter         RateLimiter           Prevent API quota exhaustion
Retry + Backoff      RetryHandler          Auto-recover from transient errors
Timeout Guard        TimeoutGuard          Hard deadline per call
Fallback Chain       FallbackChain         Multi-provider failover
Bulkhead             Bulkhead              Isolate agent resource pools
Token Budget         TokenManager          Cap cost per agent/workflow
Façade               ResilientCaller       All patterns in one call

════════════════════════════════════════
  EXCEPTION TYPES
════════════════════════════════════════
All exceptions inherit from ResilienceError and carry a `details` dict
with structured context for logging and orchestration decisions.

Exception              Meaning                  Recommended action
──────────────────────────────────────────────────────────────────────────
CircuitBreakerOpen     Service is known-down    Return cached fallback
RateLimitExhausted     Quota hit (non-blocking) Retry after `retry_after` secs
TimeoutExceeded        Deadline passed          Log, return partial result
AllFallbacksFailed     All providers down       Return static error to user
BulkheadFull           Pool at capacity         Retry with short jitter
TokenBudgetExceeded    Budget ceiling reached   Return partial/cached result
"""

from .bulkhead import Bulkhead
from .circuit_breaker import CircuitBreaker, CircuitBreakerRegistry
from .config import (
    BulkheadConfig,
    CircuitBreakerConfig,
    RateLimiterConfig,
    ResilienceConfig,
    RetryConfig,
    TimeoutConfig,
    TokenBudgetConfig,
)
from .exceptions import (
    AllFallbacksFailed,
    BulkheadFull,
    CircuitBreakerOpen,
    RateLimitExhausted,
    ResilienceError,
    TimeoutExceeded,
    TokenBudgetExceeded,
)
from .fallback_chain import FallbackChain, Provider
from .protocols import (
    BulkheadProtocol,
    CircuitBreakerProtocol,
    FallbackChainProtocol,
    RateLimiterProtocol,
    RetryHandlerProtocol,
    TimeoutGuardProtocol,
    TokenTrackerProtocol,
)
from .rate_limiter import RateLimiter
from .resilient_caller import ResilientCaller
from .retry_handler import RetryHandler
from .timeout_guard import TimeoutGuard
from .token_manager import TokenCounter, TokenManager

__all__ = [
    # ── Façade (start here) ──────────────────────────────────────
    "ResilientCaller",
    # ── Patterns ─────────────────────────────────────────────────
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "RateLimiter",
    "RetryHandler",
    "TimeoutGuard",
    "FallbackChain",
    "Provider",
    "Bulkhead",
    "TokenManager",
    "TokenCounter",
    # ── Configuration ─────────────────────────────────────────────
    "ResilienceConfig",
    "CircuitBreakerConfig",
    "RateLimiterConfig",
    "RetryConfig",
    "TimeoutConfig",
    "BulkheadConfig",
    "TokenBudgetConfig",
    # ── Exceptions ────────────────────────────────────────────────
    "ResilienceError",
    "CircuitBreakerOpen",
    "RateLimitExhausted",
    "TimeoutExceeded",
    "AllFallbacksFailed",
    "BulkheadFull",
    "TokenBudgetExceeded",
    # ── Protocols (for type-safe dependency injection) ────────────
    "CircuitBreakerProtocol",
    "RateLimiterProtocol",
    "RetryHandlerProtocol",
    "TimeoutGuardProtocol",
    "FallbackChainProtocol",
    "BulkheadProtocol",
    "TokenTrackerProtocol",
]
