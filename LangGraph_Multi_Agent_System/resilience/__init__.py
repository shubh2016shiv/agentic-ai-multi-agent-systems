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
  WHERE THIS FITS IN THE MAS ARCHITECTURE
════════════════════════════════════════

Resilience is NOT a standalone pattern or a separate layer in the graph.
It is EMBEDDED INSIDE the orchestration component module.

    ┌──────────────────────────────────────────────────────────────┐
    │                    MAS Architecture                          │
    │                                                              │
    │  orchestration/orchestrator.py                              │
    │  ├── _ORCHESTRATION_LLM_BREAKER  (shared circuit breaker)   │
    │  └── _ORCHESTRATION_CALLER       (ResilientCaller façade)   │
    │             │                                               │
    │             │  BaseOrchestrator.invoke_specialist()         │
    │             │  BaseOrchestrator.invoke_synthesizer()        │
    │             ▼                                               │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │  resilience/ — 6-layer stack (outer → inner)        │   │
    │  │  ┌────────────────────────────────────────────────┐  │   │
    │  │  │  [1] Token Budget   (fail fast if over budget) │  │   │
    │  │  │  [2] Bulkhead       (SKIPPED for linear flows) │  │   │
    │  │  │  [3] Rate Limiter   (ENABLED, smooths bursts)  │  │   │
    │  │  │  [4] Circuit Breaker (fail-fast, shared)       │  │   │
    │  │  │  [5] Retry Handler  (transient errors only)    │  │   │
    │  │  │  [6] Timeout Guard  (30s deadline, innermost)  │  │   │
    │  │  └────────────────────────────────────────────────┘  │   │
    │  └──────────────────────────────────────────────────────┘   │
    │             │                                               │
    │             ▼                                               │
    │  llm.invoke(prompt)  ← the ACTUAL LLM API call             │
    │                                                              │
    │  Orchestration patterns that trigger this stack:            │
    │  ┌────────────────────────────────────────────────────┐     │
    │  │ STAGE 1: supervisor_orchestration/agents.py        │     │
    │  │   pulmonology_worker_node, cardiology_worker_node, │     │
    │  │   nephrology_worker_node, report_synthesis_node    │     │
    │  │ STAGE 2: peer_to_peer_orchestration/agents.py      │     │
    │  │   pulmonology_peer_node, cardiology_peer_node,     │     │
    │  │   nephrology_peer_node, synthesis_node             │     │
    │  │ STAGE 3: dynamic_router_orchestration/agents.py    │     │
    │  │   pulmonology_specialist_node, cardiology_         │     │
    │  │   specialist_node, nephrology_specialist_node,     │     │
    │  │   router_report_node                               │     │
    │  │ STAGE 4: graph_of_subgraphs_orchestration/         │     │
    │  │   All 9 subgraph nodes + synthesis_node            │     │
    │  │   (via _ORCHESTRATION_CALLER.call() directly)      │     │
    │  │ STAGE 5: hybrid_orchestration/agents.py            │     │
    │  │   cardiopulmonary_pulmonology_node,                │     │
    │  │   cardiopulmonary_cardiology_node,                 │     │
    │  │   renal_specialist_node, hybrid_synthesis_node     │     │
    │  └────────────────────────────────────────────────────┘     │
    │                                                              │
    │  NOT protected (direct llm.invoke — no resilience):         │
    │    supervisor_decide_node, input_classifier_node,            │
    │    hybrid_supervisor_node (routing/classification nodes)     │
    └──────────────────────────────────────────────────────────────┘

CONNECTION: orchestration/orchestrator.py — the primary integration point.
    BaseOrchestrator creates _ORCHESTRATION_LLM_BREAKER (CircuitBreakerRegistry)
    and _ORCHESTRATION_CALLER (ResilientCaller) once, shared by all 5 patterns.

CONNECTION: resilience/resilient_caller.py — the FAÇADE entry point.
    _ORCHESTRATION_CALLER.call(llm.invoke, prompt) is how orchestration nodes
    invoke the full resilience stack without knowing the internal composition.

CONNECTION: resilience/circuit_breaker.py — the shared breaker instance.
    "orchestration_llm_api" is the registry key. All 5 patterns share ONE
    breaker so a failing LLM API stops ALL patterns immediately.

════════════════════════════════════════
  LEARNING SEQUENCE FOR RESILIENCE
════════════════════════════════════════

  Step 1: Read resilience/__init__.py (this file) — overview
  Step 2: Read resilience/config.py — configuration parameters
  Step 3: Read individual pattern files (circuit_breaker.py, retry_handler.py,
          timeout_guard.py, rate_limiter.py, token_manager.py, bulkhead.py)
  Step 4: Read resilience/resilient_caller.py — how they compose
  Step 5: Read orchestration/orchestrator.py — how orchestration uses them
  Step 6: Run scripts/orchestration/supervisor_orchestration/runner.py
          and trace how invoke_specialist() flows through the stack

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
