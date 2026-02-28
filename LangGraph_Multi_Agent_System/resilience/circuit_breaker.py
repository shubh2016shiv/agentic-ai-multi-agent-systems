"""
Circuit Breaker Pattern
=========================
Protects downstream LLM/tool services from cascading failures by
monitoring call outcomes and "opening" when failures exceed a threshold.

════════════════════════════════════════
  *** WHEN AND WHERE THE CIRCUIT BREAKER IS TRIGGERED ***
════════════════════════════════════════

TRIGGER 1 — Breaker OPENS (CLOSED → OPEN):
    Condition: fail_max CONSECUTIVE failures are recorded.
    Each failure = one exception from the wrapped call (llm.invoke, etc.).
    Failures counted: API errors, timeouts, 429, connection errors, etc.
    Default: fail_max=5 (orchestrator uses 5).
    Effect: All subsequent calls will fail fast (see Trigger 2).

TRIGGER 2 — CircuitBreakerOpen RAISED (caller sees the trigger):
    Condition: A caller invokes circuit_breaker.call() while state is OPEN.
    Effect: CircuitBreakerOpen is raised IMMEDIATELY — NO call to the
            wrapped function. No LLM request is made. Fail-fast guarantee.

WHERE IT IS INVOKED — SOURCE ↔ DESTINATION MAPPING:
    *** SOURCE: circuit_breaker.call() in resilient_caller._call_inner()
    *** DESTINATION: MAS nodes that call invoke_specialist/invoke_synthesizer
    Direct llm.invoke() calls BYPASS the circuit breaker. ***

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Script                      │ Nodes (DESTINATION — trigger circuit breaker)│
    ├─────────────────────────────┼─────────────────────────────────────────────┤
    │ supervisor_orchestration     │ pulmonology_worker_node, cardiology_worker_  │
    │                             │ node, nephrology_worker_node, report_        │
    │                             │ synthesis_node                               │
    ├─────────────────────────────┼─────────────────────────────────────────────┤
    │ peer_to_peer_orchestration   │ pulmonology_peer_node, cardiology_peer_node, │
    │                             │ nephrology_peer_node, synthesis_node        │
    ├─────────────────────────────┼─────────────────────────────────────────────┤
    │ dynamic_router_orchestration │ pulmonology_specialist_node, cardiology_     │
    │                             │ specialist_node, nephrology_specialist_node,│
    │                             │ router_report_node                            │
    ├─────────────────────────────┼─────────────────────────────────────────────┤
    │ hybrid_orchestration        │ cardiopulmonary_pulmonology_node,             │
    │                             │ cardiopulmonary_cardiology_node,             │
    │                             │ renal_specialist_node, hybrid_synthesis_node │
    └─────────────────────────────┴─────────────────────────────────────────────┘

    NOT TRIGGERED (direct llm.invoke — no circuit breaker):
        supervisor_decide_node, input_classifier_node, hybrid_supervisor_node;
        graph_of_subgraphs_orchestration (all nodes); handoff/*; MAS_architectures/*;
        guardrails/, HITL/, memory_management/, tools/, observability_and_traceability/

    Call chain (SOURCE): orchestrator.invoke_specialist/invoke_synthesizer
        → ResilientCaller.call(llm.invoke, ...) → circuit_breaker.call(_timed_call)

RECOVERY (OPEN → HALF-OPEN → CLOSED):
    After reset_timeout seconds, breaker moves to HALF-OPEN.
    One test call is allowed. Success → CLOSED. Failure → back to OPEN.

════════════════════════════════════════
  SYSTEM DESIGN — THE THREE STATES
════════════════════════════════════════

                 ┌──────────────────┐
        OK calls │                  │  fail_max reached
                 ▼                  │
    ┌──────────────────┐    ┌───────┴────────────┐
    │     CLOSED       │───▶│      OPEN          │
    │  (normal ops)    │    │  (reject all calls)│
    └──────────────────┘    └───────┬────────────┘
             ▲                      │ reset_timeout
             │ test call succeeds   ▼
             │              ┌──────────────────┐
             └──────────────│   HALF-OPEN      │
                            │  (one test call) │
                            └──────────────────┘
                              │ test call fails
                              ▼
                         re-opens (OPEN)

  CLOSED  → all calls pass through; failures are counted
  OPEN    → all calls rejected immediately (CircuitBreakerOpen raised)
  HALF-OPEN → exactly ONE call is allowed through to test recovery

════════════════════════════════════════
  WHY THIS MATTERS IN A MAS PIPELINE
════════════════════════════════════════
Without a circuit breaker, a 5-agent pipeline hitting a downed OpenAI
API experiences:

    Agent A: 30s timeout → fail
    Agent B: 30s timeout → fail  (running in parallel)
    Agent C: 30s timeout → fail
    Total:   30s latency + 3× wasted API quota

With a circuit breaker, after Agent A's failure opens the breaker:

    Agent B: <1ms → CircuitBreakerOpen raised → fallback used
    Agent C: <1ms → CircuitBreakerOpen raised → fallback used
    Total:   30s latency (first call only) → dramatically lower p99

════════════════════════════════════════
  DESIGN DECISIONS
════════════════════════════════════════
- Uses `pybreaker` under the hood (battle-tested, thread-safe).
- The `CircuitBreakerRegistry` uses the "multiton" pattern — one
  breaker instance per *named service*, shared across all agents
  calling that service. This is critical: if each agent created its
  own breaker, they'd never share failure counts.
- Listeners follow the Observer pattern — state-change events are
  decoupled from the breaker logic itself.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import pybreaker

from .config import CircuitBreakerConfig
from .exceptions import CircuitBreakerOpen

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Internal registry (Multiton pattern)
# Private to this module — external code uses CircuitBreakerRegistry.
# ──────────────────────────────────────────────────────────────────────────────

_registry: dict[str, "CircuitBreaker"] = {}


class _StateChangeListener(pybreaker.CircuitBreakerListener):
    """
    Observes pybreaker state transitions and emits structured log events.

    PATTERN: Observer — the breaker notifies listeners on state change
    without knowing what the listeners do (log, emit metric, alert, etc.).
    Adding a new observer (e.g., Prometheus counter) means adding a new
    Listener subclass — the breaker itself never changes.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def state_change(
        self,
        cb: pybreaker.CircuitBreaker,
        old_state: pybreaker.CircuitBreakerState,
        new_state: pybreaker.CircuitBreakerState,
    ) -> None:
        """Called by pybreaker whenever the state machine transitions.
        *** TRIGGER: When new_state is 'open', all future calls will raise
        CircuitBreakerOpen until reset_timeout elapses (→ HALF-OPEN). ***"""
        logger.warning(
            "Circuit breaker state change",
            extra={
                "breaker": self._name,
                "old_state": old_state.name,
                "new_state": new_state.name,
            },
        )

    def failure(self, cb: pybreaker.CircuitBreaker, exc: Exception) -> None:
        """Called on every recorded failure. *** TRIGGER: func raised an exception;
        pybreaker counted it. If fail_count >= fail_max, breaker will OPEN. ***"""
        logger.warning(
            "Circuit breaker failure recorded",
            extra={
                "breaker": self._name,
                "fail_count": cb.fail_counter,
                "fail_max": cb.fail_max,
                "error": str(exc),
            },
        )

    def success(self, cb: pybreaker.CircuitBreaker) -> None:
        """Called on every recorded success."""
        logger.debug("Circuit breaker success", extra={"breaker": self._name})


class CircuitBreaker:
    """
    A named circuit breaker that wraps calls to an external service.

    USAGE — direct instantiation:
        breaker = CircuitBreaker("openai_api", CircuitBreakerConfig())
        result  = breaker.call(llm.invoke, prompt)

    USAGE — shared instance via registry (recommended for MAS):
        breaker = CircuitBreakerRegistry.get_or_create("openai_api")
        result  = breaker.call(llm.invoke, prompt)

    Why the registry is recommended:
        All agents calling "openai_api" share the same breaker instance.
        If Agent A causes the breaker to open, Agent B's next call is
        rejected immediately — without Agent B making a real API call.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig) -> None:
        self._name = name
        self._breaker = pybreaker.CircuitBreaker(
            fail_max=config.fail_max,
            reset_timeout=config.reset_timeout,
            name=name,
            listeners=[_StateChangeListener(name)],
        )
        logger.info(
            "Circuit breaker created",
            extra={
                "breaker": name,
                "fail_max": config.fail_max,
                "reset_timeout": config.reset_timeout,
            },
        )

    # ── Public interface (satisfies CircuitBreakerProtocol) ─────────────────

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute `func` through the circuit breaker.

        *** TRIGGER CONDITIONS ***

        (A) Breaker is OPEN → CircuitBreakerOpen raised IMMEDIATELY.
            No call to func. No LLM request. Fail-fast.

        (B) Breaker is CLOSED → func is invoked. If func raises ANY exception:
            - Failure is counted (consecutive failure counter increments).
            - Exception is re-raised to caller.
            - If fail_count reaches fail_max → breaker OPENS (next call gets A).

        (C) Breaker is HALF-OPEN → One test call allowed. Success → CLOSED.
            Failure → back to OPEN.

        Args:
            func:   The callable to protect (e.g., llm.invoke).
            *args:  Positional arguments forwarded to `func`.
            **kwargs: Keyword arguments forwarded to `func`.

        Returns:
            Whatever `func` returns.

        Raises:
            CircuitBreakerOpen: if the circuit is currently OPEN.
            Any exception raised by `func` (also recorded as a failure).
        """
        try:
            # pybreaker: if OPEN → reject immediately (CircuitBreakerError).
            # pybreaker: if CLOSED/HALF-OPEN → invoke func; on exception,
            #            increment fail_counter; if fail_counter >= fail_max
            #            → transition to OPEN (next call will be rejected).
            return self._breaker.call(func, *args, **kwargs)
        except pybreaker.CircuitBreakerError as exc:
            # *** TRIGGER: Breaker was OPEN — pybreaker rejected the call
            # before invoking func. Raise our typed exception for callers.
            raise CircuitBreakerOpen(
                f"Circuit breaker '{self._name}' is OPEN — "
                f"service is unhealthy. Calls will resume after "
                f"{self._breaker.reset_timeout}s.",
                details={
                    "breaker_name": self._name,
                    "fail_count": self._breaker.fail_counter,
                    "state": str(self._breaker.current_state),
                },
            ) from exc

    @property
    def is_open(self) -> bool:
        """True when the breaker is OPEN (rejecting all calls)."""
        return self._breaker.current_state == "open"

    @property
    def name(self) -> str:
        """The unique name of this breaker."""
        return self._name

    @property
    def fail_count(self) -> int:
        """Current consecutive failure count."""
        return self._breaker.fail_counter

    def reset(self) -> None:
        """
        Manually force the breaker back to CLOSED state.

        Use this in integration tests or after a known deployment to
        clear stale failure counts. Not recommended in production.
        """
        self._breaker.close()
        logger.info("Circuit breaker manually reset", extra={"breaker": self._name})


class CircuitBreakerRegistry:
    """
    Multiton registry that ensures one CircuitBreaker per named service.

    *** WHERE BREAKERS ARE OBTAINED AND USED ***
        - orchestrator.py: get_or_create("orchestration_llm_api") → passed to
          ResilientCaller; all invoke_specialist/invoke_synthesizer go through it.
        - langgraph_integration_example.py: get_or_create("openai_api").
        - ResilientCaller: accepts circuit_breaker in constructor; used for
          every call unless skip_circuit_breaker=True.

    PATTERN: Multiton (named singleton variant)
        - One instance per key (service name).
        - Shared across all agents in the process.
        - Thread-safe: reads/writes use a single dict (GIL-protected in CPython).

    This is the RECOMMENDED way to get a circuit breaker in a MAS.
    Do not create CircuitBreaker instances directly unless you need
    isolated breakers per agent (e.g., for testing).

    Example:
        # In your LangGraph node:
        breaker = CircuitBreakerRegistry.get_or_create(
            "openai_api",
            CircuitBreakerConfig(fail_max=3)
        )
        result = breaker.call(llm.invoke, prompt)
    """

    _instances: dict[str, CircuitBreaker] = {}

    @classmethod
    def get_or_create(
        cls,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """
        Return the existing breaker for `name`, or create one with `config`.

        If `config` is omitted and the breaker already exists, the existing
        one is returned as-is. If it does not exist, defaults are used.

        Args:
            name:   Unique service name (e.g., "openai_api", "anthropic_api").
            config: Config for a newly created breaker. Ignored if the
                    breaker already exists.

        Returns:
            Shared CircuitBreaker for `name`.
        """
        if name not in cls._instances:
            cls._instances[name] = CircuitBreaker(name, config or CircuitBreakerConfig())
        return cls._instances[name]

    @classmethod
    def reset_all(cls) -> None:
        """
        Clear the registry and close all breakers.

        FOR TESTING ONLY. Clears all shared state between test cases.
        """
        for breaker in cls._instances.values():
            breaker.reset()
        cls._instances.clear()

    @classmethod
    def get_all_statuses(cls) -> dict[str, dict]:
        """
        Return a status snapshot of all registered breakers.

        Useful for a health-check endpoint:
            /health → {"openai_api": {"state": "closed", "fail_count": 0}, ...}
        """
        return {
            name: {
                "state": "open" if b.is_open else "closed",
                "fail_count": b.fail_count,
            }
            for name, b in cls._instances.items()
        }
