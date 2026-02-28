"""
Bulkhead — Resource Pool Isolation
=====================================
Prevents one agent's burst from exhausting shared resources and
starving all other agents in the system.

════════════════════════════════════════════════════════════════════════════════
  *** WHERE AND HOW BULKHEAD IS MAPPED IN MAS ***
════════════════════════════════════════════════════════════════════════════════

**SINGLE USAGE SITE (SOURCE)**:
    resilience/resilient_caller.py
    - Line 76:  from .bulkhead import Bulkhead
    - Line 166: self._bulkhead = Bulkhead(name=agent_name, config=cfg.bulkhead)
    - Lines 215–228: Layer 2 in call() — if skip_bulkhead: call _call_inner directly;
      else: with self._bulkhead.acquire(): _call_inner(...). So bulkhead is applied
      only when skip_bulkhead=False.

**HOW IT IS USED**:
    1. ResilientCaller.__init__() builds one Bulkhead per caller from config.bulkhead.
    2. On each call(), the outermost layer (after token pre-check) is the bulkhead:
       - If skip_bulkhead is True: _call_inner() is invoked directly (no pool).
       - If skip_bulkhead is False: a slot is acquired via self._bulkhead.acquire();
         the context manager runs _call_inner() inside the pool and releases on exit.
    3. acquire() uses a semaphore (max_concurrent slots). If the pool and queue are
       full, BulkheadFull is raised and the caller can map it to OrchestrationResult.

**CURRENT MAS MAPPING — BULKHEAD IS SKIPPED**:
    scripts/orchestration/_base/orchestrator.py passes skip_bulkhead=True in both
    invoke_specialist() and invoke_synthesizer(). So in the current orchestration
    patterns (supervisor, peer_to_peer, dynamic_router, hybrid), the bulkhead is
    NOT applied — no orchestration node acquires a bulkhead slot.

    Rationale (documented in orchestrator): "Bulkhead skipped (linear orchestration
    flows)." Linear supervisor → one worker at a time does not need concurrency
    isolation; bulkhead is for parallel/fan-out flows (e.g. dynamic_router with
    concurrent specialists, or batch jobs).

**WHERE IT WOULD PROTECT IF ENABLED (DESTINATION)**:
    If orchestrator called with skip_bulkhead=False, the same nodes as circuit
    breaker would be inside the bulkhead: all specialist and synthesis nodes in
    supervisor_orchestration, peer_to_peer_orchestration, dynamic_router_orchestration,
    hybrid_orchestration. One shared ResilientCaller implies one shared bulkhead
    pool for all those nodes.

**REFERENCE (bulkhead enabled)**:
    resilience/langgraph_integration_example.py uses per-agent ResilientCaller with
    different BulkheadConfig (e.g. triage max_concurrent=8, summariser max_concurrent=4).

════════════════════════════════════════
  THE SHIP ANALOGY (WHY "BULKHEAD")
════════════════════════════════════════
A bulkhead in a ship is a sealed wall that divides the hull into
watertight compartments. If one compartment floods, the bulkheads
contain the water — the rest of the ship stays afloat.

In software: if one agent floods the shared thread/connection pool,
bulkheads ensure the overload is contained to that agent's pool.
Other agents continue operating normally.

════════════════════════════════════════
  THE PROBLEM IN A MAS
════════════════════════════════════════
Imagine a MAS with a 20-thread executor pool shared by all agents:

  ┌───────────────────────────────────────┐
  │  Shared thread pool (20 threads)      │
  │                                       │
  │  Triage Agent      ████░░░░░░░ (2)    │
  │  Batch Summariser  ████████████ (18!) │← flood
  │  Pharmacology      ░░░░░░░░░░░ (0)   │← starved
  └───────────────────────────────────────┘

The Batch Summariser processes 18 documents simultaneously, grabs
all 18 available threads, and the time-critical Triage agent cannot
get a thread. User requests time out.

With bulkheads:
  ┌────────────────────────────────────────────┐
  │  Triage pool      (max 5 concurrent) ████  │
  │  Batch pool       (max 10 concurrent) ████ │
  │  Pharmacology pool(max 5 concurrent) ████  │
  └────────────────────────────────────────────┘

════════════════════════════════════════
  IMPLEMENTATION: SEMAPHORE
════════════════════════════════════════
A threading.Semaphore is the ideal primitive for a bulkhead:
- Tracks the number of available "slots" atomically.
- `acquire()` decrements the count (entering the pool).
- `release()` increments it (leaving the pool).
- If the count is 0, `acquire()` blocks or returns False (non-blocking).

Queue semantics: callers can opt into a bounded wait queue. If the pool
is full but the queue is not, the caller waits up to `queue_timeout`
seconds. If the queue is also full, BulkheadFull is raised immediately
(shed the load — don't queue indefinitely).
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from .config import BulkheadConfig
from .exceptions import BulkheadFull

logger = logging.getLogger(__name__)


class Bulkhead:
    """
    Bounds the number of concurrent calls to a resource (e.g., LLM API).

    Callers that exceed `max_concurrent` are queued (up to `max_queue`).
    Callers that exceed the queue are rejected with BulkheadFull.

    Args:
        name:   Identifier used in logs (e.g., "triage_agent_pool").
        config: BulkheadConfig with max_concurrent and max_queue limits.

    Example — single call:
        bulkhead = Bulkhead("llm_pool", BulkheadConfig(max_concurrent=5))
        result   = bulkhead.call(llm.invoke, prompt)

    Example — context manager (for multi-step operations inside one slot):
        bulkhead = Bulkhead("llm_pool", BulkheadConfig(max_concurrent=5))

        with bulkhead.acquire():
            response = llm.invoke(prompt_a)
            follow_up = llm.invoke(prompt_b)

    Example — different pools for different agent priorities:
        triage_bulkhead = Bulkhead("triage", BulkheadConfig(max_concurrent=5))
        batch_bulkhead  = Bulkhead("batch",  BulkheadConfig(max_concurrent=3))
    """

    def __init__(self, name: str, config: BulkheadConfig) -> None:
        self._name = name
        self._max_concurrent = config.max_concurrent
        self._max_queue = config.max_queue

        # Semaphore tracks available slots.
        # Initialized to max_concurrent — that many callers can enter simultaneously.
        self._semaphore = threading.Semaphore(config.max_concurrent)

        # Track currently active calls and queue size for observability.
        self._lock = threading.Lock()
        self._active: int = 0
        self._queued: int = 0

        logger.info(
            "Bulkhead created",
            extra={
                "bulkhead": name,
                "max_concurrent": config.max_concurrent,
                "max_queue": config.max_queue,
            },
        )

    @contextmanager
    def acquire(self, queue_timeout: float = 5.0) -> Iterator[None]:
        """
        Context manager that acquires a slot in the bulkhead pool.

        If the pool is full, waits up to `queue_timeout` seconds for a
        slot to open. If no slot opens in time (or the queue is full),
        raises BulkheadFull.

        *** MAS: Invoked from resilience/resilient_caller.py call() when
        skip_bulkhead=False. Orchestrator uses skip_bulkhead=True, so
        no orchestration node currently goes through the bulkhead. ***

        Args:
            queue_timeout: Seconds to wait for a free slot. Default 5s.

        Raises:
            BulkheadFull: if the pool AND queue are both full.

        Usage:
            with bulkhead.acquire():
                result = llm.invoke(prompt)
        """
        with self._lock:
            current_active = self._active
            current_queued = self._queued

            # Reject immediately if we'd exceed the queue
            if current_active >= self._max_concurrent and current_queued >= self._max_queue:
                raise BulkheadFull(
                    f"Bulkhead '{self._name}' is at capacity "
                    f"(pool={current_active}/{self._max_concurrent}, "
                    f"queue={current_queued}/{self._max_queue}). "
                    f"Shedding load.",
                    details={
                        "pool_name": self._name,
                        "max_concurrent": self._max_concurrent,
                        "current_active": current_active,
                        "current_queued": current_queued,
                        "max_queue": self._max_queue,
                    },
                )

            self._queued += 1

        try:
            # Block until a slot opens (up to queue_timeout seconds)
            acquired = self._semaphore.acquire(timeout=queue_timeout)
        finally:
            with self._lock:
                self._queued -= 1

        if not acquired:
            raise BulkheadFull(
                f"Bulkhead '{self._name}' queue timeout after {queue_timeout}s — "
                f"no slot opened in time.",
                details={
                    "pool_name": self._name,
                    "max_concurrent": self._max_concurrent,
                    "queue_timeout": queue_timeout,
                },
            )

        with self._lock:
            self._active += 1

        logger.debug(
            "Bulkhead slot acquired",
            extra={"bulkhead": self._name, "active": self._active},
        )

        try:
            yield
        finally:
            self._semaphore.release()
            with self._lock:
                self._active -= 1
            logger.debug(
                "Bulkhead slot released",
                extra={"bulkhead": self._name, "active": self._active},
            )

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute `func` inside the bulkhead pool.

        Convenience wrapper around `acquire()` for single-call usage.

        Args:
            func:     The callable to execute.
            *args:    Positional args forwarded to `func`.
            **kwargs: Keyword args forwarded to `func`.

        Returns:
            Whatever `func` returns.

        Raises:
            BulkheadFull: if pool and queue are both at capacity.
        """
        with self.acquire():
            return func(*args, **kwargs)

    @property
    def active_count(self) -> int:
        """Number of calls currently in progress."""
        return self._active

    @property
    def name(self) -> str:
        return self._name
