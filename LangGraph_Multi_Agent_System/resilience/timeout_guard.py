"""
Timeout Guard — Hard Deadline Enforcement
==========================================
Ensures every LLM call returns within a configured deadline.
Raises `TimeoutExceeded` if the call takes too long.

════════════════════════════════════════════════════════════════════════════════
  *** WHERE AND HOW TIMEOUT GUARD IS TRIGGERED IN MAS ***
════════════════════════════════════════════════════════════════════════════════

**TRIGGER POINT**: resilience/resilient_caller.py → ResilientCaller.call()
    - TimeoutGuard is instantiated in ResilientCaller.__init__() (line 160)
    - Triggered via _call_inner() → _timed_call() wrapper (innermost layer)
    - Execution order: timeout is the INNERMOST layer (closest to actual LLM call)

**WHERE IT'S USED (SOURCE)**:
    1. resilience/resilient_caller.py — ResilientCaller._call_inner()
       - Creates _timed_call() which wraps func in timeout_guard.call_with_timeout()
       - Applied to EVERY call unless skip_timeout=True

**WHERE IT PROTECTS (DESTINATION)**:
    All nodes that use ResilientCaller for LLM invocations:
    
    1. scripts/orchestration/_base/orchestrator.py
       - invoke_specialist() — protects all specialist LLM calls
       - invoke_synthesizer() — protects synthesis LLM call
       
    2. scripts/orchestration/supervisor_orchestration/agents.py
       - pulmonology_worker_node (via invoke_specialist)
       - cardiology_worker_node (via invoke_specialist)
       - nephrology_worker_node (via invoke_specialist)
       - report_synthesis_node (via invoke_synthesizer)
       
    3. scripts/orchestration/peer_to_peer_orchestration/agents.py
       - All peer nodes (pulm, cardio, nephro) via invoke_specialist
       - report node via invoke_synthesizer
       
    4. scripts/orchestration/dynamic_router_orchestration/agents.py
       - Specialist workers via invoke_specialist
       - report node via invoke_synthesizer
       
    5. scripts/orchestration/hybrid_orchestration/agents.py
       - cardiopulmonary_pulmonology_node via invoke_specialist
       - cardiopulmonary_cardiology_node via invoke_specialist
       - renal_specialist_node via invoke_specialist
       - hybrid_synthesis_node via invoke_synthesizer

**CONFIGURATION**:
    - Default timeout: 30 seconds (config.timeout.default_timeout)
    - Configurable per-call via timeout= parameter
    - Can be disabled per-call with skip_timeout=True

**WHAT HAPPENS WHEN TIMEOUT IS EXCEEDED**:
    1. TimeoutGuard.call_with_timeout() raises TimeoutExceeded
    2. ResilientCaller propagates it up
    3. Orchestrator catches it, returns OrchestrationResult(was_successful=False)
    4. Workflow continues with failed result (synthesis skips it)

════════════════════════════════════════════
  WHY TIMEOUTS ARE NON-NEGOTIABLE IN A MAS
════════════════════════════════════════════
Without timeouts, a single hung LLM call can block an entire downstream
pipeline indefinitely. Consider a 5-agent sequential workflow:

    Triage → Specialist → Pharmacology → Reviewer → Summariser

If "Specialist" hangs (e.g., model is under heavy load), all downstream
agents wait indefinitely. The user's request never completes.

Worse: if agents run in parallel, a thread-pool executor (used by
LangGraph's async engine) can exhaust its thread pool waiting for hung
calls, starving every other workflow in the process.

Timeouts enforce a contract: "I will respond in X seconds or admit
failure so the system can react intelligently."

════════════════════════════════════════════
  RECOMMENDED TIMEOUT STRATEGY
════════════════════════════════════════════
Use LAYERED timeouts — each layer slightly shorter than the one above:

    HTTP transport timeout (provider SDK):   60s  — last resort
    TimeoutGuard (this class):               30s  — primary application timeout
    LangGraph node timeout:                  45s  — node-level budget

This ensures TimeoutGuard catches the problem cleanly before the HTTP
layer tears down the connection with a less informative error.

════════════════════════════════════════════
  IMPLEMENTATION: ThreadPool vs Signal
════════════════════════════════════════════
This implementation uses concurrent.futures.ThreadPoolExecutor to run
the call in a monitored thread with a deadline.

Why not `signal.alarm`?
  - signal.alarm only works on the main thread (useless inside a LangGraph
    node, which typically runs in a worker thread).
  - signal.alarm is POSIX-only (fails on Windows).

Why not asyncio.wait_for?
  - This module must support both sync and async callers.
  - The synchronous executor approach works in both contexts.

Trade-off: Using a thread does add ~0.5ms overhead per call. Acceptable
for LLM calls (which take 1–30s). Would not use for sub-millisecond ops.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from .config import TimeoutConfig
from .exceptions import TimeoutExceeded

logger = logging.getLogger(__name__)

# A small shared executor for timeout enforcement.
# Using a small pool (not unbounded) so timeout enforcement itself
# doesn't consume excessive resources under load.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="timeout-guard")


class TimeoutGuard:
    """
    Enforces a per-call deadline on LLM and tool calls.

    Uses a background thread to run the call with a strict deadline.
    If the call exceeds the deadline, raises TimeoutExceeded.

    Args:
        config:     TimeoutConfig with `default_timeout` in seconds.
        agent_name: Used in log messages and exception details.
                    Typically the LangGraph node name.

    Example — basic usage:
        guard  = TimeoutGuard(TimeoutConfig(), agent_name="pharmacology_agent")
        result = guard.call_with_timeout(llm.invoke, prompt)

    Example — override timeout per call:
        result = guard.call_with_timeout(
            llm.invoke, prompt,
            timeout=10.0,   # this call has a tighter 10s deadline
        )
    """

    def __init__(self, config: TimeoutConfig, agent_name: str = "unknown") -> None:
        self._default_timeout = config.default_timeout
        self._agent_name = agent_name

    def call_with_timeout(
        self,
        func: Callable,
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute `func` with a hard deadline.

        Args:
            func:     Callable to execute.
            *args:    Positional args forwarded to `func`.
            timeout:  Deadline in seconds. Uses `config.default_timeout` if None.
            **kwargs: Keyword args forwarded to `func`.

        Returns:
            Whatever `func` returns if it completes in time.

        Raises:
            TimeoutExceeded: if `func` does not return within the deadline.
            Any exception raised by `func` is re-raised as-is.
        """
        deadline = timeout if timeout is not None else self._default_timeout
        start_time = time.monotonic()

        future: Future = _executor.submit(func, *args, **kwargs)

        try:
            return future.result(timeout=deadline)

        except FutureTimeoutError:
            # The call is still running in the background; cancel its future.
            # Note: `cancel()` has no effect once the call has started — we
            # accept this (the background thread will eventually finish or
            # error out on its own). The important thing is WE stop waiting.
            future.cancel()

            elapsed = round(time.monotonic() - start_time, 2)

            logger.error(
                "LLM call timed out",
                extra={
                    "agent": self._agent_name,
                    "deadline_secs": deadline,
                    "elapsed_secs": elapsed,
                },
            )

            raise TimeoutExceeded(
                f"Agent '{self._agent_name}' exceeded its {deadline}s deadline "
                f"({elapsed}s elapsed).",
                details={
                    "agent_name": self._agent_name,
                    "timeout_secs": deadline,
                    "elapsed_secs": elapsed,
                },
            )
