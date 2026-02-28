# Resilience-to-MAS Mapping

**Scope**: Patterns under `scripts/` — `orchestration/`, `handoff/`, `MAS_architectures/`. Excludes `script_XX` scripts and `_bckp` files.

---

## Contents

1. [The Call Execution Stack](#1-the-call-execution-stack)
2. [Topology-Driven Pattern Selection](#2-topology-driven-pattern-selection)
3. [Pattern Implementation Across MAS Workflows](#3-pattern-implementation-across-mas-workflows)
4. [The Circuit Breaker Registry — Cross-Agent Shared State](#4-the-circuit-breaker-registry--cross-agent-shared-state)
5. [Token Budget — The Two-Step Lifecycle](#5-token-budget--the-two-step-lifecycle)
6. [Error Propagation — From Resilience Exception to Workflow State](#6-error-propagation--from-resilience-exception-to-workflow-state)
7. [Configuration Reference](#7-configuration-reference)
8. [Adoption Guide — Adding Resilience to a New Workflow](#8-adoption-guide--adding-resilience-to-a-new-workflow)

---

## 1. The Call Execution Stack

Every LLM invocation in this system — whether from a supervisor node, a specialist worker, or a synthesis node — passes through the same ordered stack of resilience layers. The layers are applied from outermost (cheapest to evaluate, widest scope) to innermost (closest to the actual API call).

```
┌─ [Layer 1] Token Budget Pre-Check ─────────────────────────────────────┐
│   Cost: O(1) arithmetic.  Scope: entire workflow execution.            │
│   Fails fast before consuming any resource if budget is exhausted.     │
│                                                                         │
│   ┌─ [Layer 2] Bulkhead Concurrency Pool ───────────────────────────┐  │
│   │   Cost: semaphore acquire.  Scope: per agent type / per pool.  │  │
│   │   Reject before acquiring a rate-limit slot if pool is full.   │  │
│   │                                                                 │  │
│   │   ┌─ [Layer 3] Rate Limiter (sliding window) ───────────────┐  │  │
│   │   │   Cost: lock + list prune.  Scope: per API provider.    │  │  │
│   │   │   Smooth burst traffic before reaching the breaker.     │  │  │
│   │   │                                                          │  │  │
│   │   │   ┌─ [Layer 4] Circuit Breaker ────────────────────┐    │  │  │
│   │   │   │   Cost: <1ms dict lookup.  Scope: per API.     │    │  │  │
│   │   │   │   Fail immediately on known-unhealthy service. │    │  │  │
│   │   │   │                                                │    │  │  │
│   │   │   │   ┌─ [Layer 5] Retry Handler ──────────────┐  │    │  │  │
│   │   │   │   │   Catches transient exceptions only.   │  │    │  │  │
│   │   │   │   │   Exponential backoff + jitter.        │  │    │  │  │
│   │   │   │   │                                        │  │    │  │  │
│   │   │   │   │   ┌─ [Layer 6] Timeout Guard ──────┐  │  │    │  │  │
│   │   │   │   │   │   Hard per-attempt deadline.   │  │  │    │  │  │
│   │   │   │   │   │   Thread-safe; works in any    │  │  │    │  │  │
│   │   │   │   │   │   LangGraph worker thread.     │  │  │    │  │  │
│   │   │   │   │   │                                │  │  │    │  │  │
│   │   │   │   │   │       llm.invoke(prompt)       │  │  │    │  │  │
│   │   │   │   │   └────────────────────────────────┘  │  │    │  │  │
│   │   │   │   └───────────────────────────────────────┘  │    │  │  │
│   │   │   └────────────────────────────────────────────────┘    │  │  │
│   │   └──────────────────────────────────────────────────────────┘  │  │
│   └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
After call: token_manager.record_usage(agent_name, tokens_in, tokens_out)
```

**Why this order matters:**

Each layer's position reflects the cost of the check and who should bear that cost:

- **Token budget is outermost** because it is pure arithmetic (no lock, no network). If the budget is exhausted, there is no justification for acquiring a bulkhead slot or consuming a rate-limit slot.
- **Bulkhead before rate limiter** because pool slots are process-local resources. It makes no sense to consume a rate-limit slot — which counts against an API quota — only to then be rejected by a full pool.
- **Rate limiter before circuit breaker** because the circuit breaker is inside the rate limiter. If the service is known-down, the call should be rejected by the breaker without consuming a rate-limit slot.
- **Retry wraps the circuit breaker** so that each retry attempt individually goes through the breaker. If the first attempt fails and opens the circuit, subsequent retry attempts are rejected in under a millisecond — not after another 30-second timeout.
- **Timeout is innermost** because the deadline applies per attempt, not per retry sequence. Three retries with a 30-second timeout means each attempt has a 30-second deadline, giving the retry strategy up to 90 seconds of total wall time.

---

## 2. Topology-Driven Pattern Selection

Not every resilience pattern is appropriate for every workflow topology. The decision of which patterns to enable depends on whether the workflow is **linear or parallel**, and on how many agents share the same API.

This is the most important design principle in the resilience layer: **the workflow topology determines which failure modes exist, and patterns are chosen to address only those failure modes**.

### Linear Sequential Topology

```
Supervisor → Worker A → Supervisor → Worker B → Supervisor → END
```

In a linear topology, at most one agent calls the LLM at any given moment. There is no concurrency competition between agents.

| Pattern | Enable? | Reasoning |
|---|---|---|
| Circuit Breaker | **Yes** | All agents share the same API. One failure history protects all sequential calls. |
| Retry | **Yes** | Transient failures occur regardless of topology. |
| Timeout | **Yes** | A hung call blocks the entire downstream chain. |
| Rate Limiter | **Yes** | The supervisor makes routing calls between every specialist. Even in a sequential graph, 4–5 agents × the supervisor = 8–10 calls per workflow. Smoothing is still valuable. |
| Token Budget | **Yes** | Cumulative cost accumulates across sequential calls. Five agents × 500 tokens each = 2,500 tokens per workflow run. |
| Bulkhead | **No** | No concurrency competition exists. A semaphore on a sequential call adds overhead with no benefit. |

### Parallel Fan-out Topology

```
Supervisor → [Worker A ║ Worker B ║ Worker C] → Synthesis → END
```

In a parallel topology (dynamic router, voting pattern), multiple agents invoke the LLM simultaneously from separate threads. This creates failure modes that do not exist in linear flows.

| Pattern | Enable? | Reasoning |
|---|---|---|
| Circuit Breaker | **Yes** | Shared breaker means one failure history protects all parallel branches. If Worker A causes the circuit to open, Workers B and C skip in under a millisecond. |
| Retry | **Yes** | Transient failures occur in parallel branches independently. |
| Timeout | **Yes** | A hung call in one branch blocks its thread. Without timeout, a hung Worker B eventually exhausts the thread pool and starves Worker A and C. |
| Rate Limiter | **Yes, critical** | Three agents issuing calls simultaneously hits the API with a burst. Even if the hourly quota is fine, per-second sub-limits can trigger 429s. |
| Token Budget | **Yes** | Three agents running in parallel consume 3× the tokens simultaneously. Budget enforcement prevents the synthesis step from being reached when the workflow is already over budget. |
| Bulkhead | **Yes** | Without a bulkhead, a batch job with 20 parallel agents can exhaust the thread pool and starve a real-time supervisor agent. Bulkhead gives each agent type its own bounded pool. |

### Reflection / Iterative Loop Topology

```
Generator → Critic → Generator → Critic → ... (up to max_iterations)
```

The loop topology has a unique failure mode: **unbounded token cost**. A loop configured for 10 iterations that encounters a verbose response at each step can easily consume 5× the expected token budget.

| Pattern | Enable? | Reasoning |
|---|---|---|
| Circuit Breaker | **Yes** | Shared failure state across loop iterations. |
| Retry | **Yes** | Standard. |
| Timeout | **Yes** | Standard. |
| Rate Limiter | **Yes** | The loop generates N sequential calls to the same API. |
| Token Budget | **Yes, critical** | This is the topology where token budget enforcement is most important. The `max_iterations` guard on the loop is a count-based ceiling. Token budget adds a cost-based ceiling, catching cases where iteration count is within limits but each iteration is consuming far more tokens than expected. |
| Bulkhead | **Situational** | Only relevant if multiple independent loops run concurrently. |

---

## 3. Pattern Implementation Across MAS Workflows

### The Orchestration Base — `scripts/orchestration/_base/orchestrator.py`

All five orchestration patterns inherit from `BaseOrchestrator`. Resilience is wired once into `invoke_specialist()` and `invoke_synthesizer()`. Every inheriting pattern gets these protections automatically.

```
BaseOrchestrator
    │
    ├── supervisor_orchestration
    ├── peer_to_peer_orchestration
    ├── dynamic_router_orchestration
    ├── graph_of_subgraphs_orchestration
    └── hybrid_orchestration
```

**Patterns always applied** (inherited by all five orchestration types):

| Pattern | Configuration |
|---|---|
| Circuit Breaker | Shared via `CircuitBreakerRegistry.get_or_create("orchestration_llm_api")` |
| Retry | `RetryConfig` defaults (3 retries, exponential backoff, jitter) |
| Timeout | `TimeoutConfig` defaults (30s per attempt) |
| Rate Limiter | Enabled — `skip_rate_limiter=False` |

**Patterns opt-in** (must be explicitly activated via state injection):

| Pattern | How to activate |
|---|---|
| Token Budget | Create `TokenManager` in `runner.py`; inject into `initial_state`; pass `token_manager=state.get("token_manager")` to `invoke_specialist()` and `invoke_synthesizer()`. No changes to `orchestrator.py` required. |

**Patterns deliberately skipped** (default for all orchestration calls):

| Pattern | Why skipped by default |
|---|---|
| Bulkhead | Orchestration patterns are predominantly linear (supervisor routes to one worker at a time). Bulkhead adds semaphore overhead with no isolation benefit when there is no concurrent competition. See Section 2 — enable for parallel fan-out flows. |

**Token Budget adoption status by orchestration pattern:**

| Orchestration Pattern | Agent Count per Run | Token Budget | Notes |
|---|---|---|---|
| `supervisor_orchestration` | 4 (3 specialists + synthesis) | **Enabled** | Reference implementation |
| `peer_to_peer_orchestration` | 3–4 | Not yet enabled | Same state injection pattern applies |
| `dynamic_router_orchestration` | 2–5 (variable routing) | Not yet enabled | Parallel fan-out makes this higher priority |
| `graph_of_subgraphs_orchestration` | Variable | Not yet enabled | Sub-graph boundaries complicate scope |
| `hybrid_orchestration` | 4+ | Not yet enabled | Mix of sequential and parallel |

### `scripts/MAS_architectures/` — Resilience Applicability

Each architecture has a distinct topology. The pattern selection follows the topology rules from Section 2.

| Architecture | Topology | LLM Calls per Run | Priority Patterns | Notes |
|---|---|---|---|---|
| `reflection_self_critique` | Iterative loop | N × 2 (generator + critic per iteration) | Token Budget (critical), Timeout, Circuit Breaker, Retry | `max_iterations` guard exists; token budget adds cost-based ceiling independent of iteration count |
| `parallel_voting` | Parallel fan-out | 3+ simultaneous specialist calls | Bulkhead, Rate Limiter (critical), Circuit Breaker, Token Budget | All specialist calls hit the API simultaneously — rate limiter burst protection is most important |
| `adversarial_debate` | Sequential + iterative | 5 per debate (opening × 2, rebuttals × 2, judge) | Token Budget, Timeout, Circuit Breaker, Retry | Verbose generation at each step; token budget prevents runaway cost |
| `sequential_pipeline` | Linear | N (one per stage) | Circuit Breaker, Retry, Timeout | Simplest topology; bulkhead not needed |
| `supervisor` | Linear supervisor + workers | N workers + N supervisor routing calls | Circuit Breaker, Retry, Timeout, Token Budget | Mirror of orchestration/supervisor pattern |

### `scripts/handoff/` — Resilience Applicability

Handoff scripts are typically demonstration-level single-workflow scripts. The failure modes are narrower than orchestration patterns.

| Script | Topology | Recommended Patterns |
|---|---|---|
| `linear_pipeline` | Sequential | Circuit Breaker, Timeout, Retry |
| `conditional_routing` | Sequential with branches | Circuit Breaker, Timeout, Retry |
| `parallel_fanout` | Parallel | Bulkhead, Rate Limiter, Circuit Breaker, Timeout, Retry, Token Budget |
| `supervisor` | Linear supervisor + workers | Circuit Breaker, Timeout, Retry, Token Budget |
| `multihop_depth_guard` | Iterative | Token Budget (critical), Timeout, Circuit Breaker, Retry |

---

## 4. The Circuit Breaker Registry — Cross-Agent Shared State

The circuit breaker's effectiveness in a multi-agent system depends entirely on all agents sharing the same breaker instance for the same API. If each agent creates its own breaker, the failure count is not shared — each agent independently re-learns that the API is down through its own expensive timeout cycle.

The `CircuitBreakerRegistry` enforces a single instance per named API across the entire process:

```python
# In orchestrator.py — called at BaseOrchestrator initialisation
_SHARED_LLM_API_CIRCUIT_BREAKER = CircuitBreakerRegistry.get_or_create(
    "orchestration_llm_api",
    CircuitBreakerConfig(fail_max=5, reset_timeout=60),
)

# The same object is returned every time for the same name
assert (
    CircuitBreakerRegistry.get_or_create("orchestration_llm_api")
    is CircuitBreakerRegistry.get_or_create("orchestration_llm_api")
)
```

The practical consequence of shared state in a multi-agent workflow:

```
t=0.0s  Specialist A calls invoke_specialist() → API returns 503
t=0.0s  Failure #1 recorded on shared breaker (fail_count: 1/5)

t=5.0s  Specialist B calls invoke_specialist() → API returns 503
t=5.0s  Failure #2 recorded (fail_count: 2/5)

...

t=20.0s Specialist E causes failure #5 → breaker OPENS

t=20.1s Report Synthesis calls invoke_synthesizer() → CircuitBreakerOpen raised in <1ms
        No API call made. No 30-second timeout incurred.
        Synthesis receives OrchestrationResult(was_successful=False) immediately.
```

Without the shared registry, each agent would have its own breaker at fail_count=0. The synthesis node would make a real API call, wait 30 seconds for a timeout, and only then fail. With the shared registry, it fails in under a millisecond.

### The Three States

```
                   fail_max consecutive failures
  ┌──────────┐   ──────────────────────────────────►   ┌──────────┐
  │  CLOSED  │                                          │   OPEN   │
  │ (healthy)│   ◄────────────────────────────────────  │  (down)  │
  └──────────┘     test call succeeds                   └────┬─────┘
        ▲                                                     │
        │             ┌──────────────┐                       │ reset_timeout seconds
        └─────────────│  HALF-OPEN   │◄──────────────────────┘
      test succeeded  │  (one test)  │
                      └──────────────┘
                            │
                            │ test call fails
                            ▼
                      re-opens (OPEN)
```

**CLOSED**: All calls pass through. Each failure increments the counter. A success resets it to zero.

**OPEN**: All calls are rejected immediately (`CircuitBreakerOpen` raised). No API call is made. The breaker remains open for `reset_timeout` seconds, then transitions to HALF-OPEN.

**HALF-OPEN**: Exactly one test call is admitted. If it succeeds, the breaker closes and the failure count resets. If it fails, the breaker reopens and the `reset_timeout` timer restarts.

---

## 5. Token Budget — The Two-Step Lifecycle

The token budget enforces a ceiling on cumulative API cost across an entire workflow run. Unlike other patterns, which are purely reactive (they intercept or retry calls), the token budget has a two-step lifecycle: a pre-call check and a post-call record.

```
  BEFORE the LLM call                          AFTER the LLM call
  ─────────────────                            ──────────────────
  [1] Estimate input tokens                    [3] Extract actual usage from response
      estimated = counter.count(prompt)             actual_in  = response.usage_metadata["input_tokens"]
                                                     actual_out = response.usage_metadata["output_tokens"]
  [2] Check budget (Layer 1 of stack)
      manager.check_budget(                    [4] Record actual usage
          agent_name,                               manager.record_usage(
          estimated_tokens=estimated                    agent_name,
      )                                                 tokens_in=actual_in,
      # Raises TokenBudgetExceeded                      tokens_out=actual_out,
      # if budget would be exceeded.               )
      # No API call is made.                      # Updates the running total
                                                  # that [2] reads next call.
  ──────────────────────────────────────────────────────────────────────────────
                              ↑
                    The actual LLM call happens here.
```

**Why estimate, not count exactly?** The exact input token count is only known after the call completes — it comes back in the response metadata. Before the call, we use `TokenCounter.count(prompt)` which tokenizes the prompt text locally using tiktoken. This estimate is accurate for input tokens. Output token count cannot be predicted (it depends on the model's generation). Estimation is intentionally conservative: better to over-estimate input and refuse a call that would have been within budget than to under-estimate and exceed the budget on output.

**Why pre-check at all, not just post-record?** The pre-check is the fail-fast mechanism. Without it, the call would be made, tokens would be consumed, and only afterward would you discover the budget was exceeded. The pre-check prevents wasted API calls when the budget is already exhausted.

**Scope of the TokenManager instance:** One `TokenManager` must be created per workflow execution, not per process boot. The manager's internal counter accumulates across all calls for one workflow run. If it were reused across runs, cumulative counts from previous executions would reduce the apparent remaining budget for new ones.

```python
# In runner.py — one fresh manager per workflow execution
per_run_token_manager = TokenManager(
    TokenBudgetConfig(max_tokens_per_workflow=8_000)
)

initial_state = {
    "token_manager": per_run_token_manager,
    ...
}
```

**The cumulative effect across agents in supervisor_orchestration:**

```
Start of workflow:   budget used =     0 / 8,000 tokens

Supervisor call #1:  estimate 150 in  → admitted
                     actual:  148 in + 95 out  → record 243
                     budget used =   243 / 8,000

Pulmonology node:    estimate 600 in  → admitted
                     actual:  590 in + 380 out → record 970
                     budget used = 1,213 / 8,000

Cardiology node:     estimate 600 in  → admitted
                     actual:  610 in + 420 out → record 1,030
                     budget used = 2,243 / 8,000

Nephrology node:     estimate 600 in  → admitted
                     actual:  580 in + 350 out → record 930
                     budget used = 3,173 / 8,000

Synthesis node:      estimate 2,000 in → admitted (3,173 + 2,000 = 5,173 < 8,000)
                     actual:  1,890 in + 720 out → record 2,610
                     budget used = 5,783 / 8,000
```

Each `check_budget` call reads the cumulative total from all prior calls, not just from the current node. This is why the manager must be shared across all agents via state.

---

## 6. Error Propagation — From Resilience Exception to Workflow State

Resilience exceptions must not crash the LangGraph graph. If a specialist node raises an unhandled exception, LangGraph's execution engine will typically propagate it up and abort the entire workflow — not just the failing node.

The orchestration base catches all resilience exceptions and converts them to a typed result:

```python
# In BaseOrchestrator.invoke_specialist()
try:
    response = _ORCHESTRATION_CALLER.call(llm.invoke, prompt, ...)
    return OrchestrationResult(
        was_successful=True,
        content=response.content,
    )
except CircuitBreakerOpen as exc:
    return OrchestrationResult(was_successful=False, error_message=str(exc))
except TimeoutExceeded as exc:
    return OrchestrationResult(was_successful=False, error_message=str(exc))
except TokenBudgetExceeded as exc:
    return OrchestrationResult(was_successful=False, error_message=str(exc))
# ... other resilience exceptions ...
except Exception as exc:
    return OrchestrationResult(was_successful=False, error_message=str(exc))
```

This converts every possible failure mode into the same result type. The worker node then reads `result.was_successful` to decide what to store in state.

**Why the synthesis node is different:**

`invoke_synthesizer()` re-raises resilience exceptions as `RuntimeError` rather than converting them to `OrchestrationResult`. The reasoning is that synthesis failure is a workflow-level failure, not a recoverable specialist failure. The report node that calls `invoke_synthesizer()` handles the `RuntimeError` and returns a partial or error report.

```
Specialist failure:    OrchestrationResult(was_successful=False)
                       → downstream synthesis receives empty specialist output
                       → synthesis still runs; uses only successful results

Synthesis failure:     RuntimeError raised to report node
                       → report node returns a partial report with error noted
                       → workflow completes with degraded but useful output
```

**Downstream behavior for each exception type:**

| Exception | Raised by | Converted to | Downstream effect |
|---|---|---|---|
| `CircuitBreakerOpen` | `invoke_specialist` | `OrchestrationResult(was_successful=False)` | Synthesis skips this specialist's output |
| `TimeoutExceeded` | `invoke_specialist` | `OrchestrationResult(was_successful=False)` | Same |
| `TokenBudgetExceeded` | `invoke_specialist` | `OrchestrationResult(was_successful=False)` | Same; remaining specialists still run but their pre-checks will also fail |
| `RateLimitExhausted` | `invoke_specialist` | `OrchestrationResult(was_successful=False)` | Same; only raised in non-blocking rate limiter mode |
| `BulkheadFull` | `invoke_specialist` | `OrchestrationResult(was_successful=False)` | Same; only raised when bulkhead is enabled |
| `CircuitBreakerOpen` | `invoke_synthesizer` | `RuntimeError` | Report node handles; returns partial report |
| `TimeoutExceeded` | `invoke_synthesizer` | `RuntimeError` | Same |
| `TokenBudgetExceeded` | `invoke_synthesizer` | `RuntimeError` | Same; by this point all specialist tokens are spent |

**Note on `TokenBudgetExceeded` in multi-specialist flows:** When the token budget is exhausted mid-workflow, the first specialist to trigger `TokenBudgetExceeded` stores `OrchestrationResult(was_successful=False)` in state. All subsequent specialist nodes will also fail their pre-check (Layer 1) because the shared `TokenManager` counter is at or beyond the limit. The synthesis node receives multiple failed specialist results and composes a partial report.

---

## 7. Configuration Reference

All configuration is defined in `resilience/config.py` as frozen dataclasses. The orchestration base uses the following defaults, which reflect conservative production settings:

### Circuit Breaker

| Parameter | Default | Rationale |
|---|---|---|
| `fail_max` | `5` | Five consecutive failures before opening. Low enough to detect a real outage quickly; high enough that three random 429s in a busy window do not trip the breaker. |
| `reset_timeout` | `60` | Sixty seconds OPEN before attempting a HALF-OPEN test. Most LLM provider rate-limit windows reset within 60 seconds. |

### Retry

| Parameter | Default | Rationale |
|---|---|---|
| `max_retries` | `3` | Three retries after the initial attempt (four total). At `initial_wait=1.0` with exponential backoff, the fourth attempt comes after ~7 seconds — acceptable for LLM calls. Going above 5 retries with exponential backoff can exceed 30 seconds of total wait. |
| `initial_wait` | `1.0` | One second base. LLM providers typically recover from transient errors in 1–5 seconds. |
| `max_wait` | `30.0` | Cap at 30 seconds per wait interval. Prevents backoff from growing to impractical values on attempt 5+. |
| `jitter` | `1.0` | ±1 second random offset per wait. Prevents thundering herd when multiple agents encounter a rate limit simultaneously. |

Retry applies only to `TRANSIENT_EXCEPTIONS`. These are defined as a module-level tuple in `resilience/retry_handler.py` and are the single authoritative list. When adding a new LLM provider, add its transient exception types there, not in each agent.

### Timeout

| Parameter | Default | Rationale |
|---|---|---|
| `default_timeout` | `30.0` | 30 seconds per attempt. Most LLM providers have a server-side timeout of 60 seconds; using 30 seconds at the application layer leaves a full timeout cycle available for retry. Layered against an HTTP transport timeout of 60 seconds, the application layer always fires first. |

### Rate Limiter

| Parameter | Default | Rationale |
|---|---|---|
| `max_calls` | `60` | 60 calls per minute = 1 per second average. Conservative for most LLM provider tiers. |
| `period` | `60.0` | One-minute sliding window. Matches most provider RPM rate limit windows. |
| `block` | `True` | Blocking mode: sleep until a slot opens. Appropriate for orchestration agents which have no other work to do while waiting. Set `False` in async systems where blocking a thread is unacceptable. |

The rate limiter uses a **sliding window**, not a fixed reset counter. A fixed counter allows a burst of 60 calls at second :59 followed by another 60 at second :01 — 120 calls in two seconds. The sliding window prevents this by tracking timestamps of the last 60 calls and blocking until the oldest expires.

### Bulkhead

| Parameter | Default | Rationale |
|---|---|---|
| `max_concurrent` | `10` | Default pool size. Appropriate for moderate-concurrency fan-out workflows. |
| `max_queue` | `20` | Callers waiting for a pool slot before `BulkheadFull` is raised. Setting `0` disables queuing entirely. |

Bulkhead is skipped by default (`skip_bulkhead=True`) in all orchestration calls. Enable by passing `skip_bulkhead=False` or by removing the skip flag for parallel-heavy patterns.

### Token Budget

| Parameter | Default | Rationale |
|---|---|---|
| `max_tokens_per_workflow` | `8,000` | For the supervisor orchestration reference implementation with 4 agents (3 specialists + synthesis). At GPT-4o pricing (~$0.005/1K input, ~$0.015/1K output), 8,000 tokens costs approximately $0.04–0.12 depending on input/output split. Adjust based on your cost ceiling. |
| `max_tokens_per_agent` | `3,000` | Per-agent ceiling. The workflow-level check is currently the enforced limit; this value is available for future per-agent enforcement. |

---

## 8. Adoption Guide — Adding Resilience to a New Workflow

This section describes the steps to add resilience to any new LangGraph workflow, whether under `scripts/handoff/`, `scripts/MAS_architectures/`, or a new directory.

### Step 1 — Assess the topology (see Section 2)

Answer two questions:
1. Does this workflow fan out to multiple concurrent agents? → Enable bulkhead and rate limiter burst protection.
2. Does this workflow iterate (loop)? → Enable token budget as a cost-based ceiling alongside iteration count guard.

### Step 2 — Wire the ResilientCaller

The minimum wiring — applicable to any topology — is a shared circuit breaker and a single `ResilientCaller`:

```python
from resilience import (
    CircuitBreakerRegistry, CircuitBreakerConfig,
    ResilientCaller, ResilienceConfig,
)

# Create once at module load (shared across all workflow executions in the process)
_shared_circuit_breaker = CircuitBreakerRegistry.get_or_create(
    "your_workflow_llm_api",
    CircuitBreakerConfig(fail_max=5, reset_timeout=60),
)

_workflow_resilient_caller = ResilientCaller(
    config=ResilienceConfig(),
    agent_name="your_workflow",
    circuit_breaker=_shared_circuit_breaker,
    # token_manager injected per-execution — see Step 3
)
```

### Step 3 — Inject the TokenManager per workflow execution

The `TokenManager` must be created fresh for each workflow execution (not at module load). The idiomatic LangGraph approach is to create it in the runner and inject it into the initial state:

```python
# In runner.py / your entry point
from resilience import TokenManager, TokenBudgetConfig

token_manager_for_this_run = TokenManager(
    TokenBudgetConfig(max_tokens_per_workflow=8_000)
)

initial_state = {
    "token_manager": token_manager_for_this_run,
    ...
}

result = graph.invoke(initial_state)

# After the workflow completes
print(token_manager_for_this_run.get_workflow_summary())
```

Add `token_manager: object | None` to the state `TypedDict`. No changes to the resilience module are required.

### Step 4 — Wrap each LLM call

In each node function, the pattern is: estimate → check (happens inside `caller.call`) → invoke → record.

```python
def your_agent_node(state: YourState) -> dict:
    prompt = build_prompt(state)
    
    # estimate for pre-check
    estimated_tokens = token_counter.count(prompt)
    
    try:
        response = _workflow_resilient_caller.call(
            llm.invoke,
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)],
            estimated_tokens=estimated_tokens,
            timeout=30.0,
        )
    except CircuitBreakerOpen:
        return {"result": None, "error": "circuit_open"}
    except TokenBudgetExceeded:
        return {"result": None, "error": "budget_exceeded"}
    except TimeoutExceeded:
        return {"result": None, "error": "timeout"}

    # record actual usage after the call
    if state.get("token_manager"):
        state["token_manager"].record_usage(
            agent_name="your_agent",
            tokens_in=response.usage_metadata.get("input_tokens", estimated_tokens),
            tokens_out=response.usage_metadata.get("output_tokens", 150),
        )

    return {"result": response.content, "error": None}
```

### Step 5 — Enable bulkhead for parallel topologies

If your workflow fans out to concurrent agents, enable the bulkhead by not passing `skip_bulkhead=True` (or explicitly passing `skip_bulkhead=False`):

```python
response = _workflow_resilient_caller.call(
    llm.invoke,
    messages,
    skip_bulkhead=False,  # Enable for parallel flows
    estimated_tokens=estimated_tokens,
)
```

Create separate `ResilientCaller` instances with different `BulkheadConfig` for agents with different priority levels:

```python
_high_priority_caller = ResilientCaller(
    config=ResilienceConfig(bulkhead=BulkheadConfig(max_concurrent=5)),
    agent_name="real_time_agent",
    circuit_breaker=_shared_circuit_breaker,
)

_batch_caller = ResilientCaller(
    config=ResilienceConfig(bulkhead=BulkheadConfig(max_concurrent=2)),
    agent_name="batch_agent",
    circuit_breaker=_shared_circuit_breaker,
)
```

The high-priority agent and the batch agent share the circuit breaker (same failure history) but have separate, bounded resource pools (independent bulkheads).

---

## References

| Resource | Purpose |
|---|---|
| `resilience/config.py` | All configuration dataclasses with field-level tuning documentation |
| `resilience/exceptions.py` | Exception hierarchy with `details` dict keys and recovery guidance per exception |
| `resilience/resilient_caller.py` | The façade class; execution stack order is documented in the module docstring |
| `resilience/circuit_breaker.py` | Circuit breaker state machine and `CircuitBreakerRegistry` multiton implementation |
| `resilience/token_manager.py` | `TokenManager` and `TokenCounter` with SRP rationale |
| `resilience/langgraph_integration_example.py` | Full working example: triage → specialist → summariser with shared breaker, token manager, per-node error handling, and fallback chain |
| `scripts/orchestration/_base/orchestrator.py` | Production integration point for all orchestration patterns |
| `scripts/orchestration/supervisor_orchestration/TOKEN_BUDGET_GUIDE.md` | Token budget integration walkthrough for the supervisor pattern |