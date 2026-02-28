# Resilience Patterns in LLM-Backed Multi-Agent Systems

---

## Preface

This document covers the theoretical basis, practical motivation, and implementation logic for the resilience layer used in LangGraph-based multi-agent systems. It is written for developers who have some familiarity with Python and LLM APIs, but who may not yet have built systems where multiple agents coordinate across shared infrastructure.

The goal is not merely to describe what each pattern does, but to build an understanding of the failure conditions that made each pattern necessary — and how the patterns interact with one another when composed into a single call stack.

---

## Table of Contents

1. How Multi-Agent Systems Fail
2. The Architecture of a Resilience Layer
3. Circuit Breaker
4. Rate Limiter
5. Retry Handler
6. Timeout Guard
7. Fallback Chain
8. Bulkhead
9. Token Budget
10. Composing Patterns — The ResilientCaller Facade
11. The Role of Typed Exceptions
12. Configuration as a First-Class Concern
13. Protocols and the Dependency Inversion Principle
14. Practical Execution Flow in a Multi-Agent System

---

## Chapter 1 — How Multi-Agent Systems Fail

### 1.1 The Single-Agent Assumption

When a developer first integrates an LLM API, they write something roughly like this:

```python
response = llm.invoke(prompt)
```

This works reliably in development. Failures are rare, requests are sequential, and when something goes wrong, rerunning the script is acceptable. The mental model is: one caller, one provider, one attempt.

Multi-agent systems violate every part of that model simultaneously.

### 1.2 What Changes in a Multi-Agent System

A multi-agent system is a coordinated set of LLM calls where the output of one agent becomes the input of the next, or where multiple agents operate concurrently and their outputs are merged. Consider a medical triage workflow:

```
User Query
    │
    ▼
┌──────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  Triage  │────►│  Pulmonology │     │  Cardiology   │     │  Nephrology  │
│  Agent   │     │  Specialist  │     │  Specialist   │     │  Specialist  │
└──────────┘     └──────┬───────┘     └──────┬────────┘     └──────┬───────┘
                        │                    │                      │
                        └────────────────────┴──────────────────────┘
                                             │
                                             ▼
                                    ┌────────────────┐
                                    │   Synthesis    │
                                    │    Agent       │
                                    └────────────────┘
```

Five LLM calls, all hitting the same external API. Several properties of this arrangement create failure modes that simply do not exist in single-agent code:

**Cascading failure.** If the API begins returning errors, all five agents will eventually encounter them. Without any circuit-breaking logic, each agent independently goes through its full timeout cycle — typically 30–60 seconds — before giving up. Five agents times 30 seconds is 150 seconds of blocked execution, even though the API was clearly down after the first failure.

**Quota exhaustion.** The same API quota is shared across all agents. A burst of parallel calls from three specialists simultaneously can exhaust per-minute rate limits even when the total request count for the hour is well within the allowed range. The API returns 429 errors not because the system is over quota globally, but because it issued too many calls in too short a window.

**Cost amplification.** Token consumption is additive. One workflow with five agents, each consuming 500 tokens on average, costs five times what a single-agent call would. If one agent enters a reflection loop — generating output, critiquing it, regenerating — the cost multiplier grows further. Without enforced budgets, a single problematic workflow can consume the budget intended for hundreds.

**Resource contention.** When multiple workflows run concurrently, their agents compete for shared resources: thread pool slots, connection pool entries, rate-limit quota. A batch job agent that spawns 20 concurrent LLM calls can starve a real-time user-facing agent of the resources it needs to respond within an acceptable latency.

### 1.3 The Failure Propagation Model

Understanding how failures propagate through a multi-agent pipeline is a prerequisite for understanding why each resilience pattern is positioned where it is.

There are two propagation models:

**Sequential propagation.** In a linear pipeline (Agent A → Agent B → Agent C), a failure in Agent A means Agent B never receives valid input. If Agent B does not check for upstream failure before invoking the LLM, it will make an LLM call with a corrupt or empty input, receive a confused response, pass that downstream to Agent C, and eventually produce a final output that looks like a response but contains meaningless content. No exception was raised; the system appears to have succeeded.

**Parallel propagation.** In a fan-out pattern (Triage → [Specialist A | Specialist B | Specialist C] → Synthesis), a shared resource failure hits all branches simultaneously. If the rate limiter is not shared, all three specialists may independently exhaust the same API quota window before any of them has had a chance to back off.

The resilience patterns in this codebase are designed to interrupt these propagation paths — either by catching the failure early and converting it to a structured exception, or by preventing the conditions that cause failure in the first place.

---

## Chapter 2 — The Architecture of a Resilience Layer

### 2.1 Patterns as Concentric Guards

The resilience layer is composed of seven distinct patterns. Rather than applying them in an ad hoc order, they are arranged as concentric guards around the actual LLM call. Each guard has a specific scope: it checks one condition, and if that condition is violated, it raises a specific, typed exception.

```
┌─ Token Budget Check ──────────────────────────────────────────────┐
│  Checks: Would this call exceed the workflow token budget?        │
│  Raises: TokenBudgetExceeded                                      │
│                                                                   │
│  ┌─ Bulkhead ────────────────────────────────────────────────┐   │
│  │  Checks: Is there a free slot in the concurrency pool?    │   │
│  │  Raises: BulkheadFull                                     │   │
│  │                                                           │   │
│  │  ┌─ Rate Limiter ──────────────────────────────────────┐  │   │
│  │  │  Checks: Are we within quota for this time window?  │  │   │
│  │  │  Raises: RateLimitExhausted (or blocks)             │  │   │
│  │  │                                                     │  │   │
│  │  │  ┌─ Circuit Breaker ────────────────────────────┐   │  │   │
│  │  │  │  Checks: Is the service known-healthy?       │   │  │   │
│  │  │  │  Raises: CircuitBreakerOpen                  │   │  │   │
│  │  │  │                                             │   │  │   │
│  │  │  │  ┌─ Retry Handler ──────────────────────┐  │   │  │   │
│  │  │  │  │  Catches: transient exceptions       │  │   │  │   │
│  │  │  │  │  Waits:   exponential backoff+jitter │  │   │  │   │
│  │  │  │  │                                     │  │   │  │   │
│  │  │  │  │  ┌─ Timeout Guard ───────────────┐  │  │   │  │   │
│  │  │  │  │  │  Enforces: deadline per call  │  │  │   │  │   │
│  │  │  │  │  │  Raises:   TimeoutExceeded    │  │  │   │  │   │
│  │  │  │  │  │                              │  │  │   │  │   │
│  │  │  │  │  │     llm.invoke(prompt)       │  │  │   │  │   │
│  │  │  │  │  │                              │  │  │   │  │   │
│  │  │  │  │  └──────────────────────────────┘  │  │   │  │   │
│  │  │  │  └─────────────────────────────────────┘  │   │  │   │
│  │  │  └────────────────────────────────────────────┘   │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
         After call: record_usage(agent_name, tokens_in, tokens_out)
```

### 2.2 Why This Specific Order

The ordering is not arbitrary. Each position in the stack reflects the cost of the check and the appropriate scope of its enforcement.

**Token budget is outermost** because it is the cheapest check — pure arithmetic against an in-memory counter — and it protects against wasted work. If the budget is already exhausted, there is no reason to acquire a bulkhead slot, consume a rate-limit token, or warm up a connection.

**Bulkhead is second** because concurrency isolation should happen before the caller begins consuming any shared quota. If the pool is full, the caller should be turned away before it claims a rate-limit slot. Claiming a slot only to then block waiting for a pool entry would waste quota.

**Rate limiter is third** because quota should be consumed only when the call has a realistic chance of succeeding. Consuming a rate-limit slot and then immediately getting a CircuitBreakerOpen exception would be a slot wasted.

**Circuit breaker is fourth** because it prevents calls to known-down services. When the circuit is open, failing fast costs almost nothing — a dictionary lookup and an exception raise, taking less than a millisecond. This is far preferable to waiting for a connection timeout.

**Retry wraps the real call** because retries must account for multiple attempts, each of which should go through the circuit breaker and timeout. If retry were outside the circuit breaker, a series of retries on a circuit-open condition would be meaningless — all of them would be rejected instantly anyway.

**Timeout is innermost** because the deadline applies to each individual attempt, not to the total retry sequence. A three-retry configuration with a 30-second timeout means each of the three attempts has a 30-second deadline.

### 2.3 The Single Entry Point

The `ResilientCaller` class acts as a facade over all seven patterns. Its purpose is to allow every LangGraph node to make a single function call without needing to know which patterns exist or in what order they apply:

```python
# Without facade — the caller must know and wire everything manually
rate_limiter.acquire()
if circuit_breaker.is_open:
    raise CircuitBreakerOpen(...)
response = retry_handler.call_with_retry(
    lambda: timeout_guard.call_with_timeout(llm.invoke, prompt)
)
token_manager.record_usage(...)

# With facade — the caller knows only about the call
response = caller.call(llm.invoke, prompt)
```

This is the Facade design pattern. The caller's code is insulated from the composition details. If the composition changes — for example, if a new pattern is added, or the order changes — the node code does not need to be updated.

---

## Chapter 3 — Circuit Breaker

### 3.1 The Problem It Solves

When an external service starts failing, the naive behavior of any caller is to keep trying. Every new request goes through the full timeout cycle before the caller accepts that the call failed. In a single-agent system, this means one user waits 30 seconds. In a five-agent pipeline where all agents share the same service, this means five sequential 30-second timeouts if the agents run one after another, or a single 30-second wait if they run in parallel — but with five times the API calls being made against an already-struggling service.

The circuit breaker solves this by maintaining state about the health of the downstream service. Once enough consecutive failures have been observed, the breaker moves to a state where it immediately rejects all calls without actually making them.

### 3.2 The State Machine

A circuit breaker is, at its core, a state machine with three states:

```
                  fail_max consecutive failures
  ┌─────────┐   ──────────────────────────────►   ┌──────────┐
  │  CLOSED │                                      │   OPEN   │
  │(healthy)│   ◄──────────────────────────────   │  (down)  │
  └─────────┘     test call succeeds               └────┬─────┘
       ▲                                                │
       │           ┌────────────┐                      │ reset_timeout seconds
       └───────────│ HALF-OPEN  │◄─────────────────────┘
    test succeeded │ (testing)  │
                   └────────────┘
                         │
                         │ test call fails
                         ▼
                   re-opens (OPEN)
```

**CLOSED state.** All calls pass through normally. Each failure is counted. When the failure count reaches `fail_max`, the breaker transitions to OPEN. A success resets the counter.

**OPEN state.** All calls are rejected immediately with a `CircuitBreakerOpen` exception. No actual API call is made. The breaker stays open for `reset_timeout` seconds, then moves to HALF-OPEN.

**HALF-OPEN state.** Exactly one test call is allowed through. If it succeeds, the breaker closes and normal operation resumes. If it fails, the breaker reopens and the `reset_timeout` timer restarts.

### 3.3 The Multiton Registry Pattern

A critical detail in multi-agent systems is that all agents calling the same downstream service must share the same circuit breaker instance. If each agent creates its own breaker, the failure state is not shared — Agent A might see five consecutive failures and open its breaker, while Agent B, with its own fresh breaker, keeps making calls to the same failing service.

The `CircuitBreakerRegistry` addresses this through the multiton pattern, which is a generalization of the singleton: instead of one instance per class, there is one instance per named key.

```python
# Both of these return the exact same CircuitBreaker object
breaker_used_by_triage     = CircuitBreakerRegistry.get_or_create("openai_api")
breaker_used_by_specialist = CircuitBreakerRegistry.get_or_create("openai_api")

# Therefore: when triage causes 5 failures, the specialist's next call
# is immediately rejected without making an API call
```

The key is the service name, not the agent name. Every agent that calls the same API shares a breaker keyed to that API, regardless of which agent it is.

### 3.4 Interaction with the Fallback Chain

The circuit breaker becomes even more powerful when combined with the fallback chain (discussed in Chapter 7). When a provider's circuit is open, the fallback chain recognizes the `CircuitBreakerOpen` exception as a signal to skip that provider without waiting. The skip costs less than a millisecond, compared to the 30-second timeout that would otherwise be required to determine that the provider is unavailable.

```
FallbackChain.call(prompt)
    │
    ├── Try Provider 1 (gpt4o)
    │     └── CircuitBreaker is OPEN → catch CircuitBreakerOpen
    │             └── append to errors, skip to next
    │
    ├── Try Provider 2 (claude)
    │     └── call succeeds → return result
    │
    └── (Provider 3 never tried)
```

Total time: ~1ms for the circuit-open rejection + actual latency of Provider 2.

---

## Chapter 4 — Rate Limiter

### 4.1 The Problem It Solves

LLM API providers enforce quotas on how many requests can be made within a time window. A typical restriction might be 60 requests per minute. In a single-agent system, this limit is rarely reached because calls are sequential and each takes several seconds. In a multi-agent system, this calculation changes dramatically.

Consider a voting pattern where five specialist agents are invoked in parallel:

```
t=0.0s: Specialist A calls llm.invoke()
t=0.0s: Specialist B calls llm.invoke()
t=0.0s: Specialist C calls llm.invoke()
t=0.0s: Specialist D calls llm.invoke()
t=0.0s: Specialist E calls llm.invoke()
```

Five calls at the same instant. Even if the hourly quota is not close to exhausted, a burst of five simultaneous calls may exceed a per-second or per-10-second sub-limit that the provider enforces. The result is 429 errors — "too many requests" — even though the system is technically within its minute-level quota.

The rate limiter smooths this burst. Rather than all five calls reaching the API simultaneously, they are spaced out across the time window so the per-second sub-limits are respected.

### 4.2 The Sliding Window Algorithm

A rate limiter can be implemented in several ways. The simplest is a fixed counter that resets at the top of each minute. This has a flaw: a burst of 60 calls at second :59 followed by another 60 calls at second :01 of the next minute is technically within quota, but delivers 120 calls in a two-second window.

The sliding window algorithm avoids this by looking backward in time from the current moment, not forward to the next reset:

```
Current time: t=62s
Window size: 60s
Stored timestamps: [t=5s, t=12s, t=20s, ..., t=58s, t=60s, t=61s]

Step 1: Remove all timestamps where (now - t) >= 60s
        → Remove t=5s (62-5=57 — wait, 57 < 60, keep)
        → Actually remove timestamps where t < (62-60) = t < 2s
        → None to remove in this example

Step 2: Count remaining timestamps
        → If count < max_calls, admit the call

Step 3: If count >= max_calls
        → Oldest timestamp is t=5s
        → Sleep until (5s + 60s) = t=65s, i.e., sleep 3 seconds
        → After sleeping, re-check and admit
```

The "oldest timestamp tells us when the next slot opens" insight is the key to the sliding window: the slot that was consumed earliest will be the first one to expire, and its expiry time is exactly `oldest_timestamp + period`.

### 4.3 Blocking vs. Non-blocking Mode

The rate limiter supports two modes, controlled by `config.block`:

In **blocking mode** (the default), when the quota is exhausted, the limiter sleeps until a slot opens. The caller's thread blocks. This is appropriate for synchronous orchestration where the agent simply needs to wait its turn and does not have other work to do in the meantime.

In **non-blocking mode**, when the quota is exhausted, a `RateLimitExhausted` exception is raised immediately. This is appropriate for asynchronous systems where blocking a thread is unacceptable — instead, the caller catches the exception and schedules a retry through an async queue.

### 4.4 Thread Safety

In a multi-agent system, multiple agents may call `acquire()` simultaneously from different threads. Without proper synchronization, two agents could both observe "one slot remaining," both record their timestamp, and both proceed — admitting two calls when only one slot was available. The `threading.Lock` in the implementation ensures that the check-and-record operation is atomic.

---

## Chapter 5 — Retry Handler

### 5.1 The Problem It Solves

Not all API failures are permanent. A connection that times out due to brief network congestion, a rate-limit error from a momentary burst, or a 503 from a server under brief overload — all of these are transient conditions that typically resolve within seconds. Making the caller handle these conditions manually would mean every LangGraph node needs its own retry loop, its own backoff logic, and its own decision about which errors to retry.

The retry handler centralizes this logic. It wraps a callable, catches a defined set of transient exceptions, waits an appropriate amount of time, and re-executes the callable — up to a configurable limit.

### 5.2 The Critical Distinction: Transient vs. Permanent Errors

The most important concept in retry logic is the distinction between errors that will resolve on their own and errors that will not.

| Error Type | Example | Retry? | Reasoning |
|---|---|---|---|
| `ConnectionError` | Network blip | Yes | Will likely resolve |
| `TimeoutError` | Server overloaded | Yes | Overload is temporary |
| `RateLimitError` | 429 from API | Yes | Quota window will reset |
| `AuthenticationError` | Invalid API key | No | Retrying won't fix the key |
| `InvalidRequestError` | Malformed prompt | No | Retrying same prompt gives same error |
| `ContentPolicyError` | Policy violation | No | Policy won't change between retries |

Retrying permanent errors wastes quota, adds latency, and fills logs with noise. The `TRANSIENT_EXCEPTIONS` tuple in `retry_handler.py` is the single authoritative list of which exception types qualify for retry. When a new LLM provider is integrated, its transient exception types are added to this list — not scattered across each agent that uses it.

### 5.3 Exponential Backoff with Jitter

When a retry is warranted, the question is how long to wait before the next attempt. Retrying immediately is usually counterproductive — if the API was overloaded one millisecond ago, it is likely still overloaded now.

Exponential backoff increases the wait time multiplicatively between attempts:

```
Attempt 1 (initial call): executes immediately
Attempt 2 (retry 1):      wait ~1s
Attempt 3 (retry 2):      wait ~2s
Attempt 4 (retry 3):      wait ~4s

Formula: wait = min(initial_wait × 2^(attempt-1), max_wait)
```

This gives the service increasing amounts of time to recover. The `max_wait` cap prevents the backoff from growing to impractical values.

Jitter adds a random component to each wait period. Without jitter, if 10 agents all encounter a rate limit at the same moment, they will all wait exactly 1 second and then all retry simultaneously — a "thundering herd" that creates a new rate-limit burst. With jitter, the 10 agents wait different amounts (0.9s, 1.3s, 1.1s, 1.7s...) and their retries are spread across the recovery window.

```
Without jitter (thundering herd):
t=0.0  ████████████████████████████  10 calls, all rate-limited
t=1.0  ████████████████████████████  10 retries, simultaneous, all rate-limited again
t=2.0  ████████████████████████████  10 retries, simultaneous, all rate-limited again

With jitter (spread):
t=0.0  ████████████████████████████  10 calls, all rate-limited
t=0.9  ██                            1 retry (earliest jitter)
t=1.1  ████                          2 retries
t=1.3  ██                            1 retry
t=1.7  ████                          2 retries
t=2.0  ████████                      4 retries — spread, fewer collisions
```

### 5.4 Position in the Stack

The retry handler wraps the circuit breaker and timeout together, not just the raw API call. This is a deliberate choice. The retry handler should retry at the level of "protected call," not at the level of "raw call":

```python
# Correct: retry wraps the circuit-breaker-protected call
retry_handler.call_with_retry(
    lambda: circuit_breaker.call(
        lambda: timeout_guard.call_with_timeout(llm.invoke, prompt)
    )
)

# Incorrect: retry bypasses the circuit breaker
circuit_breaker.call(
    lambda: retry_handler.call_with_retry(
        lambda: timeout_guard.call_with_timeout(llm.invoke, prompt)
    )
)
```

In the correct arrangement, each retry attempt goes through the circuit breaker. If the first attempt fails and opens the circuit, the second retry attempt immediately receives `CircuitBreakerOpen` — preventing further wasted calls and allowing the breaker to do its job.

---

## Chapter 6 — Timeout Guard

### 6.1 The Problem It Solves

LLM providers occasionally hang. A request reaches the server but receives no response — not an error, not a timeout at the HTTP layer, just silence. Without application-level timeout enforcement, the calling thread blocks indefinitely.

In a sequential pipeline, one hanging agent blocks every downstream agent. In a concurrent pipeline, hanging agents accumulate in the thread pool, eventually exhausting all available threads. No new requests can be processed until the hung calls eventually fail at the HTTP transport level, which may take minutes.

The timeout guard enforces a hard deadline: if the call does not return within a configured number of seconds, execution is interrupted and `TimeoutExceeded` is raised. The caller can then decide how to respond — retry with the same or different provider, return a partial result, or abort the workflow.

### 6.2 Why Signal-Based Timeouts Do Not Work Here

The standard Unix approach to timeouts is `signal.alarm`, which delivers a SIGALRM after a specified number of seconds. This approach fails in multi-agent systems for two reasons:

First, `signal.alarm` can only be set from the main thread. LangGraph nodes typically execute in worker threads managed by the graph's async engine. A call to `signal.alarm` inside a worker thread either fails silently or raises an exception immediately.

Second, `signal.alarm` is POSIX-only and does not work on Windows.

The thread-based approach used here submits the call to a background thread and uses `Future.result(timeout=deadline)` to wait with a deadline. If the deadline passes, the future is cancelled and `TimeoutExceeded` is raised. The background thread may continue running for some time after the cancellation signal (Python threads cannot be forcibly terminated), but the calling code has already been released to handle the timeout.

### 6.3 Layered Timeout Strategy

The timeout guard does not replace the HTTP transport timeout configured in the LLM client SDK. Rather, the two operate as complementary layers with deliberately different deadlines:

```
HTTP transport timeout (in LLM SDK):   60s   ← last resort; unstructured error
TimeoutGuard (application layer):      30s   ← primary; raises TimeoutExceeded
LangGraph node-level timeout:          45s   ← intermediate; rarely triggered
```

The application-layer timeout fires first, producing a structured `TimeoutExceeded` exception with known fields (`agent_name`, `timeout_secs`, `elapsed_secs`). The HTTP layer timeout is a backstop for cases where the application timeout itself fails — for example, if the thread pool is saturated and the timeout enforcement thread cannot be scheduled in time.

### 6.4 Per-Call Timeout Override

Different agents in the same workflow may have different latency requirements. A triage agent that the user is waiting on in real time should have a shorter deadline than a batch summarization agent running in the background. The `timeout` parameter on `call_with_timeout` allows per-call overrides:

```python
# Triage: user is waiting, short deadline
timeout_guard.call_with_timeout(llm.invoke, prompt, timeout=10.0)

# Background synthesis: quality matters more than speed, longer deadline  
timeout_guard.call_with_timeout(llm.invoke, prompt, timeout=60.0)
```

The `ResilientCaller.call()` method exposes this as a top-level parameter, so callers can adjust the deadline without bypassing the rest of the resilience stack.

---

## Chapter 7 — Fallback Chain

### 7.1 The Problem It Solves

Every LLM provider has an availability SLA — typically 99.9%, which translates to roughly 8.7 hours of downtime per year. For a system that relies on a single provider, that provider's downtime is the system's downtime. There is no architectural feature in single-provider systems that can recover from a provider-side outage.

A fallback chain addresses this by defining an ordered list of providers. When the primary provider fails — for any reason — the chain automatically attempts the next provider in the list. The caller receives a response without needing to know which provider produced it.

### 7.2 Chain of Responsibility

The pattern used here is Chain of Responsibility: a series of handlers, each of which either handles a request or passes it to the next handler. The providers in the fallback chain are the handlers. Each provider either returns a result (handles the request) or raises an exception (passes to the next handler).

```
FallbackChain.call(prompt)
    │
    ├─ Provider 1: openai_gpt4o
    │      call attempt → ConnectionError
    │      catch exception, append to errors
    │      │
    │      └─ Provider 2: anthropic_claude
    │             call attempt → success
    │             return result ──────────────────────────► caller
    │
    └─ (Provider 3 never reached)
```

The chain is ordered by a `weight` field on each provider. Lower weight means higher priority. The chain sorts providers by weight before iterating, so callers do not need to supply them in order.

### 7.3 What to Catch vs. What to Pass Through

Not all exceptions should cause the chain to try the next provider. There are two categories:

**Provider failures** — errors caused by the provider being unavailable, overloaded, or rate-limited. These should trigger fallthrough: `ConnectionError`, `RateLimitError`, `TimeoutError`, `CircuitBreakerOpen`. The next provider is not subject to the same conditions and may succeed.

**Input errors** — errors caused by the request itself being invalid. A content policy violation means the prompt was rejected. A `BadRequestError` means the request was malformed. These errors will produce the same result on every provider — they are caused by the input, not the provider. Falling through wastes quota on attempts that are guaranteed to fail.

The `non_fallback_exceptions` parameter on `FallbackChain` accepts a tuple of exception types that should propagate immediately without attempting the next provider:

```python
from openai import BadRequestError

chain = FallbackChain(
    providers=[gpt4o_provider, claude_provider, gemini_provider],
    non_fallback_exceptions=(BadRequestError,),
)
```

### 7.4 The Relationship Between Fallback Chain and Circuit Breaker

These two patterns are frequently confused because both respond to provider failures. They serve different purposes and operate at different timescales.

The circuit breaker addresses **latency under failure**: once a provider has failed enough times to be considered unhealthy, subsequent calls are rejected in under a millisecond rather than waiting 30 seconds for a timeout. It is a performance optimization for a known-bad service.

The fallback chain addresses **availability under failure**: when any provider fails (whether detected quickly by a circuit breaker or slowly by a timeout), the chain tries the next alternative. It is a routing mechanism for finding a working provider.

Used together, a circuit breaker per provider allows the fallback chain to skip known-bad providers instantly:

```
FallbackChain.call(prompt)
    │
    ├─ Provider 1: openai
    │      CircuitBreaker state: OPEN
    │      → CircuitBreakerOpen raised
    │      → catch, skip in <1ms
    │
    ├─ Provider 2: anthropic
    │      CircuitBreaker state: CLOSED
    │      call attempt → success
    │      return result
```

Without the circuit breaker, skipping Provider 1 would require waiting for a connection timeout — 30 seconds — before concluding the provider is unavailable.

---

## Chapter 8 — Bulkhead

### 8.1 The Problem It Solves

The name "bulkhead" comes from naval architecture. A ship's hull is divided by sealed walls (bulkheads) into watertight compartments. If one compartment floods, the bulkheads prevent the water from spreading to other compartments, preserving the ship's ability to stay afloat.

In a multi-agent system, the analogous problem is resource starvation. A shared thread pool, connection pool, or semaphore serves all agents. If one agent type — say, a batch summarization job — issues 20 concurrent LLM calls, it can claim all 20 available thread pool slots. A real-time triage agent that needs one thread to serve a user request finds the pool exhausted and must wait.

The bulkhead pattern assigns a bounded resource pool to each agent type. The batch job's pool is capped at 5 concurrent calls; the triage pool is capped at 5 as well. Even if the batch job fills its pool completely, 5 slots remain available exclusively for triage.

### 8.2 The Semaphore Primitive

The bulkhead is implemented using a counting semaphore. A semaphore maintains an internal counter that tracks how many "slots" are available. Two operations modify the counter:

- `acquire()` decrements the counter. If the counter is already zero, the call blocks until another thread calls `release()`.
- `release()` increments the counter, potentially unblocking a waiting `acquire()` call.

Initialized with `max_concurrent = 5`, the semaphore permits up to five simultaneous acquires before blocking:

```
Initial state: [5 slots available]

Agent A acquires: [4 slots available]
Agent B acquires: [3 slots available]
Agent C acquires: [2 slots available]
Agent D acquires: [1 slot available]
Agent E acquires: [0 slots available]
Agent F tries to acquire: BLOCKS (waits up to queue_timeout seconds)

Agent A releases: [1 slot available]
Agent F unblocks: [0 slots available]
```

### 8.3 Queue vs. Immediate Rejection

When all slots are taken, the bulkhead can either queue the caller (wait for a slot to open) or reject immediately. The choice depends on whether queuing is appropriate for the workload.

For a real-time triage agent, queuing is appropriate: the user is waiting, so it is better to wait 2 seconds for a slot than to reject the request outright.

For a batch job under heavy load, immediate rejection (with a short retry) may be preferable. Queueing 100 batch jobs and processing them slowly provides a worse user experience than rejecting some with a "retry in a moment" signal.

The `max_queue` parameter controls the maximum number of callers that can wait simultaneously. Once that queue depth is reached, new callers receive `BulkheadFull` immediately.

### 8.4 When to Enable vs. Skip the Bulkhead

The bulkhead is most useful in systems where multiple agent types share a process and compete for the same underlying resources. In a linear sequential pipeline — Triage → Specialist → Synthesis, one at a time — the bulkhead adds overhead (semaphore acquisition) without providing isolation benefit, because there is no concurrent competition for resources.

The `skip_bulkhead` flag in `ResilientCaller.call()` supports this: linear orchestration flows can skip the bulkhead, while parallel fan-out flows (voting patterns, parallel specialist invocations) should enable it.

---

## Chapter 9 — Token Budget

### 9.1 The Problem It Solves

LLM API costs are proportional to token consumption — the sum of input tokens (the prompt) and output tokens (the response). In single-agent systems, the cost per call is roughly predictable and bounded by the length of the input prompt and the model's typical response length.

In multi-agent systems, several dynamics make costs unpredictable:

**Accumulation across agents.** Five agents each consuming 500 tokens equals 2,500 tokens per workflow invocation, not 500. If 100 users invoke the workflow concurrently, that is 250,000 tokens in one minute.

**Reflection loops.** An agent architecture that generates output, critiques that output, and regenerates can loop many times. Without a budget, a loop configured for "up to 10 iterations" might run all 10, consuming 10 times the expected tokens.

**Verbose prompts.** Agents that pass their full context forward through the pipeline accumulate longer and longer prompts. Agent C's prompt might include Agent A's output, Agent B's output, and system instructions — far longer than any individual agent's prompt.

The token manager tracks cumulative token consumption across all agents in a single workflow execution and raises `TokenBudgetExceeded` before making a call that would breach the configured limit.

### 9.2 Check Before, Record After

The usage lifecycle has two steps:

```python
# 1. Before the call: check if the budget allows it
manager.check_budget(agent_name="specialist", estimated_tokens=300)

# 2. Make the call
response = llm.invoke(prompt)

# 3. After the call: record what was actually consumed
manager.record_usage(
    agent_name="specialist",
    tokens_in=response.usage_metadata["input_tokens"],
    tokens_out=response.usage_metadata["output_tokens"],
)
```

The pre-call check uses an estimate. This is deliberate: the exact token count is only known after the call completes (the API returns it in the response metadata). The estimate is produced by `TokenCounter`, which uses the tiktoken library to tokenize the prompt text and count the result. This estimate is accurate for the input but cannot account for output length.

The post-call record updates the running totals with actual values, maintaining an accurate cumulative count for subsequent checks.

### 9.3 Separation of Concerns — TokenCounter and TokenManager

The original code had `TokenManager.count_tokens()` — a token counting method on the budget tracking class. These are two distinct responsibilities that change for different reasons:

`TokenManager` changes when budget enforcement rules change: when the budget limit is revised, when per-agent quotas are introduced, when the granularity of tracking changes from per-workflow to per-session.

`TokenCounter` changes when the tokenization strategy changes: when a new model uses a different tokenizer, when provider-reported counts (from the API response) replace local estimates, when a more efficient counting library becomes available.

Keeping them separate means either can be changed independently. You can swap from tiktoken-based estimation to provider-reported actuals in `TokenCounter` without touching `TokenManager`'s budget enforcement logic.

### 9.4 One Manager Per Workflow Execution

A `TokenManager` instance maintains a running total. This total must be scoped to a single workflow execution — not shared across executions.

The correct pattern is to create a fresh `TokenManager` at the start of each workflow invocation and pass it through LangGraph state so all nodes in that execution share it:

```python
def create_initial_state(user_query: str) -> WorkflowState:
    return WorkflowState(
        user_query=user_query,
        token_manager=TokenManager(TokenBudgetConfig()),  # fresh per execution
        ...
    )
```

If the same instance were reused across executions, the cumulative token count would grow unboundedly, causing the budget to be exceeded for all executions after the first few.

---

## Chapter 10 — Composing Patterns — The ResilientCaller Facade

### 10.1 The Need for Composition

Each pattern described in the preceding chapters solves a distinct problem. But their value is not fully realized in isolation — it comes from their interaction. The circuit breaker can only fast-fail the fallback chain if the chain catches `CircuitBreakerOpen`. The retry handler only prevents thundering herds if jitter is applied correctly. The token budget only prevents runaway costs if it is checked before the bulkhead slot is acquired.

Composing these patterns correctly requires understanding their interdependencies. Requiring every LangGraph node author to understand and implement this composition correctly is unreliable. A single node that wires the patterns in the wrong order undermines the system's reliability guarantees.

The `ResilientCaller` class encodes the correct composition once, centrally, and exposes a single method — `call()` — that applies all patterns in the right order.

### 10.2 Call Sequence for a Single LLM Invocation

The following sequence diagram traces a single call through the full stack:

```
Node Code             ResilientCaller       Components
─────────              ───────────────       ──────────
  │                         │                    │
  │  caller.call(            │                    │
  │    llm.invoke,           │                    │
  │    prompt,               │                    │
  │    estimated_tokens=300  │                    │
  │  )                       │                    │
  │ ─────────────────────── ►│                    │
  │                          │                    │
  │                          │  token_manager.    │
  │                          │  check_budget()   ─┤──► TokenManager
  │                          │  [budget OK]       │
  │                          │                    │
  │                          │  bulkhead.         │
  │                          │  acquire()        ─┤──► Semaphore
  │                          │  [slot acquired]   │
  │                          │                    │
  │                          │  rate_limiter.     │
  │                          │  acquire()        ─┤──► Sliding Window
  │                          │  [slot acquired]   │
  │                          │                    │
  │                          │  retry_handler.    │
  │                          │  call_with_retry() │
  │                          │  ─────────────────►│──► Attempt 1
  │                          │                    │
  │                          │                    │  circuit_breaker.call()
  │                          │                    │──► CircuitBreaker [CLOSED]
  │                          │                    │
  │                          │                    │  timeout_guard.
  │                          │                    │  call_with_timeout()
  │                          │                    │──► ThreadPoolExecutor
  │                          │                    │
  │                          │                    │  llm.invoke(prompt)
  │                          │                    │──► LLM API ──► response
  │                          │                    │
  │                          │  [success,         │
  │                          │   return response] │
  │◄─────────────────────── ─│                    │
  │                          │                    │
  │  token_manager.          │                    │
  │  record_usage()          │                    │
  │ ─────────────────────── ►│──────────────────►─┤──► TokenManager
  │                          │                    │
```

### 10.3 Call Sequence Under Failure

Now trace the same call when the API is experiencing issues — five consecutive failures have opened the circuit breaker:

```
Node Code             ResilientCaller       Components
─────────              ───────────────       ──────────
  │                         │                    │
  │  caller.call(...)        │                    │
  │ ─────────────────────── ►│                    │
  │                          │                    │
  │                          │  check_budget()    │
  │                          │  [budget OK]       │
  │                          │                    │
  │                          │  bulkhead.acquire()│
  │                          │  [slot acquired]   │
  │                          │                    │
  │                          │  rate_limiter.     │
  │                          │  acquire()         │
  │                          │  [slot acquired]   │
  │                          │                    │
  │                          │  retry_handler.    │
  │                          │  call_with_retry() │
  │                          │  ─────────────────►│──► Attempt 1
  │                          │                    │
  │                          │                    │  circuit_breaker.call()
  │                          │                    │──► CircuitBreaker [OPEN]
  │                          │                    │    └─ raises CircuitBreakerError
  │                          │                    │
  │                          │  [CircuitBreakerOpen                     │
  │                          │   not in TRANSIENT_EXCEPTIONS]           │
  │                          │  → no retry, propagate up                │
  │                          │                    │
  │◄── CircuitBreakerOpen ───│                    │
  │                          │                    │
  │  except CircuitBreakerOpen:                   │
  │      return cached_fallback()                 │
  │                          │                    │
```

Notice that `CircuitBreakerOpen` is not in `TRANSIENT_EXCEPTIONS` — it is a permanent condition (until the circuit resets itself) and should not be retried. The retry handler propagates it immediately.

### 10.4 Shared vs. Per-Caller Components

Some components should be shared across all agents in a workflow; others should be per-agent. The `ResilientCaller` constructor accepts optional shared instances for components that need cross-agent coordination:

| Component | Scope | Reason |
|---|---|---|
| `CircuitBreaker` | Per-service (shared) | All agents calling the same API must share failure state |
| `TokenManager` | Per-workflow (shared) | Budget must be tracked cumulatively across all agents |
| `RateLimiter` | Per-caller (private) | Each agent's quota contribution is tracked separately |
| `RetryHandler` | Per-caller (private) | Retry config may differ per agent |
| `TimeoutGuard` | Per-caller (private) | Timeout may differ per agent |
| `Bulkhead` | Per-pool (configurable) | Pool boundaries should reflect resource allocation intent |

---

## Chapter 11 — The Role of Typed Exceptions

### 11.1 Generic Exceptions in Multi-Agent Systems

When a resilience layer raises a generic `Exception` or `RuntimeError`, the calling code faces an impossible task: it must parse the error message string to determine what went wrong and how to respond. This is fragile, locale-dependent, and breaks whenever the error message format changes.

In a multi-agent system, the orchestrator must make different decisions depending on the failure type:

- `CircuitBreakerOpen` → return a cached response from a previous successful call
- `TokenBudgetExceeded` → return a partial result based on agents that already completed
- `RateLimitExhausted` → schedule a retry after the `retry_after` delay
- `BulkheadFull` → shed load with a brief jitter delay and retry
- `TimeoutExceeded` → log the timeout and continue with remaining agents
- `AllFallbacksFailed` → return a static error message and alert on-call

These are six distinct recovery strategies. A single `except Exception` handler cannot implement all of them correctly.

### 11.2 Structured Exception Hierarchy

All resilience exceptions inherit from `ResilienceError`, which carries a `details` dict:

```
ResilienceError (base)
    message:  str             — human-readable description
    details:  dict[str, Any]  — structured context for programmatic use

├── CircuitBreakerOpen
│     details: breaker_name, fail_count, state

├── RateLimitExhausted  
│     details: limiter_name, limit, period, retry_after

├── TimeoutExceeded
│     details: agent_name, timeout_secs, elapsed_secs

├── AllFallbacksFailed
│     details: providers_tried, last_error, errors

├── BulkheadFull
│     details: pool_name, max_concurrent, current_active

└── TokenBudgetExceeded
      details: scope, agent_name, used, limit, estimated_add
```

The `details` dict allows the orchestrator to log structured context:

```python
except CircuitBreakerOpen as e:
    logger.warning("circuit_breaker_open", extra=e.details)
    # Logs: {"breaker_name": "openai_api", "fail_count": 5, "state": "open"}
    return cached_fallback_response()
```

This is vastly more useful than `"circuit breaker 'openai_api' is OPEN"` when querying logs after an incident.

---

## Chapter 12 — Configuration as a First-Class Concern

### 12.1 The Problem with Scattered Settings

In the original code, each resilience module read from a shared `settings` object directly:

```python
# In circuit_breaker.py
fail_max = settings.circuit_breaker_fail_max

# In rate_limiter.py  
max_calls = settings.rate_limit_calls_per_minute

# In token_manager.py
max_tokens = settings.max_tokens_per_agent_call
```

This creates two problems. First, coupling: each resilience class depends on the global `settings` object. Writing a unit test for `CircuitBreaker` requires mocking `settings`, even though the test only cares about the circuit breaker logic itself.

Second, discoverability: there is no single place where a developer can see all the configurable parameters. They must search multiple files. When a new developer asks "what controls the retry wait time?", the answer requires a grep.

### 12.2 The Configuration Dataclass

The `ResilienceConfig` dataclass is a plain, immutable data object that collects all configuration in one place:

```python
@dataclass(frozen=True)
class ResilienceConfig:
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    rate_limiter:    RateLimiterConfig    = field(default_factory=RateLimiterConfig)
    retry:           RetryConfig          = field(default_factory=RetryConfig)
    timeout:         TimeoutConfig        = field(default_factory=TimeoutConfig)
    bulkhead:        BulkheadConfig       = field(default_factory=BulkheadConfig)
    token_budget:    TokenBudgetConfig    = field(default_factory=TokenBudgetConfig)
```

`frozen=True` means the config object is immutable once created. This prevents accidental mutation of shared config objects and makes the config safe to share across threads.

The `from_settings()` class method bridges the gap between the new approach and existing Pydantic/Django settings objects — it reads the same attribute names as before, but consolidates the reading into a single factory function.

### 12.3 Defaults as Documentation

Every field in the config dataclasses has a default value and a docstring that explains why that value was chosen:

```python
@dataclass(frozen=True)
class CircuitBreakerConfig:
    fail_max: int = 5
    """
    How many consecutive failures before the breaker opens.
    Default 5 — low enough to catch real outages quickly,
    high enough to tolerate transient flakes.
    """
    
    reset_timeout: int = 60
    """
    Seconds the breaker stays OPEN before trying HALF-OPEN.
    Default 60s — one minute is usually enough for an LLM
    provider to recover from a rate-limit window.
    """
```

This transforms the config file into a tuning guide. A developer tasked with reducing false-positive circuit-breaker trips knows to look at `fail_max` and understands why the default exists.

---

## Chapter 13 — Protocols and the Dependency Inversion Principle

### 13.1 The Problem with Concrete Dependencies

If `ResilientCaller` imports `CircuitBreaker` directly and constructs it internally, it is coupled to that specific implementation. Testing `ResilientCaller` requires running a real `CircuitBreaker`, which in turn requires `pybreaker`. Adding a distributed circuit breaker backed by Redis requires modifying `ResilientCaller` itself.

This is a violation of the Dependency Inversion Principle: high-level modules should depend on abstractions, not concrete implementations.

### 13.2 Python Protocols as Interfaces

Python's `typing.Protocol` provides structural subtyping — an interface defined by the shape of the class (its methods and properties), not by explicit inheritance. Any class that implements the required methods automatically satisfies the protocol, without needing to declare `class MyBreaker(CircuitBreakerProtocol)`.

```python
@runtime_checkable
class CircuitBreakerProtocol(Protocol):
    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any: ...
    
    @property
    def is_open(self) -> bool: ...
    
    @property
    def name(self) -> str: ...
```

A class satisfies this protocol if it has a `call` method with the right signature and `is_open` and `name` properties. The `pybreaker`-backed `CircuitBreaker` class satisfies it. A test stub satisfies it:

```python
class AlwaysClosedBreaker:
    """Test stub — always passes calls through."""
    
    def call(self, func, *args, **kwargs):
        return func(*args, **kwargs)
    
    @property
    def is_open(self):
        return False
    
    @property
    def name(self):
        return "test-stub"

# AlwaysClosedBreaker satisfies CircuitBreakerProtocol
# without any explicit inheritance
assert isinstance(AlwaysClosedBreaker(), CircuitBreakerProtocol)
```

### 13.3 Practical Benefits

For the developer writing a unit test for a LangGraph node, protocols mean the test does not need to simulate real circuit-breaker state, wait for reset timeouts, or install pybreaker. A stub that satisfies the protocol is injected instead.

For the developer adding a new implementation — say, a circuit breaker whose state is stored in Redis so multiple processes share it — the new class is written to satisfy the protocol. All callers that depend on `CircuitBreakerProtocol` work with the new implementation without modification.

The Interface Segregation Principle (ISP) is also respected: each protocol covers exactly one capability. `RateLimiterProtocol` has only `acquire()`. `TokenTrackerProtocol` has only `check_budget()`, `record_usage()`, and `get_remaining_budget()`. A caller that only needs rate limiting depends only on `RateLimiterProtocol` — not on a large combined interface that also includes circuit-breaking and token tracking.

## Chapter 14 — Practical Execution Flow in a Multi-Agent System

### 14.1 The LangGraph Supervisor Architecture

To see how these patterns operate in practice, consider the execution flow of `mas_resiliency_usage_example.py`. This script implements a routing supervisor pattern in LangGraph, where a central `ClinicalWorkflowSupervisorOrchestrator` determines which specialist agent should execute next based on the presence or absence of data in the shared state.

```
Patient Query ─► Supervisor ─► Triage Agent ─► Supervisor ─► Cardiovascular Agent ─► Supervisor ─► FINISH
```

Unlike single-agent setups, this workflow requires multiple distinct inferences sequentially, all sharing the same token budget, rate limits, and circuit breaker.

### 14.2 Step-by-Step Execution Journey

**Stage 1: System Initialization**
Before any agent runs, the global infrastructure is created. A single `TokenManager` is initialized with a workflow-wide budget (e.g., 4000 tokens). A single `CircuitBreaker` is created and registered for the `openai_api` service. These shared resources are injected into the LangGraph state so every node can access them.

**Stage 2: Supervisor Routing (Initial)**
The workflow begins at the supervisor. The supervisor examines the state: "Medical Triage Completed: NO, Cardiovascular Analysis Completed: NO". 
* **Resilience Stack Execution:** The supervisor node invokes the `ResilientCaller`.
    1. **Token Budget:** Checks if the supervisor's estimated prompt will breach the limit. (Proceeds).
    2. **Bulkhead & Rate Limiter:** Acquires concurrency and rate limit slots.
    3. **Circuit Breaker:** Verifies `openai_api` is CLOSED (healthy).
    4. **Execution:** The LLM call is made. The supervisor decides to route to `MedicalSymptomTriageAndExtractionAgent`.
    5. **Post-call:** The `TokenManager` records the actual tokens used, deducting them from the global budget.

**Stage 3: Specialist Agent Execution (Triage)**
Control transitions to the Triage Agent. 
* **Resilience Stack Execution:** The Triage Agent invokes the *same* `ResilientCaller` configuration, utilizing the *same* workflow-scoped `TokenManager` and global `CircuitBreaker`.
    1. **Token Budget:** The manager verifies that the remaining budget can accommodate this new call.
    2. **Circuit Breaker:** If the previous supervisor call had failed repeatedly, the circuit breaker would be OPEN, and this agent would instantly fail without making a network request. Since it's healthy, it proceeds.
    3. **Execution:** The Triage Agent extracts symptoms. 
    4. **Post-call:** The tokens are deducted. The extracted data is written to the global state.

**Stage 4: Supervisor Routing (Intermediate)**
Control returns to the supervisor. It updates its prompt based on the new state: "Medical Triage Completed: YES, Cardiovascular Analysis Completed: NO". 
* **Continuous Budget Depletion:** By this third LLM call, the shared token budget has been consumed twice. The `TokenManager` effectively limits not just the size of one prompt, but the total number of turns this routing loop can take. If the supervisor were to get stuck in an infinite routing loop, the token budget would eventually raise a `TokenBudgetExceeded` exception, terminating the loop gracefully before massive API costs are incurred.

**Stage 5: Specialist Agent Execution (Cardiovascular)**
The supervisor routes to the `CardiovascularDifferentialDiagnosisAgent`.
* **Execution:** This agent uses a primary provider and a fallback provider. If the primary provider were down, the `FallbackChain` would catch the `CircuitBreakerOpen` from the resilience stack and immediately attempt the backup. The failure would be absorbed by the fallback logic, keeping the Multi-Agent System operational.

**Stage 6: Finalization**
Control returns to the supervisor ("Medical Triage Completed: YES, Cardiovascular Analysis Completed: YES"), which routes to the special `FINISH` node. The LangGraph engine terminates the run. 

### 14.3 The Value of Shared Resilience

This execution trace demonstrates why resilience must act at the *System* level rather than the *Agent* level:
1. **Global Cost Control:** The supervisor, triage agent, and cardiovascular agent all draw from a single token wallet. No individual agent can monopolize it without starving the others, guaranteeing a fixed maximum cost per workflow run.
2. **Instant Failure Propagation:** If the Clinical LLM model goes offline while the Triage agent is running, the `CircuitBreaker` trips. When control passes back to the Supervisor, the Supervisor does not waste 30 seconds waiting for a connection timeout—it instantly receives `CircuitBreakerOpen`, allowing the system to fail fast or trigger fallbacks system-wide.

---

## Appendix A — Pattern Reference

| Pattern | Class | Raises | Primary Guard Against |
|---|---|---|---|
| Circuit Breaker | `CircuitBreaker` | `CircuitBreakerOpen` | Cascading failures on known-down service |
| Rate Limiter | `RateLimiter` | `RateLimitExhausted` | API quota exhaustion from bursts |
| Retry Handler | `RetryHandler` | (re-raises last exception) | Transient network/server failures |
| Timeout Guard | `TimeoutGuard` | `TimeoutExceeded` | Hung calls blocking the pipeline |
| Fallback Chain | `FallbackChain` | `AllFallbacksFailed` | Single-provider outages |
| Bulkhead | `Bulkhead` | `BulkheadFull` | Resource starvation between agent types |
| Token Budget | `TokenManager` | `TokenBudgetExceeded` | Runaway costs from loops or parallel agents |

---

## Appendix B — Exception Recovery Reference

| Exception | Do | Do Not |
|---|---|---|
| `CircuitBreakerOpen` | Return cached or fallback response | Retry — the circuit will reset itself |
| `RateLimitExhausted` | Wait `details["retry_after"]` seconds, then retry | Retry immediately |
| `TimeoutExceeded` | Log, return partial result or fallback | Retry without a timeout increase |
| `AllFallbacksFailed` | Return static error to user, alert on-call | Retry the chain — all providers are down |
| `BulkheadFull` | Wait 50–200ms with jitter, then retry | Queue indefinitely |
| `TokenBudgetExceeded` | Return partial result from completed agents | Make more LLM calls |

---

## Appendix C — Integration Checklist for a New LangGraph Node

1. Determine which LLM API the node calls. Use `CircuitBreakerRegistry.get_or_create(api_name)` to share the breaker with all other nodes calling the same API.

2. Determine whether this node is on a latency-sensitive path. If so, use a shorter `TimeoutConfig.default_timeout`. If not, use the default or increase it.

3. Determine whether this node runs concurrently with other nodes. If the workflow is strictly sequential, pass `skip_bulkhead=True`. If parallel, configure a `Bulkhead` with a capacity appropriate to the resource allocation intent.

4. Retrieve the workflow-level `TokenManager` from state (do not create a new one). Pass it to the `ResilientCaller` constructor.

5. Before the LLM call, estimate token count using `TokenCounter.count(prompt)` and pass as `estimated_tokens` to `caller.call()`.

6. After the LLM call, call `caller.token_manager.record_usage()` with actual token counts from `response.usage_metadata`.

7. Wrap the `caller.call()` invocation in specific exception handlers — at minimum `CircuitBreakerOpen` and `TokenBudgetExceeded`. Do not use a bare `except Exception` handler for resilience exceptions.

8. If upstream failure should cause this node to skip its LLM call, check `state.get("error")` before invoking the caller.