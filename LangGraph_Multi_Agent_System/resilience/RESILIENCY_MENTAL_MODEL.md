# Resilience in Agentic AI — Mental Models for Architects and Senior Engineers

This document does not repeat what each pattern does. It captures the thinking
frameworks, decision rules, and mnemonics that let you reason quickly and
correctly about resilience when designing or reviewing a multi-agent system.

The goal is internalization — these models should eventually run in the
background while you are sketching an architecture, not something you
look up after the fact.

---

## Contents

1. [Mental Model 1 — The Stack as a Cost Ladder](#1-the-stack-as-a-cost-ladder)
2. [Mental Model 2 — Topology is Destiny](#2-topology-is-destiny)
3. [Mental Model 3 — Share Awareness, Isolate Resources](#3-share-awareness-isolate-resources)
4. [Mental Model 4 — Two Clocks, Not One](#4-two-clocks-not-one)
5. [Mental Model 5 — The Transient Gate](#5-the-transient-gate)
6. [Mental Model 6 — Budget Before You Spend](#6-budget-before-you-spend)
7. [Mental Model 7 — The City Alarm, Not the Apartment Detector](#7-the-city-alarm-not-the-apartment-detector)
8. [Mental Model 8 — The Thundering Herd Trap](#8-the-thundering-herd-trap)
9. [Mental Model 9 — Per-Process vs Per-Workflow](#9-per-process-vs-per-workflow)
10. [Mental Model 10 — The Three Topology Risks](#10-the-three-topology-risks)
11. [The Architect's Three Questions](#the-architects-three-questions)
12. [Quick-Reference Cheat Sheet](#quick-reference-cheat-sheet)

---

## 1. The Stack as a Cost Ladder

**The core principle:** The further out a layer sits in the stack, the cheaper
its check must be. Every layer that runs before an LLM call is a tax on that
call. The most expensive checks (spawning threads, sleeping, making network
calls) must only happen when cheaper checks have already confirmed the call
is worth making.

```
Outer (cheapest)                               Inner (most expensive)
     │                                                   │
     ▼                                                   ▼
[Token Budget]──[Bulkhead]──[Rate Limiter]──[Circuit]──[Retry]──[Timeout]──[LLM]
   nanosec       microsec      microsec      microsec   seconds   ms+call   seconds
   arithmetic    semaphore    lock+list      dict       sleep     thread     network
```

**The mnemonic — "Two Bright Red Cars Race Through":**

```
T wo    → Token Budget    (arithmetic — trivially cheap)
B right → Bulkhead        (semaphore — near-zero cost)
R ed    → Rate Limiter    (sliding window — lock + list prune)
C ars   → Circuit Breaker (dict lookup — sub-millisecond)
R ace   → Retry           (exponential sleep — seconds of cost)
T hrough → Timeout        (thread + deadline — overhead + full call duration)
```

Recite this and you have the correct outer-to-inner stack order from memory.
If you ever find yourself writing a stack where a more expensive check
precedes a cheaper one, the order is wrong.

**The diagnostic question:** Before placing a layer, ask — "If the check
after me would have rejected this call anyway, have I wasted anything?"
If yes, move inward. If no, move outward.

---

## 2. Topology is Destiny

**The core principle:** The failure modes of a multi-agent system are
determined entirely by its graph topology. You cannot choose the right
resilience patterns without first characterising the topology. Pattern
selection is downstream of topology selection, never the other way around.

There are three canonical topology shapes. Every MAS you encounter is either
one of these or a composition of them:

```
SEQUENTIAL                PARALLEL                   ITERATIVE
(Linear Pipeline)         (Fan-out / Fan-in)          (Reflection Loop)

A → B → C → D            A → [B ║ C ║ D] → E         A → B → A → B → ...

Risk profile:             Risk profile:               Risk profile:
  Accumulation              Burst + Starvation          Runaway cost
  One failure blocks        All branches share          Iteration count ≠
  the pipeline              resources simultaneously     cost ceiling
```

**Topology-to-pattern mapping:**

| Topology | Primary Risk | Non-Negotiable Patterns | Optional |
|---|---|---|---|
| Sequential | Cost accumulation across N calls | Circuit Breaker, Retry, Timeout, Token Budget | Rate Limiter |
| Parallel | API burst; resource starvation between agent types | Rate Limiter, Bulkhead, Circuit Breaker, Timeout | Token Budget |
| Iterative | Runaway token cost independent of iteration count | Token Budget (as cost ceiling), Timeout | Circuit Breaker |

**The key insight for iterative topology:** An iteration count guard
(`max_iterations = 10`) caps how many times the loop runs. It does not cap
how many tokens each iteration consumes. If the generator becomes verbose
under adversarial critique, each iteration costs 3× what it did in testing.
Token budget is the only mechanism that enforces a cost-based ceiling
regardless of iteration count.

**The composition rule:** Most real systems are composed topologies.
A supervisor pattern with parallel specialists is sequential (Supervisor ↔ Workers)
composed with parallel (Workers run concurrently). Apply the pattern sets
for each component topology, then de-duplicate. You end up with the union
of both pattern sets — in this case, all seven patterns are relevant.

---

## 3. Share Awareness, Isolate Resources

**The core principle:** Every component in the resilience layer falls into
exactly one of two categories. Confusing these categories is the most common
architectural mistake with resilience patterns.

```
SHARE when the component tracks STATE THAT AGENTS MUST AGREE ON

  Circuit Breaker Registry — "Is the API healthy right now?"
                              All agents must have the same answer.
                              If Agent A knows the API is down, Agent B
                              must not re-learn this through its own timeout.

  Token Manager            — "How much have we spent so far this run?"
                              All agents accumulate into one shared counter.
                              If Triage used 800 tokens, Specialist's budget
                              must account for those 800 tokens already spent.


ISOLATE when the component manages RESOURCES THAT AGENTS COMPETE FOR

  Bulkhead                 — Each agent type gets its own bounded pool.
                              A batch job filling its pool must not starve
                              a real-time triage agent.

  Rate Limiter             — Each agent can have its own sliding window,
                              preventing one agent from burning all quota.
```

**The decision rule as a single question:**
> "If this component's state changes because of Agent A's action, does
> Agent B need to see that change?"

- Yes → Share the instance across agents
- No  → Create a separate instance per agent (or per agent type)

**Where this rule applies concretely:**

```
shared_circuit_breaker = CircuitBreakerRegistry.get_or_create("api_name")
# Same object returned every time for the same key.
# Agent A's failure increments the same counter Agent B reads.

triage_rate_limiter   = RateLimiter("triage",   config)
specialist_rate_limiter = RateLimiter("specialist", config)
# Separate instances. Triage exhausting its quota does not
# block specialist from making calls.
```

---

## 4. Two Clocks, Not One

**The core principle:** When retry and timeout compose, there are two
distinct time constraints operating simultaneously. Confusing them leads
to either under-protecting or over-protecting the system.

```
THE ATTEMPT CLOCK (Timeout — innermost layer)
  Governs: one individual LLM call attempt
  Resets:  after every attempt (success or failure)
  Example: 30s per attempt

THE RETRY SEQUENCE (Retry — wraps Timeout)
  Governs: the full retry strategy across all attempts
  Duration: sum of all attempt times + all backoff waits
  Example: attempt 1 (30s) + wait(1s) + attempt 2 (30s) + wait(2s) + attempt 3 (30s)
           = up to ~93s maximum wall time for one "call"
```

**Why Timeout must be innermost:**
If Timeout were outside Retry, the 30-second deadline would govern
the entire retry sequence. One hung call would exhaust the deadline
and no retries would ever happen. Timeout inside Retry means each attempt
gets its own 30-second deadline — the retry strategy works as intended.

```
WRONG — Timeout outside Retry:
  Retry(
    Timeout(30s)(
      llm.invoke(prompt)
    )
  )
  → Total deadline: 30s. If attempt 1 takes 29s and fails, attempt 2
    has 1 second. Retry is effectively disabled.

CORRECT — Timeout inside Retry:
  Retry(
    call: Timeout(30s)(llm.invoke(prompt))
  )
  → Each attempt gets 30s independently.
    3 retries × 30s = 90s max exposure, which is acceptable.
```

**The memory anchor:** Think of a surgeon who gets three attempts to
complete a procedure, each with a 30-minute time limit. The time limit
applies per attempt, not to all three combined. Retry is the "three
attempts" policy; Timeout is the "30-minute limit per attempt" policy.

---

## 5. The Transient Gate

**The core principle:** Retry is not a general-purpose error recovery
mechanism. It is specifically for conditions where the same call, issued
again after a short wait, has a reasonable probability of succeeding.
Retrying permanent errors wastes quota, adds latency, and fills logs
with noise.

**The single question that classifies any exception:**

```
"Will retrying this exact call, after waiting, change the outcome?"

  YES → Transient. Retry it.        NO → Permanent. Propagate immediately.
  ─────────────────────────         ─────────────────────────────────────────
  ConnectionError (network blip)    AuthenticationError (bad API key)
  TimeoutError    (server busy)     InvalidRequestError (malformed prompt)
  RateLimitError  (quota window)    ContentPolicyError  (policy violation)
  APIConnectionError (upstream)     NotFoundError       (endpoint gone)
```

**The architectural implication:** The `TRANSIENT_EXCEPTIONS` tuple in
`retry_handler.py` is the single source of truth for this classification.
It is not in each agent. It is not in each node. It is in one place.
When you integrate a new LLM provider, you add its transient exception
types to this one tuple — every retry site in the entire codebase
inherits the update automatically.

**The idempotency prerequisite:** Retry is only safe on operations that
are idempotent — where calling the same operation twice produces the
same result as calling it once. LLM inference is idempotent in this
sense: the same prompt sent twice produces acceptable output both times.
Tool calls that write state (database inserts, API mutations) are not
idempotent. Do not apply retry to non-idempotent tool invocations without
an explicit idempotency mechanism at the tool level.

---

## 6. Budget Before You Spend

**The core principle:** Token budget enforcement has a two-step lifecycle
that is unlike every other resilience pattern. Every other pattern is
purely reactive — it intercepts a call and either admits or rejects it
based on current conditions. Token budget is partially predictive: it
checks whether a call should happen before it happens, then records
what actually happened after.

**The mental model — "Check Your Balance Before Entering the Store":**

```
NAIVE APPROACH (wrong):
  Make the call → See tokens used → Discover you're over budget
  Problem: You already spent the tokens. The API call already happened.
           The cost is already incurred. The check is meaningless.

CORRECT APPROACH:
  Check balance before entering → Make the call → Update balance with receipt

In code:
  manager.check_budget(agent_name, estimated_tokens)  ← "Can I afford this?"
  response = llm.invoke(prompt)                        ← "Make the purchase"
  manager.record_usage(agent_name, tokens_in, tokens_out)  ← "Update the ledger"
```

**Why estimation, not exact count:**
The exact token count is only knowable after the call. The provider returns
it in `response.usage_metadata`. Before the call, we estimate input tokens
using a local tokenizer. Output tokens cannot be predicted — they depend on
what the model generates. The estimation is intentionally conservative on
input to avoid false positives. The post-call record corrects to actuals,
so all subsequent checks use accurate cumulative totals.

**The cascading effect:** Once the budget is exhausted mid-workflow, every
subsequent agent's Layer 1 check fails — not just the one that tipped it
over. The shared `TokenManager` instance means all agents read the same
exhausted counter. This is correct behaviour: if the workflow budget is
spent, no remaining agent should make another call.

---

## 7. The City Alarm, Not the Apartment Detector

**The core principle:** The value of a circuit breaker in a multi-agent
system is entirely dependent on one property: all agents calling the same
service must share one circuit breaker instance. If each agent has its own,
the failure state is not shared — each agent independently re-learns that
the API is down through its own expensive timeout cycle.

**The analogy:**

```
APARTMENT DETECTOR (per-agent circuit breaker — wrong):

  Agent A's breaker: failure 1/5 ... failure 5/5 → OPENS after 5 attempts
  Agent B's breaker: failure 1/5 ... failure 5/5 → OPENS after 5 attempts
  Agent C's breaker: failure 1/5 ... failure 5/5 → OPENS after 5 attempts

  Each agent independently waits 30s per call × 5 failures = 150s of wasted
  timeout per agent. In a 5-agent workflow, that is 750s of wasted wait time
  before any breaker opens.


CITY ALARM (shared circuit breaker — correct):

  Agent A: failure 1/5
  Agent B: failure 2/5  ← same counter
  Agent C: failure 3/5  ← same counter
  Agent D: failure 4/5  ← same counter
  Agent E: failure 5/5 → BREAKER OPENS ← shared state

  Report/Synthesis:  CircuitBreakerOpen raised in <1ms.
                     No API call made. No 30-second timeout.
```

**The implementation:** The `CircuitBreakerRegistry` is a multiton — one
instance per named key across the entire process. The key is the service
name, not the agent name. Every agent calling `"openai_api"` gets the same
object back from the registry. If you name breakers by agent instead of by
service, you have recreated the apartment detector problem.

```
WRONG — breaker per agent:
  agent_a_breaker = CircuitBreaker("agent_a", config)
  agent_b_breaker = CircuitBreaker("agent_b", config)

CORRECT — breaker per service:
  shared_breaker = CircuitBreakerRegistry.get_or_create("openai_api", config)
  # Both agents receive the same object
```

---

## 8. The Thundering Herd Trap

**The core principle:** Without jitter, retry logic in a multi-agent system
creates synchronized retry bursts that are often worse than the original
failure. This is not a theoretical concern — it is a reliably reproducible
failure mode in any system with more than two agents retrying simultaneously.

**The scenario:**

```
t=0s:   5 agents hit a rate limit simultaneously (burst request)

WITHOUT JITTER — synchronized retry:
  t=1s:   5 agents retry simultaneously → hit rate limit again
  t=2s:   5 agents retry simultaneously → hit rate limit again
  t=4s:   5 agents retry simultaneously → hit rate limit again
  Result: System thrashes. Rate limit is never cleared because each
          retry burst immediately re-exhausts it.

WITH JITTER — desynchronized retry:
  t=1.1s: Agent A retries             → admitted (limit has some headroom)
  t=1.4s: Agent B retries             → admitted
  t=1.7s: Agent C retries             → admitted
  t=2.1s: Agent D retries             → admitted
  t=2.3s: Agent E retries             → admitted
  Result: Load spread across the recovery window. All agents succeed.
```

**The formula:**
```
wait = min(initial_wait × 2^(attempt - 1), max_wait) + random(0, jitter)

With defaults (initial=1s, max=30s, jitter=1s):
  Attempt 1:  0s     (immediate)
  Attempt 2:  ~1–2s  (1s base + ±1s jitter)
  Attempt 3:  ~2–3s  (2s base + ±1s jitter)
  Attempt 4:  ~4–5s  (4s base + ±1s jitter)
```

**The memory anchor:** Imagine 100 people all running for the same single
door at the same moment. They all arrive, all try to enter, all collide,
all back off, all try again simultaneously. Jitter tells each person to
wait a slightly different amount before trying again — they desynchronize
naturally and everyone eventually gets through.

---

## 9. Per-Process vs Per-Workflow

**The core principle:** Every object in the resilience layer belongs to
one of two lifetimes. Confusion between these lifetimes causes either
stale state (reusing what should be fresh) or wasted initialisation
(recreating what should be shared).

```
PER-PROCESS (create once at application boot, reuse across all workflow runs)

  CircuitBreakerRegistry    — Failure history is meaningful across runs.
                              A service that was down 30 seconds ago might
                              still be recovering. Resetting the breaker
                              per workflow run loses this signal.

  LLM Client                — Connection pools and auth are expensive to
                              initialise. One client serves all agents.

  ResilientCaller           — Config-level object. Contains rate limiter
                              state (sliding window), bulkhead (semaphore),
                              retry config. Appropriate to persist.

  TokenCounter              — Stateless after init. Tokenizer loading is
                              the expensive part. Reuse freely.


PER-WORKFLOW (create fresh for each workflow execution)

  TokenManager              — This is the only object that MUST be fresh
                              per run. Its internal counter accumulates.
                              If reused, run 2 starts with run 1's totals
                              and immediately fails its budget check.

  The test: "Does this object's state need to reset between workflow runs?"
    No  → Per-process. Create at boot.
    Yes → Per-workflow. Create in runner.py, inject into initial_state.
```

**A common mistake:** Creating a `TokenManager` at module load (alongside
the `CircuitBreaker` and `ResilientCaller`). This appears correct because
the other objects are also created at module load. The difference is that
the TokenManager's counter is cumulative — it is the one object in the
resilience layer whose state is specific to one workflow execution.

---

## 10. The Three Topology Risks

This is the synthesis of everything above into a single diagnostic tool.
When evaluating any new MAS architecture, identify which topology category
applies, then read the corresponding risk profile and mandatory patterns.

**The Risk Triangle:**

```
                    ITERATIVE
                    (Loops)
                       ▲
                       │  Primary risk:
                       │  RUNAWAY COST
                       │  Iteration count ≠ cost.
                       │  Token budget is the
                       │  only cost-based ceiling.
                       │
          ─────────────┼─────────────
         ╱             │              ╲
        ╱              │               ╲
   SEQUENTIAL          │            PARALLEL
   (Pipelines)         │            (Fan-out)
                       │
   Primary risk:       │   Primary risk:
   ACCUMULATION        │   BURST + STARVATION
   Cost adds up        │   Simultaneous calls
   across N agents.    │   exhaust quota and
   One failure blocks  │   thread pools.
   the whole chain.
```

**Pattern selection by risk:**

```
ACCUMULATION risk → Token Budget (cumulative cost tracking)
                 → Circuit Breaker (shared failure awareness)
                 → Retry + Timeout (standard per-call protection)

BURST + STARVATION risk → Rate Limiter (smooth the burst)
                       → Bulkhead (separate pools per agent type)
                       → Circuit Breaker (shared failure state)

RUNAWAY COST risk → Token Budget (cost ceiling independent of iteration count)
                 → Timeout (prevents one iteration from hanging indefinitely)
                 → Circuit Breaker (catches API failures during long loops)
```

**The composition rule for real systems:**
Most production MAS architectures combine topologies. A supervisor pattern
with parallel specialist workers is simultaneously sequential (Supervisor ↔
Workers lifecycle) and parallel (Worker invocations within one cycle). Apply
the pattern sets for each component topology. The result is the union.

---

## The Architect's Three Questions

Before finalising any MAS architecture decision involving resilience,
three questions must have clear answers. If any answer is "I'm not sure",
the design is not ready.

**Question 1 — What is the topology, and what does it imply?**

Draw the graph. Classify each subgraph: sequential, parallel, or iterative.
For each classification, identify the primary risk (accumulation, burst +
starvation, runaway cost). The risk determines the mandatory patterns.

**Question 2 — What is shared, and what is isolated?**

For each resilience component, apply the "Share Awareness, Isolate Resources"
rule. The circuit breaker and token manager must be shared. Rate limiter and
bulkhead pools should be isolated per agent type. Make this explicit in the
design — do not leave it to the developer implementing the nodes to decide.

**Question 3 — What is the failure boundary?**

When a resilience layer blocks a call, what happens to the workflow? Trace
the exception path from the blocked call through the node, through the graph
edges, to the user-visible output. A resilience layer that raises a typed
exception and a node that does not catch it will crash the graph — not degrade
gracefully. The failure boundary must be defined at design time, not discovered
in production.

```
Each exception type needs a mapped outcome before the architecture is final:

  CircuitBreakerOpen  → return cached result / skip specialist / degrade output
  TimeoutExceeded     → return partial result / escalate to user
  TokenBudgetExceeded → terminate workflow / return what completed so far
  BulkheadFull        → return 503 equivalent / queue for retry
  AllFallbacksFailed  → return static error / alert on-call
```

---

## Quick-Reference Cheat Sheet

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STACK ORDER (outer → inner)   "Two Bright Red Cars Race Through"           │
│  Token Budget → Bulkhead → Rate Limiter → Circuit → Retry → Timeout → LLM  │
├─────────────────────────────────────────────────────────────────────────────┤
│  TOPOLOGY → RISK → PATTERN                                                  │
│  Sequential  → Accumulation      → Token Budget + Circuit + Retry + Timeout │
│  Parallel    → Burst + Starvation → Rate Limiter + Bulkhead + Circuit       │
│  Iterative   → Runaway Cost       → Token Budget (critical) + Timeout       │
├─────────────────────────────────────────────────────────────────────────────┤
│  SHARE vs ISOLATE                                                            │
│  Share   → failure awareness + cost accounting (Circuit Breaker, Token Mgr) │
│  Isolate → resource pools + throughput control (Bulkhead, Rate Limiter)     │
├─────────────────────────────────────────────────────────────────────────────┤
│  RETRY CLASSIFICATION — "Will retrying change the outcome?"                 │
│  Yes → Transient (ConnectionError, TimeoutError, RateLimitError)            │
│  No  → Permanent (AuthError, InvalidRequest, ContentPolicy) — never retry   │
├─────────────────────────────────────────────────────────────────────────────┤
│  TOKEN BUDGET LIFECYCLE                                                      │
│  check_budget() BEFORE call  →  llm.invoke()  →  record_usage() AFTER call  │
│  TokenManager is PER-WORKFLOW — create fresh in runner.py every execution   │
├─────────────────────────────────────────────────────────────────────────────┤
│  CIRCUIT BREAKER NAMING — key by SERVICE, not by AGENT                      │
│  CircuitBreakerRegistry.get_or_create("openai_api")  ← correct             │
│  CircuitBreaker("triage_agent_breaker")              ← apartment detector  │
├─────────────────────────────────────────────────────────────────────────────┤
│  OBJECT LIFETIMES                                                            │
│  Per-process: CircuitBreaker, ResilientCaller, LLM client, TokenCounter     │
│  Per-workflow: TokenManager  ← only one that resets between executions      │
├─────────────────────────────────────────────────────────────────────────────┤
│  TIMEOUT POSITION — "Per attempt, not per sequence"                         │
│  Timeout INSIDE Retry: each attempt gets its own deadline (correct)         │
│  Timeout OUTSIDE Retry: entire retry sequence shares one deadline (wrong)   │
├─────────────────────────────────────────────────────────────────────────────┤
│  BULKHEAD ENABLE/DISABLE RULE                                                │
│  Sequential (one agent at a time)   → skip_bulkhead=True  (no competition)  │
│  Parallel   (multiple concurrently) → skip_bulkhead=False (isolation needed) │
├─────────────────────────────────────────────────────────────────────────────┤
│  THREE ARCHITECT QUESTIONS                                                   │
│  1. What is the topology, and what risk does it imply?                       │
│  2. What is shared (awareness) vs isolated (resources)?                     │
│  3. What is the failure boundary for each resilience exception type?        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## A Note on Over-Engineering

The most common mistake with resilience patterns is applying all of them
uniformly to every architecture. Bulkhead on a sequential pipeline adds
semaphore overhead with no benefit. Token budget on a single-step demo
adds state management complexity with no meaningful cost protection.

The mental models in this document are tools for deciding which patterns
apply in a given context, not a prescription to use all seven everywhere.
The discipline is in the decision, not in the pattern count.

A system with three patterns chosen correctly for its topology is more
resilient than a system with seven patterns applied uniformly without
understanding why.