# Chapter 3 — Deep Dive: The `BaseOrchestrator`

> **Learning chapter** — A granular breakdown of `orchestrator.py`, the beating heart of the 5 multi-agent patterns.

---

## 3.1 The Abstract Base Class (ABC)

```python
class BaseOrchestrator(ABC):
    @property
    @abstractmethod
    def pattern_name(self) -> str: ...
```

The `BaseOrchestrator` is an Abstract Base Class. It cannot be instantiated on its own. 
Instead, inside every `agents.py` file in the `scripts/orchestration/` folders, a new class inherits it:
```python
# In scripts/orchestration/supervisor_orchestration/agents.py
class SupervisorOrchestrator(BaseOrchestrator):
    pattern_name = "supervisor"
```
This forces every architecture to provide a unified telemetry name (`pattern_name`) while inheriting the massive resilience capabilities embedded in the parent class.

---

## 3.2 `invoke_specialist` (The Engine)

This is the most critical function in the entire repository. Every time a Pulmonology Agent, Cardiology Agent, or Nephrology Agent does work, it is calling this single function.

```python
def invoke_specialist(
    self,
    specialty: str,
    patient: PatientCase,
    context: str = "",
    max_words: int = 120,
    token_manager: TokenManager | None = None,
) -> OrchestrationResult:
```

### Step 1: Prompt Construction
It performs a dictionary lookup against `SPECIALIST_SYSTEM_PROMPTS` using the `specialty` string. It formats the `patient` into text, appends any upstream `context` from previous agents, and instructs the LLM to stay under `max_words`.

### Step 2: Telemetry and Token Budgeting
It builds a Langfuse trace config tagged with `[orchestration, self.pattern_name, specialty]`.
If a `token_manager` was passed, it runs `_TOKEN_COUNTER.count(prompt)`. If the prompt alone blows past the workflow's configured budget, it fails fast *before* making the network call, saving money.

### Step 3: The Resilient Call
```python
response = _ORCHESTRATION_CALLER.call(
    llm.invoke,
    prompt,
    config=config,
    skip_rate_limiter=False,  
    skip_bulkhead=True,       
)
```
Instead of calling `llm.invoke(prompt)` directly, it passes the function pointer to `_ORCHESTRATION_CALLER.call()`. This wraps the network request in 6 layers of armor (Timeouts, Retries, Circuit Breakers).

### Step 4: Graceful Degradation (Catching Exceptions)
If the API is down, `ResilientCaller` will throw a `CircuitBreakerOpen` exception. 
`invoke_specialist` catches this, and instead of crashing the Python script, it returns an `OrchestrationResult` with `was_successful=False` and the error message wrapped safely inside.

---

## 3.3 `invoke_synthesizer` (The Assembler)

Once the specialists finish, the graph edges route their `OrchestrationResult` envelopes to the synthesis node.

```python
def invoke_synthesizer(
    self,
    results: list[OrchestrationResult],
    patient: PatientCase,
) -> str:
```

### Step 1: Flattening the Results
It iterates through the `list[OrchestrationResult]`, skipping any that have `was_successful=False` (graceful degradation!). It formats them into a massive block of text:
```text
[PULMONOLOGY]: Patient needs inhaler.
[CARDIOLOGY]: Patient needs diuretic.
```

### Step 2: The Final Call
Just like `invoke_specialist`, it fires the LLM through the `_ORCHESTRATION_CALLER`. 

### Step 3: Hard Failures
Unlike `invoke_specialist`, if `invoke_synthesizer` encounters a Resilience Exception (like the API going down), it **re-raises it as a `RuntimeError`**. 
Why? Because if a specialist fails, the Synthesizer can still write a partial report. But if the Synthesizer fails, the entire workflow has failed to produce the final deliverable.

---

## 3.4 The Global Circuit Breaker

At the top of the file, you will find:
```python
_ORCHESTRATION_LLM_BREAKER = CircuitBreakerRegistry.get_or_create(
    "orchestration_llm_api",
    CircuitBreakerConfig(fail_max=5, reset_timeout=60),
)
```
Because this is instantiated at the module level, **it is shared globally**.
If you are running the `PeerToPeer` pattern and the exact same `Supervisor` pattern concurrently across different threads, and the OpenAI API crashes, the very first 5 failing requests will trip the breaker into `OPEN`.
Instantly, ALL patterns across ALL threads will stop trying to hit OpenAI, gracefully degrading immediately.
