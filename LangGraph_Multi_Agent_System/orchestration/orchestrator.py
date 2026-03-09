"""
============================================================
Base Orchestrator — Abstract Interface
============================================================
Defines the contract that all orchestration patterns implement.

Every orchestrator must:
    1. Accept a patient workload
    2. Coordinate specialist agents
    3. Produce a final synthesized result

The base class provides shared utility methods for invoking
specialist LLMs and formatting prompts, so individual patterns
only implement their coordination logic.

════════════════════════════════════════
  WHERE THIS FITS IN THE MAS ARCHITECTURE
════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────┐
    │                   MAS Architecture                       │
    │                                                          │
    │  orchestration/orchestrator.py  <── YOU ARE HERE         │
    │  ────────────────────────────────────────────────────    │
    │  The ROOT component module for all orchestration.        │
    │  Pattern scripts (scripts/orchestration/*) extend this.  │
    │                                                          │
    │  ┌──────────────────────────────────────────────────┐    │
    │  │  resilience/ (6-layer stack, always on)          │    │
    │  │  ┌────────────────────────────────────────────┐  │    │
    │  │  │  ResilientCaller (_ORCHESTRATION_CALLER)   │  │    │
    │  │  │  ├─ Token Budget (outermost)               │  │    │
    │  │  │  ├─ Bulkhead (skipped for linear flows)    │  │    │
    │  │  │  ├─ Rate Limiter (ENABLED, smooths bursts) │  │    │
    │  │  │  ├─ Circuit Breaker (shared _LLM_BREAKER)  │  │    │
    │  │  │  ├─ Retry (transient errors only)          │  │    │
    │  │  │  └─ Timeout (per-call deadline, innermost) │  │    │
    │  │  └────────────────────────────────────────────┘  │    │
    │  └──────────────────────────────────────────────────┘    │
    │                       ▲                                  │
    │                       │ invoke_specialist()              │
    │                       │ invoke_synthesizer()             │
    │                       │                                  │
    │  ┌────────────────────┴─────────────────────────────┐    │
    │  │              BaseOrchestrator                    │    │
    │  │  (abstract: pattern_name, description)           │    │
    │  └──────────────────────────────────────────────────┘    │
    │                       ▲                                  │
    │       ┌───────────────┼────────────────┐                 │
    │  SupervisorOrch  PeerToPeerOrch  DynamicRouterOrch ...   │
    │  (STAGE 1.2)     (STAGE 2.2)     (STAGE 3.2)             │
    └──────────────────────────────────────────────────────────┘

RESILIENCE INTEGRATION (see resilience/RESILIENCE_MAS_MAPPING.md):
    All LLM calls in invoke_specialist() and invoke_synthesizer() go through
    _ORCHESTRATION_CALLER (a ResilientCaller with a shared circuit breaker).

    RESILIENCE LAYERS ACTIVE:
        - Circuit Breaker: shared across all orchestration patterns; if the
          LLM API is down, ALL patterns fail fast immediately without retrying.
        - Retry: automatic retry on transient errors (429, timeout, connection).
        - Timeout: per-call deadline (30s default) to avoid hung graph nodes.
        - Rate Limiter: ENABLED (skip_rate_limiter=False) to smooth request bursts.
        - Token Budget: OPTIONAL — pass token_manager arg to invoke_specialist/synthesizer.
        - Bulkhead: SKIPPED — linear orchestration flows don't need concurrency isolation.

    CONNECTION: resilience/circuit_breaker.py — _ORCHESTRATION_LLM_BREAKER is the
    shared CircuitBreaker instance. All invoke_specialist/invoke_synthesizer calls
    flow through it.

    CONNECTION: resilience/resilient_caller.py — _ORCHESTRATION_CALLER.call()
    applies the full 6-layer resilience stack to every llm.invoke().

SOLID Design:
    - Open/Closed: new orchestration patterns extend this class
    - Dependency Inversion: graph builders depend on this ABC
    - Liskov Substitution: any subclass works interchangeably
============================================================
"""

import sys
import time
from abc import ABC, abstractmethod


from core.config import get_llm
from core.models import PatientCase
from observability.callbacks import build_callback_config

# CONNECTION: resilience/ root module — these imports wire the 6-layer
# resilience stack into every LLM call this orchestrator makes.
from resilience import (
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    ResilienceConfig,
    ResilientCaller,
    TimeoutConfig,
    TokenCounter,
    TokenManager,
)
from resilience.exceptions import (
    BulkheadFull,
    CircuitBreakerOpen,
    RateLimitExhausted,
    ResilienceError,
    TimeoutExceeded,
    TokenBudgetExceeded,
)

# CONNECTION: orchestration/models.py — OrchestrationResult is the standard
# envelope that invoke_specialist() returns and invoke_synthesizer() receives.
from orchestration.models import OrchestrationResult, format_patient_for_prompt

# ── Resilience layer (orchestration integration point) ─────────────────────
# CONNECTION: resilience/circuit_breaker.py — CircuitBreakerRegistry.get_or_create
# creates a NAMED breaker shared by ALL agents hitting the same LLM API. When the
# API goes down, ONE open breaker stops ALL patterns from making further calls.
# Name "orchestration_llm_api" matches the registry key in circuit_breaker.py docs.
_ORCHESTRATION_LLM_BREAKER = CircuitBreakerRegistry.get_or_create(
    "orchestration_llm_api",
    CircuitBreakerConfig(fail_max=5, reset_timeout=60),
)

# CONNECTION: resilience/resilient_caller.py — ResilientCaller is the FAÇADE
# that composes all 6 resilience layers. One instance shared by all 5 patterns
# through BaseOrchestrator._ORCHESTRATION_CALLER. This avoids each pattern
# creating its own breaker/retry/timeout setup (DRY principle).
_ORCHESTRATION_CALLER = ResilientCaller(
    config=ResilienceConfig(
        timeout=TimeoutConfig(default_timeout=30.0),
    ),
    agent_name="orchestration",
    circuit_breaker=_ORCHESTRATION_LLM_BREAKER,
    token_manager=None,  # Token budget disabled by default; pass via subclass if needed
)

# Token counter for pre-call estimation (reusable across all workflows)
_TOKEN_COUNTER = TokenCounter(model="gpt-4o")


# ============================================================
# Specialist System Prompts
# ============================================================
# Centralized prompt definitions so all orchestration patterns
# give agents the same instructions, ensuring fair comparison.

SPECIALIST_SYSTEM_PROMPTS = {
    "pulmonology": (
        "You are a pulmonology specialist in a clinical decision support system. "
        "Focus on respiratory function, COPD management, oxygen status, and "
        "inhaler therapy. Assess whether the current respiratory regimen is "
        "adequate and identify any acute exacerbation signs."
    ),
    "cardiology": (
        "You are a cardiology specialist in a clinical decision support system. "
        "Focus on heart failure management, BNP levels, fluid status, and "
        "cardiac medication optimization. Evaluate whether diuretic dosing "
        "and heart failure medications need adjustment."
    ),
    "nephrology": (
        "You are a nephrology specialist in a clinical decision support system. "
        "Focus on renal function (eGFR, creatinine), electrolyte management, "
        "and medication dose adjustments for renal impairment. Flag any "
        "nephrotoxic medications or dose changes needed."
    ),
}


class BaseOrchestrator(ABC):
    """
    Abstract base class for all orchestration patterns.

    Provides shared utilities:
        - invoke_specialist(): call a specialist LLM with standard config
        - format_patient(): convert PatientCase to prompt string
        - build_synthesis_prompt(): create a synthesis request from results

    Subclasses implement:
        - pattern_name (property): unique identifier for this pattern
        - description (property): one-line description of the pattern

    CONNECTION: All 5 pattern orchestrators in scripts/orchestration/*/agents.py
    extend this class (SupervisorOrchestrator, PeerToPeerOrchestrator, etc.).
    """

    # -- Abstract interface ------------------------------------------------

    @property
    @abstractmethod
    def pattern_name(self) -> str:
        """Unique name for this orchestration pattern."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description of what this pattern does."""
        ...

    # -- Shared utilities --------------------------------------------------

    def invoke_specialist(
        self,
        specialty: str,
        patient: PatientCase,
        context: str = "",
        max_words: int = 120,
        trace_prefix: str = "",
        token_manager: TokenManager | None = None,
    ) -> OrchestrationResult:
        """
        Invoke a specialist LLM and return a standardized result.

        CONNECTION (resilience/circuit_breaker.py): This method is the
        SOURCE that triggers the circuit breaker. The circuit breaker will
        OPEN if 5 consecutive calls fail. All subsequent calls raise
        CircuitBreakerOpen immediately (no LLM call made).

        CONNECTION (resilience/rate_limiter.py): Rate limiter is ENABLED
        (skip_rate_limiter=False). Smooths request bursts when multiple
        specialists hit the LLM API in quick succession.

        CONNECTION (resilience/resilient_caller.py): _ORCHESTRATION_CALLER
        applies the full stack: token budget → bulkhead → rate limiter →
        circuit breaker → retry → timeout (innermost).

        TOKEN MANAGER (optional per-workflow):
            If token_manager is provided:
            1. Estimates tokens BEFORE call (fail fast if over budget).
            2. Checks budget via token_manager.check_budget(specialty, estimated).
            3. Records actual usage AFTER call from response.usage_metadata.

            WHY: Multi-agent workflows multiply costs. Supervisor pattern invokes
            3 specialists + synthesis = 4 LLM calls. At 500 tokens each = 2,000
            tokens per workflow. Token budget caps cost at workflow level.

        DESTINATION NODES that trigger this method:
            - STAGE 1.2: pulmonology_worker_node, cardiology_worker_node,
                         nephrology_worker_node (supervisor_orchestration/agents.py)
            - STAGE 2.2: pulmonology_peer_node, cardiology_peer_node,
                         nephrology_peer_node (peer_to_peer_orchestration/agents.py)
            - STAGE 3.2: pulmonology_specialist_node, cardiology_specialist_node,
                         nephrology_specialist_node (dynamic_router_orchestration/agents.py)
            - STAGE 4.2: assessment/risk/recommendation nodes via ResilientCaller
                         (graph_of_subgraphs_orchestration/agents.py)
            - STAGE 5.2: cardiopulmonary_pulmonology_node, cardiopulmonary_cardiology_node,
                         renal_specialist_node (hybrid_orchestration/agents.py)

        Args:
            specialty: Clinical specialty (must be in SPECIALIST_SYSTEM_PROMPTS)
            patient: The patient case to assess
            context: Optional upstream findings from other agents
            max_words: Word limit for the response
            trace_prefix: Prefix for the Langfuse trace name
            token_manager: Optional TokenManager for per-workflow budget enforcement

        Returns:
            OrchestrationResult with the agent's assessment
        """
        system_prompt = SPECIALIST_SYSTEM_PROMPTS.get(
            specialty,
            f"You are a {specialty} specialist. Provide a focused clinical assessment."
        )

        patient_text = format_patient_for_prompt(patient)
        prompt = f"{system_prompt}\n\n{patient_text}"
        if context:
            prompt += f"\n\nFindings from other specialists:\n{context}"
        prompt += f"\n\nProvide your assessment in under {max_words} words."

        trace_name = f"{trace_prefix or self.pattern_name}_{specialty}"
        config = build_callback_config(
            trace_name=trace_name,
            tags=["orchestration", self.pattern_name, specialty],
        )

        start_time = time.time()

        # ── TOKEN BUDGET: Estimate and check BEFORE call ────────────────────
        # CONNECTION: resilience/token_manager.py — check_budget raises
        # TokenBudgetExceeded if this agent would exceed its allocation.
        estimated_tokens = 0
        if token_manager:
            estimated_tokens = _TOKEN_COUNTER.count(prompt)
            try:
                token_manager.check_budget(specialty, estimated_tokens)
            except TokenBudgetExceeded as e:
                duration = time.time() - start_time
                return OrchestrationResult(
                    agent_name=f"{specialty}_specialist",
                    specialty=specialty,
                    output="",
                    duration_seconds=round(duration, 2),
                    was_successful=False,
                    error_message=f"TokenBudgetExceeded: {e.message}",
                )

        try:
            llm = get_llm()
            # CONNECTION: resilience/resilient_caller.py — _ORCHESTRATION_CALLER
            # applies: rate limiter → circuit breaker → retry → timeout around llm.invoke.
            response = _ORCHESTRATION_CALLER.call(
                llm.invoke,
                prompt,
                config=config,
                skip_rate_limiter=False,  # RATE LIMITER ENABLED (smooths bursts)
                skip_bulkhead=True,       # Bulkhead skipped (linear flows)
            )
            duration = time.time() - start_time

            # ── TOKEN BUDGET: Record actual usage AFTER call ────────────────
            if token_manager and hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                token_manager.record_usage(
                    specialty,
                    tokens_in=usage.get("input_tokens", estimated_tokens),
                    tokens_out=usage.get("output_tokens", 0),
                )

            return OrchestrationResult(
                agent_name=f"{specialty}_specialist",
                specialty=specialty,
                output=response.content,
                duration_seconds=round(duration, 2),
                was_successful=True,
            )
        except (CircuitBreakerOpen, TimeoutExceeded, TokenBudgetExceeded,
                RateLimitExhausted, BulkheadFull) as e:
            duration = time.time() - start_time
            return OrchestrationResult(
                agent_name=f"{specialty}_specialist",
                specialty=specialty,
                output="",
                duration_seconds=round(duration, 2),
                was_successful=False,
                error_message=f"{type(e).__name__}: {e.message}",
            )
        except ResilienceError as e:
            duration = time.time() - start_time
            return OrchestrationResult(
                agent_name=f"{specialty}_specialist",
                specialty=specialty,
                output="",
                duration_seconds=round(duration, 2),
                was_successful=False,
                error_message=str(e),
            )
        except Exception as error:
            duration = time.time() - start_time
            return OrchestrationResult(
                agent_name=f"{specialty}_specialist",
                specialty=specialty,
                output="",
                duration_seconds=round(duration, 2),
                was_successful=False,
                error_message=str(error),
            )

    def build_synthesis_prompt(
        self,
        results: list[OrchestrationResult],
        patient: PatientCase,
        max_words: int = 200,
    ) -> str:
        """
        Build a prompt that asks the LLM to synthesize multiple
        specialist results into a unified clinical report.
        """
        specialist_findings = "\n\n".join(
            f"[{result.specialty.upper()} ({result.agent_name})]:\n{result.output}"
            for result in results
            if result.was_successful
        )

        return f"""Synthesize these specialist findings into a unified clinical report:

Patient: {patient.age}y {patient.sex}, {patient.chief_complaint}

{specialist_findings}

Produce a structured report:
1) Critical Findings
2) Cross-Specialty Interactions (e.g., renal function affecting cardiac medication)
3) Integrated Treatment Plan
4) Monitoring Priorities

Keep under {max_words} words."""

    def invoke_synthesizer(
        self,
        results: list[OrchestrationResult],
        patient: PatientCase,
        max_words: int = 200,
        token_manager: TokenManager | None = None,
    ) -> str:
        """
        Synthesize multiple specialist results into a final report.

        CONNECTION (resilience/circuit_breaker.py): This method triggers the
        circuit breaker via _ORCHESTRATION_CALLER. Same shared breaker as
        invoke_specialist — if the API is down, synthesis fails fast too.

        CONNECTION (resilience/resilient_caller.py): Full 6-layer stack applies.
        Rate limiter ENABLED. Bulkhead skipped.

        TOKEN MANAGER (optional):
            Synthesis is the final, most expensive call (aggregates all specialist
            outputs in prompt = longest input). Failing here saves the most tokens.

        DESTINATION NODES that trigger this method:
            - STAGE 1.2: report_synthesis_node (supervisor_orchestration/agents.py)
            - STAGE 2.2: synthesis_node (peer_to_peer_orchestration/agents.py)
            - STAGE 3.2: router_report_node (dynamic_router_orchestration/agents.py)
            - STAGE 4.2: synthesis_node via ResilientCaller.call() (graph_of_subgraphs)
            - STAGE 5.2: hybrid_synthesis_node (hybrid_orchestration/agents.py)

        Resilience exceptions are re-raised as RuntimeError so the caller
        can handle synthesis failure (e.g. return partial report).
        """
        prompt = self.build_synthesis_prompt(results, patient, max_words)
        config = build_callback_config(
            trace_name=f"{self.pattern_name}_synthesis",
            tags=["orchestration", self.pattern_name, "synthesis"],
        )

        # ── TOKEN BUDGET: Estimate and check BEFORE call ────────────────────
        estimated_tokens = 0
        if token_manager:
            estimated_tokens = _TOKEN_COUNTER.count(prompt)
            try:
                token_manager.check_budget("synthesis", estimated_tokens)
            except TokenBudgetExceeded as e:
                raise RuntimeError(f"TokenBudgetExceeded: {e.message}") from e

        try:
            llm = get_llm()
            # CONNECTION: resilience/resilient_caller.py — same _ORCHESTRATION_CALLER
            # as invoke_specialist; synthesis shares the circuit breaker state.
            response = _ORCHESTRATION_CALLER.call(
                llm.invoke,
                prompt,
                config=config,
                skip_rate_limiter=False,  # RATE LIMITER ENABLED (smooths bursts)
                skip_bulkhead=True,
            )

            # ── TOKEN BUDGET: Record actual usage AFTER call ────────────────
            if token_manager and hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                token_manager.record_usage(
                    "synthesis",
                    tokens_in=usage.get("input_tokens", estimated_tokens),
                    tokens_out=usage.get("output_tokens", 0),
                )

            return response.content
        except (CircuitBreakerOpen, TimeoutExceeded, TokenBudgetExceeded,
                RateLimitExhausted, BulkheadFull) as e:
            raise RuntimeError(f"{type(e).__name__}: {e.message}") from e
        except ResilienceError as e:
            raise RuntimeError(str(e)) from e
        except Exception as e:
            raise RuntimeError(str(e)) from e

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(pattern='{self.pattern_name}')>"
