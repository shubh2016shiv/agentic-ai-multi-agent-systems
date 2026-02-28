"""
Resilient Multi-Agent Medical Triage System — Step-by-Step Learning Example
=============================================================================
This script is a complete, runnable learning resource. Its purpose is to make
every resilience pattern VISIBLE as it executes, so you can observe not just
WHAT happened, but WHY each layer made the decision it did.

WHAT THIS SCRIPT DEMONSTRATES:
    Every single LLM call in this workflow is preceded by a 6-layer resilience
    stack inspection printed to the console. You will see:
      - Token budget remaining before the call is attempted
      - Bulkhead pool occupancy (how many slots are taken vs. available)
      - Rate limiter quota window configuration and mode
      - Circuit breaker current state (CLOSED / OPEN) and failure count
      - Retry handler backoff configuration and retryable exception list
      - Timeout guard deadline and enforcement mechanism

    After each call you will see:
      - Full, untruncated LLM response
      - Actual tokens consumed (input + output separately)
      - Cumulative workflow token budget consumed so far
      - Time elapsed for that specific LLM call

WORKFLOW TOPOLOGY (Supervisor Pattern):

    Patient Query
         │
         ▼
    ┌────────────────────────────────────────────────────┐
    │   ClinicalWorkflowSupervisorOrchestrator           │  ◄─────┐
    │   Decides which agent runs next based on state.    │        │
    └────────────────────────────────────────────────────┘        │
         │               │                    │                   │
         ▼               ▼                    ▼                   │
    ┌─────────┐  ┌─────────────────┐     ┌────────┐               │
    │  route  │  │  route to       │     │  FINISH│               │
    │  to     │  │  CardioVascular │     │  (END) │               │
    │  Triage │  │  Specialist     │     └────────┘               │
    └────┬────┘  └────────┬────────┘                              │
         │               │                                        │
         ▼               ▼                                        │
    ┌───────────────────────────────────────────────┐             │
    │  MedicalSymptomTriageAndExtractionAgent       │─────────────┘
    │  OR                                           │   (always reports
    │  CardiovascularDifferentialDiagnosisAgent     │    back to Supervisor)
    └───────────────────────────────────────────────┘

RESILIENCE LAYERS (applied in this order for every LLM call):
    [Layer 1]  Token Budget Pre-Check      — fail fast before consuming resources
    [Layer 2]  Bulkhead Concurrency Pool   — cap concurrent calls per agent type
    [Layer 3]  Rate Limiter Sliding Window — smooth bursts, stay within API quota
    [Layer 4]  Circuit Breaker State Check — skip known-down services in <1ms
    [Layer 5]  Retry Handler (backoff)     — auto-recover from transient failures
    [Layer 6]  Timeout Guard (deadline)    — hard per-call deadline, thread-safe

MEDICAL SCENARIO:
    A 55-year-old male presents with sudden-onset crushing chest pain, diaphoresis,
    and shortness of breath. History includes hypertension. This is a high-acuity
    presentation requiring rapid triage and deep specialist analysis.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import get_llm
from resilience import (
    BulkheadConfig,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitBreakerRegistry,
    ResilientCaller,
    ResilienceConfig,
    TimeoutConfig,
    TokenBudgetConfig,
    TokenCounter,
    TokenManager,
)
from resilience.exceptions import TimeoutExceeded, TokenBudgetExceeded


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONSOLE FORMATTING CONSTANTS AND HELPERS
#
# All print functions are grouped here. Separating formatting from logic means:
#   - Node functions (Section 6) contain ONLY agent and resilience logic
#   - If you want to change output style, you change it in one place
#   - The educational annotations stay in the formatter, not scattered
# ═══════════════════════════════════════════════════════════════════════════════

CONSOLE_OUTPUT_LINE_WIDTH = 84


def print_major_workflow_stage_banner(
    stage_number: int,
    stage_title: str,
    stage_description: str,
) -> None:
    """
    Prints a full-width banner for each major workflow stage.
    Called once per LangGraph node invocation and at initialization.

    Example output:
        ╔══════════════════════════════════════════════════════════╗
        ║  STAGE 2 ▸ SUPERVISOR: FIRST ROUTING DECISION            ║
        ╚══════════════════════════════════════════════════════════╝
    """
    border = "═" * (CONSOLE_OUTPUT_LINE_WIDTH - 2)
    print(f"\n\n╔{border}╗")
    print(f"║  STAGE {stage_number} ▸ {stage_title:<{CONSOLE_OUTPUT_LINE_WIDTH - 12}}║")
    print(f"╚{border}╝")
    print(f"\n  {stage_description}\n")


def print_substage_header(
    stage_number: int,
    substage_number: int,
    substage_title: str,
    substage_explanation: str,
) -> None:
    """
    Prints a sub-stage header within a major stage.
    The stage_number.substage_number format lets you trace exactly where
    execution is in the sequence.
    """
    label = f"[{stage_number}.{substage_number}]"
    print(f"  {label} {substage_title}")
    print(f"  {'─' * (CONSOLE_OUTPUT_LINE_WIDTH - 4)}")
    for explanation_line in substage_explanation.splitlines():
        print(f"         {explanation_line}")
    print()


def print_resilience_stack_inspection_banner(
    agent_display_name: str,
    call_purpose_description: str,
) -> None:
    """
    Prints the header of the resilience stack inspection block.
    Appears before every LLM call to clearly mark that the next output
    describes the 6-layer resilience check.
    """
    inner_width = CONSOLE_OUTPUT_LINE_WIDTH - 4
    print(f"\n  ╔{'═' * inner_width}╗")
    print(f"  ║  RESILIENCE STACK INSPECTION{' ' * (inner_width - 29)}║")
    print(f"  ║  Agent   : {agent_display_name:<{inner_width - 12}}║")
    print(f"  ║  Purpose : {call_purpose_description:<{inner_width - 12}}║")
    print(f"  ╚{'═' * inner_width}╝")


def print_resilience_layer_token_budget(
    stage_number: int,
    substage_number: int,
    token_manager: TokenManager | None,
    estimated_token_count_for_upcoming_call: int,
) -> None:
    """
    Prints Layer 1 (Token Budget Pre-Check) status.

    This is the OUTERMOST layer — cheapest check, no resource consumed yet.
    If the budget is exceeded here, we never touch the bulkhead, rate limiter,
    or circuit breaker. Maximum fail-fast efficiency.
    """
    label = f"[{stage_number}.{substage_number}]"
    print(f"\n  {label} LAYER 1 — Token Budget Pre-Check")
    print(f"         Why outermost: Fail before consuming ANY resource (bulkhead slot,")
    print(f"         rate-limit quota, circuit-breaker check) if budget is exhausted.")

    if token_manager is None:
        print(f"         Status   : SKIPPED — no TokenManager attached to this caller.")
        print(f"                    Token usage for this agent will not be tracked.")
        return

    tokens_consumed_so_far = token_manager._total_in + token_manager._total_out
    workflow_budget_limit = token_manager._max_per_workflow
    tokens_remaining_before_call = token_manager.remaining_budget
    projected_tokens_after_call = tokens_consumed_so_far + estimated_token_count_for_upcoming_call
    utilisation_percentage_now = (tokens_consumed_so_far / workflow_budget_limit * 100) if workflow_budget_limit else 0
    would_exceed_budget = projected_tokens_after_call > workflow_budget_limit

    print(f"         Budget Limit     : {workflow_budget_limit:>8,} tokens total for this workflow run")
    print(f"         Consumed So Far  : {tokens_consumed_so_far:>8,} tokens ({utilisation_percentage_now:.1f}% of budget used)")
    print(f"         Remaining Now    : {tokens_remaining_before_call:>8,} tokens")
    print(f"         Est. This Call   : {estimated_token_count_for_upcoming_call:>8,} tokens (input tokens only, output unknown)")
    print(f"         Projected Total  : {projected_tokens_after_call:>8,} tokens after this call")

    if would_exceed_budget:
        tokens_over_budget = projected_tokens_after_call - workflow_budget_limit
        print(f"         Decision         : BLOCKED — would exceed budget by {tokens_over_budget:,} tokens")
        print(f"                            TokenBudgetExceeded will be raised before making the call.")
    else:
        tokens_headroom_remaining = workflow_budget_limit - projected_tokens_after_call
        print(f"         Decision         : ADMITTED — {tokens_headroom_remaining:,} tokens of headroom remain after this call")


def print_resilience_layer_bulkhead(
    stage_number: int,
    substage_number: int,
    resilient_caller: ResilientCaller,
) -> None:
    """
    Prints Layer 2 (Bulkhead Concurrency Pool) status.

    The bulkhead prevents one agent from exhausting shared thread resources.
    Each ResilientCaller gets its own pool. If a batch job fills its pool,
    the real-time triage agent's pool is unaffected.
    """
    label = f"[{stage_number}.{substage_number}]"
    bulkhead = resilient_caller._bulkhead
    active_calls_in_pool = bulkhead.active_count
    maximum_concurrent_calls_allowed = bulkhead._max_concurrent
    maximum_callers_allowed_to_queue = bulkhead._max_queue
    available_slots = maximum_concurrent_calls_allowed - active_calls_in_pool

    print(f"\n  {label} LAYER 2 — Bulkhead Concurrency Pool Check")
    print(f"         Why here: Acquire pool slot before rate-limit slot. If pool is full,")
    print(f"         reject early without burning a rate-limit quota token.")
    print(f"         Pool Name        : {bulkhead.name}")
    print(f"         Slots Active     : {active_calls_in_pool} of {maximum_concurrent_calls_allowed} maximum concurrent calls")
    print(f"         Slots Available  : {available_slots}")
    print(f"         Queue Depth Limit: {maximum_callers_allowed_to_queue} callers may wait before BulkheadFull is raised")
    print(f"         Queue Timeout    : 5.0s — if no slot opens in 5s, BulkheadFull is raised")

    if active_calls_in_pool >= maximum_concurrent_calls_allowed:
        print(f"         Decision         : QUEUED — pool full; this call waits for a slot (up to 5.0s)")
    else:
        print(f"         Decision         : ADMITTED — acquiring slot {active_calls_in_pool + 1} of {maximum_concurrent_calls_allowed}")


def print_resilience_layer_rate_limiter(
    stage_number: int,
    substage_number: int,
    resilient_caller: ResilientCaller,
) -> None:
    """
    Prints Layer 3 (Rate Limiter Sliding Window) status.

    The rate limiter smooths burst traffic. In a 5-agent parallel voting pattern,
    all 5 agents might call the LLM simultaneously. Without throttling, this burst
    can exhaust a per-second sub-limit even if hourly quota is fine.
    """
    label = f"[{stage_number}.{substage_number}]"
    rate_limiter = resilient_caller._rate_limiter
    calls_per_window = rate_limiter._max_calls
    window_duration_seconds = rate_limiter._period
    is_blocking_mode = rate_limiter._block
    current_recorded_call_count = len(rate_limiter._call_times)

    print(f"\n  {label} LAYER 3 — Rate Limiter Sliding-Window Check")
    print(f"         Why here: Consume quota only after bulkhead admits the call.")
    print(f"         Sliding window avoids burst-at-reset-boundary vulnerabilities.")
    print(f"         Limiter Name     : {rate_limiter.name}")
    print(f"         Quota Config     : {calls_per_window} calls per {window_duration_seconds:.0f} seconds")
    print(f"         Calls in Window  : {current_recorded_call_count} recorded in current {window_duration_seconds:.0f}s window")
    print(f"         Mode             : {'BLOCKING — sleeps until a quota slot opens (default)' if is_blocking_mode else 'NON-BLOCKING — raises RateLimitExhausted immediately'}")
    print(f"         Algorithm        : Sliding window (not fixed reset) prevents burst-at-boundary")

    if current_recorded_call_count >= calls_per_window:
        print(f"         Decision         : THROTTLED — quota full; sleeping until oldest window entry expires")
    else:
        remaining_quota_slots = calls_per_window - current_recorded_call_count
        print(f"         Decision         : ADMITTED — {remaining_quota_slots} quota slots remain in current window")


def print_resilience_layer_circuit_breaker(
    stage_number: int,
    substage_number: int,
    resilient_caller: ResilientCaller,
) -> None:
    """
    Prints Layer 4 (Circuit Breaker State Check) status.

    This is why the circuit breaker sits INSIDE the rate limiter:
    if the API is known-down (OPEN state), we don't want to waste a
    rate-limit slot on a call that will be rejected in <1ms anyway.
    """
    label = f"[{stage_number}.{substage_number}]"
    circuit_breaker = resilient_caller.circuit_breaker
    is_circuit_open = circuit_breaker.is_open
    consecutive_failure_count = circuit_breaker.fail_count
    failure_threshold_to_open = circuit_breaker._breaker.fail_max
    seconds_before_half_open_retry = circuit_breaker._breaker.reset_timeout
    current_state_name = "OPEN — REJECTING ALL CALLS" if is_circuit_open else "CLOSED — healthy"

    print(f"\n  {label} LAYER 4 — Circuit Breaker State Check")
    print(f"         Why here: Fail in <1ms if API is known-unhealthy.")
    print(f"         Avoids waiting 30s for a timeout on a service that's clearly down.")
    print(f"         Breaker Name        : {circuit_breaker.name}")
    print(f"         Current State       : {current_state_name}")
    print(f"         Consecutive Failures: {consecutive_failure_count} of {failure_threshold_to_open} before breaker OPENS")
    print(f"         Reset Timeout       : {seconds_before_half_open_retry}s — breaker stays OPEN this long,")
    print(f"                               then moves to HALF-OPEN to allow one test call through")
    print(f"         Shared Registry     : YES — all agents calling same API share this breaker")
    print(f"                               If this breaker opens, ALL agents fail fast immediately")

    if is_circuit_open:
        print(f"         Decision            : BLOCKED — CircuitBreakerOpen raised; no API call made")
    elif consecutive_failure_count > 0:
        failures_remaining_before_open = failure_threshold_to_open - consecutive_failure_count
        print(f"         Decision            : ADMITTED — {failures_remaining_before_open} more failures would open the circuit")
    else:
        print(f"         Decision            : ADMITTED — no failures recorded; circuit healthy")


def print_resilience_layer_retry_handler(
    stage_number: int,
    substage_number: int,
    resilient_caller: ResilientCaller,
) -> None:
    """
    Prints Layer 5 (Retry Handler) configuration.

    The retry handler wraps the circuit-breaker-protected call, not the raw call.
    This is deliberate: each retry attempt goes through the circuit breaker.
    If the first attempt fails and opens the circuit, retry attempt 2 is
    rejected immediately — the circuit breaker prevents wasted retries.
    """
    label = f"[{stage_number}.{substage_number}]"
    retry_config = resilient_caller._retry_handler._config
    max_total_attempts = retry_config.max_retries + 1
    backoff_sequence_preview = [
        round(min(retry_config.initial_wait * (2 ** attempt_index), retry_config.max_wait), 1)
        for attempt_index in range(retry_config.max_retries)
    ]

    print(f"\n  {label} LAYER 5 — Retry Handler with Exponential Backoff + Jitter")
    print(f"         Why here: Wraps circuit breaker so each retry goes THROUGH the breaker.")
    print(f"         If circuit opens mid-retry sequence, remaining retries fail fast.")
    print(f"         Max Attempts         : {max_total_attempts} total ({retry_config.max_retries} retries after initial failure)")
    print(f"         Backoff Sequence     : {' → '.join(str(s) + 's' for s in backoff_sequence_preview)} (before jitter)")
    print(f"         Jitter Range         : ±{retry_config.jitter}s random offset per wait")
    print(f"                               Prevents thundering herd when many agents retry simultaneously")
    print(f"         Max Single Wait      : {retry_config.max_wait}s cap")
    print(f"         Retries On (transient): ConnectionError, TimeoutError, RateLimitError,")
    print(f"                                APIConnectionError, APITimeoutError")
    print(f"         No Retry On (permanent): AuthenticationError, InvalidRequestError,")
    print(f"                                  ContentPolicyError — retrying won't fix these")
    print(f"         Decision             : WRAPPING — monitoring for transient exceptions")


def print_resilience_layer_timeout_guard(
    stage_number: int,
    substage_number: int,
    resilient_caller: ResilientCaller,
    per_call_timeout_override_seconds: float | None,
) -> None:
    """
    Prints Layer 6 (Timeout Guard) configuration.

    Timeout guard is INNERMOST because the deadline applies per ATTEMPT,
    not per retry sequence. 3 retries × 30s deadline = 90s max wall time.
    If timeout were outside retry, 1 × 30s = 30s max regardless of retries.
    """
    label = f"[{stage_number}.{substage_number}]"
    default_timeout_seconds = resilient_caller._timeout_guard._default_timeout
    effective_timeout_seconds = (
        per_call_timeout_override_seconds
        if per_call_timeout_override_seconds is not None
        else default_timeout_seconds
    )
    is_using_override = per_call_timeout_override_seconds is not None

    print(f"\n  {label} LAYER 6 — Timeout Guard (Hard Deadline per Attempt)")
    print(f"         Why innermost: Deadline applies per attempt, not per retry sequence.")
    print(f"         One hung call doesn't block the entire downstream pipeline.")
    print(f"         Default Deadline     : {default_timeout_seconds}s")
    print(f"         Effective Deadline   : {effective_timeout_seconds}s {'(OVERRIDE applied)' if is_using_override else '(using default)'}")
    print(f"         Implementation       : ThreadPoolExecutor + Future.result(timeout=N)")
    print(f"                               Works in ANY thread — safe inside LangGraph worker threads")
    print(f"                               signal.alarm NOT used (only works on main thread, POSIX-only)")
    print(f"         On Deadline Exceeded : TimeoutExceeded raised; background thread eventually cleans up")
    print(f"         Decision             : WRAPPING — {effective_timeout_seconds}s deadline enforced on the LLM call")


def print_llm_call_execution_and_result(
    stage_number: int,
    substage_number: int,
    agent_display_name: str,
    full_llm_response_text: str,
    actual_input_tokens: int,
    actual_output_tokens: int,
    elapsed_wall_time_seconds: float,
    token_manager: TokenManager | None,
) -> None:
    """
    Prints the full LLM response and post-call token accounting.
    Response text is NEVER truncated — full output is always shown.
    """
    label = f"[{stage_number}.{substage_number}]"
    total_tokens_this_call = actual_input_tokens + actual_output_tokens

    print(f"\n  {label} LLM CALL COMPLETED — {agent_display_name}")
    print(f"         Elapsed Time   : {elapsed_wall_time_seconds:.2f}s")
    print(f"         Tokens In      : {actual_input_tokens:,} (prompt / input tokens)")
    print(f"         Tokens Out     : {actual_output_tokens:,} (completion / output tokens)")
    print(f"         Tokens Total   : {total_tokens_this_call:,} for this call")

    if token_manager is not None:
        updated_workflow_total = token_manager._total_in + token_manager._total_out
        updated_remaining = token_manager.remaining_budget
        updated_utilisation = (updated_workflow_total / token_manager._max_per_workflow * 100)
        print(f"         Workflow Total : {updated_workflow_total:,} tokens consumed across ALL agents so far")
        print(f"         Budget Remaining: {updated_remaining:,} tokens ({100 - updated_utilisation:.1f}% headroom)")

    inner_width = CONSOLE_OUTPUT_LINE_WIDTH - 4
    print(f"\n  ╔{'═' * inner_width}╗")
    print(f"  ║  FULL LLM RESPONSE (untruncated){' ' * (inner_width - 33)}║")
    print(f"  ╠{'═' * inner_width}╣")

    for response_line in full_llm_response_text.splitlines():
        # Wrap long lines at inner_width - 4 to fit inside the box
        remaining_line_content = response_line if response_line else " "
        while len(remaining_line_content) > inner_width - 4:
            print(f"  ║  {remaining_line_content[:inner_width - 4]}  ║")
            remaining_line_content = remaining_line_content[inner_width - 4:]
        padded = remaining_line_content.ljust(inner_width - 4)
        print(f"  ║  {padded}  ║")

    print(f"  ╚{'═' * inner_width}╝")


def print_workflow_state_snapshot(
    state_snapshot_label: str,
    workflow_state: dict,
) -> None:
    """
    Prints the current LangGraph state at key decision points.
    Called before every supervisor routing decision so you can see
    exactly what data the supervisor is reasoning over.
    """
    inner_width = CONSOLE_OUTPUT_LINE_WIDTH - 4
    print(f"\n  ╔{'═' * inner_width}╗")
    print(f"  ║  CURRENT WORKFLOW STATE SNAPSHOT — {state_snapshot_label:<{inner_width - 37}}║")
    print(f"  ╠{'═' * inner_width}╣")

    triage_result = workflow_state.get("medical_triage_classification_result") or "Not yet collected"
    cardiovascular_result = workflow_state.get("cardiovascular_differential_diagnosis_result") or "Not yet collected"
    error_state = workflow_state.get("workflow_resilience_error_state") or "None — no errors so far"
    message_count = len(workflow_state.get("accumulated_conversation_messages", []))

    fields_to_display = [
        ("Messages in History", f"{message_count} message(s) accumulated"),
        ("Triage Result", triage_result[:80] + "..." if len(triage_result) > 80 else triage_result),
        ("Cardiovascular Analysis", cardiovascular_result[:80] + "..." if len(cardiovascular_result) > 80 else cardiovascular_result),
        ("Resilience Error State", error_state),
    ]

    for field_label, field_value in fields_to_display:
        padding = inner_width - len(field_label) - len(str(field_value)) - 5
        if padding < 0:
            padding = 0
        print(f"  ║  {field_label}: {str(field_value)[:inner_width - len(field_label) - 4]}{' ' * max(0, inner_width - len(field_label) - min(len(str(field_value)), inner_width - len(field_label) - 4) - 4)}║")

    print(f"  ╚{'═' * inner_width}╝\n")


def print_final_resilience_and_token_usage_report(
    per_workflow_token_budget_manager: TokenManager,
) -> None:
    """
    Prints the complete end-of-workflow resilience health report.
    Includes circuit breaker state for all registered breakers,
    per-agent token breakdown, and overall budget utilisation.
    """
    inner_width = CONSOLE_OUTPUT_LINE_WIDTH - 4
    print(f"\n  ╔{'═' * inner_width}╗")
    print(f"  ║  CIRCUIT BREAKER REGISTRY — HEALTH REPORT{' ' * (inner_width - 43)}║")
    print(f"  ╠{'═' * inner_width}╣")

    all_circuit_breaker_statuses = CircuitBreakerRegistry.get_all_statuses()
    if not all_circuit_breaker_statuses:
        print(f"  ║  No circuit breakers registered.{' ' * (inner_width - 34)}║")
    else:
        for breaker_name, breaker_status_dict in all_circuit_breaker_statuses.items():
            state_indicator = "🔴 OPEN" if breaker_status_dict["state"] == "open" else "🟢 CLOSED"
            line = f"  ║  {breaker_name}: {state_indicator}, failures={breaker_status_dict['fail_count']}"
            print(f"{line}{' ' * (inner_width + 4 - len(line))}║")

    print(f"  ╠{'═' * inner_width}╣")
    print(f"  ║  TOKEN BUDGET — WORKFLOW SUMMARY{' ' * (inner_width - 34)}║")
    print(f"  ╠{'═' * inner_width}╣")

    workflow_summary = per_workflow_token_budget_manager.get_workflow_summary()
    summary_rows = [
        ("Total Tokens In  (all agents)", f"{workflow_summary['total_tokens_in']:,}"),
        ("Total Tokens Out (all agents)", f"{workflow_summary['total_tokens_out']:,}"),
        ("Total Tokens     (in + out)  ", f"{workflow_summary['total_tokens']:,}"),
        ("Budget Limit                 ", f"{workflow_summary['budget_limit']:,}"),
        ("Budget Remaining             ", f"{workflow_summary['remaining']:,}"),
        ("Budget Utilisation           ", f"{workflow_summary['utilization_pct']:.1f}%"),
    ]
    for row_label, row_value in summary_rows:
        line = f"  ║  {row_label}: {row_value}"
        print(f"{line}{' ' * (inner_width + 4 - len(line))}║")

    per_agent_usage_list = per_workflow_token_budget_manager.get_all_agents_summary()
    if per_agent_usage_list:
        print(f"  ╠{'═' * inner_width}╣")
        print(f"  ║  PER-AGENT TOKEN BREAKDOWN{' ' * (inner_width - 28)}║")
        print(f"  ╠{'═' * inner_width}╣")
        for agent_usage_dict in per_agent_usage_list:
            agent_row = (
                f"  ║  {agent_usage_dict['agent']}: "
                f"{agent_usage_dict['total_tokens']:,} total "
                f"({agent_usage_dict['tokens_in']:,} in + {agent_usage_dict['tokens_out']:,} out, "
                f"{agent_usage_dict['calls']} call(s))"
            )
            print(f"{agent_row}{' ' * max(0, inner_width + 4 - len(agent_row))}║")

    print(f"  ╚{'═' * inner_width}╝")


def print_agent_transition(source_agent: str, destination_agent: str) -> None:
    """
    Prints a clear visual indicator when execution moves from one agent to another.
    """
    inner_width = CONSOLE_OUTPUT_LINE_WIDTH - 4
    print(f"\n  {'▼' * inner_width}")
    print(f"  🔄 TRANSITION : {source_agent}")
    print(f"           └──► : {destination_agent}")
    print(f"  {'▲' * inner_width}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RESILIENCE STACK PRE-CALL INSPECTOR
#
# ResilienceStackPreCallInspector reads the live internal state of every
# resilience component inside a ResilientCaller and prints a numbered,
# layer-by-layer diagnostic BEFORE the call is made.
#
# Design principle: this class ONLY reads state (never mutates it).
# It is purely an observability and learning tool.
# ═══════════════════════════════════════════════════════════════════════════════

class ResilienceStackPreCallInspector:
    """
    Reads the live state of all 6 resilience layers inside a ResilientCaller
    and prints a detailed, numbered inspection for each layer.

    One inspector is created per LLM call (not per agent). It maintains a
    substage counter so each printed layer gets a unique [stage.substage] label.

    Args:
        resilient_caller_to_inspect: The ResilientCaller whose state to read.
        workflow_stage_number:       The current major workflow stage number.
        starting_substage_number:    First substage number to use (allows
                                     callers to offset if some substages were
                                     printed before the stack inspection).
        agent_display_name:          Human-readable agent name for the banner.
    """

    def __init__(
        self,
        resilient_caller_to_inspect: ResilientCaller,
        workflow_stage_number: int,
        starting_substage_number: int,
        agent_display_name: str,
    ) -> None:
        self._resilient_caller = resilient_caller_to_inspect
        self._stage_number = workflow_stage_number
        self._current_substage_number = starting_substage_number
        self._agent_display_name = agent_display_name

    def _advance_and_get_substage_number(self) -> int:
        current = self._current_substage_number
        self._current_substage_number += 1
        return current

    def inspect_and_print_all_six_layers(
        self,
        estimated_input_token_count: int,
        call_purpose_description: str,
        per_call_timeout_override_seconds: float | None = None,
    ) -> int:
        """
        Prints the complete 6-layer stack inspection.

        Layers printed in order (outermost to innermost):
            1. Token Budget Pre-Check
            2. Bulkhead Concurrency Pool
            3. Rate Limiter Sliding Window
            4. Circuit Breaker State
            5. Retry Handler Configuration
            6. Timeout Guard Deadline

        Args:
            estimated_input_token_count:       Estimated prompt tokens for Layer 1 check.
            call_purpose_description:          Shown in the banner for context.
            per_call_timeout_override_seconds: Shown in Layer 6 if a custom deadline is used.

        Returns:
            The next available substage number (so the caller can continue numbering).
        """
        print_resilience_stack_inspection_banner(
            agent_display_name=self._agent_display_name,
            call_purpose_description=call_purpose_description,
        )

        print_resilience_layer_token_budget(
            stage_number=self._stage_number,
            substage_number=self._advance_and_get_substage_number(),
            token_manager=self._resilient_caller.token_manager,
            estimated_token_count_for_upcoming_call=estimated_input_token_count,
        )

        print_resilience_layer_bulkhead(
            stage_number=self._stage_number,
            substage_number=self._advance_and_get_substage_number(),
            resilient_caller=self._resilient_caller,
        )

        print_resilience_layer_rate_limiter(
            stage_number=self._stage_number,
            substage_number=self._advance_and_get_substage_number(),
            resilient_caller=self._resilient_caller,
        )

        print_resilience_layer_circuit_breaker(
            stage_number=self._stage_number,
            substage_number=self._advance_and_get_substage_number(),
            resilient_caller=self._resilient_caller,
        )

        print_resilience_layer_retry_handler(
            stage_number=self._stage_number,
            substage_number=self._advance_and_get_substage_number(),
            resilient_caller=self._resilient_caller,
        )

        print_resilience_layer_timeout_guard(
            stage_number=self._stage_number,
            substage_number=self._advance_and_get_substage_number(),
            resilient_caller=self._resilient_caller,
            per_call_timeout_override_seconds=per_call_timeout_override_seconds,
        )

        print(f"\n  {'─' * (CONSOLE_OUTPUT_LINE_WIDTH - 4)}")
        print(f"  Passing through all 6 layers... invoking {self._agent_display_name}")
        print(f"  {'─' * (CONSOLE_OUTPUT_LINE_WIDTH - 4)}")

        return self._current_substage_number


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LANGGRAPH STATE DEFINITION
#
# TypedDict defines the shape of the graph's shared state. Every node reads
# from and writes to this state. The Annotated[list, add_messages] annotation
# tells LangGraph to APPEND new messages rather than replace the whole list.
#
# WHY THESE FIELDS:
#   accumulated_conversation_messages       — full message history for context
#   supervisor_next_agent_routing_decision  — supervisor writes this; router reads it
#   medical_triage_classification_result   — MedicalSymptomTriageAgent output
#   cardiovascular_differential_diagnosis_result — CardiovascularAgent output
#   workflow_resilience_error_state        — any resilience exception message
#   current_workflow_execution_stage_number — for correlated log output
# ═══════════════════════════════════════════════════════════════════════════════

class ClinicalWorkflowGraphState(TypedDict):
    accumulated_conversation_messages: Annotated[list[BaseMessage], add_messages]
    supervisor_next_agent_routing_decision: str
    medical_triage_classification_result: str | None
    cardiovascular_differential_diagnosis_result: str | None
    workflow_resilience_error_state: str | None
    current_workflow_execution_stage_number: int


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SHARED RESILIENCE INFRASTRUCTURE (INITIALISED ONCE AT BOOT)
#
# These objects are created ONCE when the module loads and shared across all
# workflow executions in the process lifetime. This is intentional:
#
#   shared_llm_api_circuit_breaker:
#       All three agents (Supervisor, Triage, Cardiovascular) call the same
#       LLM API. They share one circuit breaker. If the Triage agent causes
#       5 consecutive failures, the breaker opens — and the Supervisor's NEXT
#       call is rejected in <1ms without ever reaching the API.
#       One failure history, shared across the entire pipeline.
#
#   prompt_token_estimator:
#       Stateless after init. Created once, reused for every token estimation.
#       Uses tiktoken cl100k_base encoding.
#
# PER-WORKFLOW OBJECTS (created fresh per main() call in production):
#   per_workflow_token_budget_manager:
#       MUST be fresh per workflow execution. If reused, the cumulative counter
#       grows unboundedly, causing false budget-exceeded errors on later runs.
#       In a server environment, create this per request and store in state.
# ═══════════════════════════════════════════════════════════════════════════════

shared_llm_api_circuit_breaker = CircuitBreakerRegistry.get_or_create(
    "clinical_llm_api",
    CircuitBreakerConfig(
        fail_max=5,          # 5 consecutive failures before opening
        reset_timeout=60,    # stay OPEN for 60s, then try HALF-OPEN
    ),
)

prompt_token_estimator = TokenCounter(model="gpt-4o")

# Created fresh per workflow execution (see SECTION 4 note above)
per_workflow_token_budget_manager = TokenManager(
    TokenBudgetConfig(
        max_tokens_per_workflow=12_000,   # ~$0.04–0.12 at current model pricing
        max_tokens_per_agent=4_096,       # per-agent cap (currently checked at workflow level)
    )
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PER-AGENT RESILIENT CALLERS
#
# Each agent gets its own ResilientCaller because:
#
#   Different timeout requirements:
#       The Supervisor makes short routing decisions (15s deadline is generous).
#       The Cardiovascular Specialist does deep differential diagnosis (30s needed).
#
#   Different concurrency limits (bulkhead):
#       Supervisor is called after EVERY other agent — higher throughput needed.
#       Triage handles real-time user requests — moderate concurrency.
#       Cardiovascular Specialist does intensive analysis — small pool, quality over speed.
#
#   Shared circuit breaker (all agents):
#       All three share `shared_llm_api_circuit_breaker`. One shared failure history.
#
#   Shared token manager (all agents):
#       All three share `per_workflow_token_budget_manager`. One shared budget.
# ═══════════════════════════════════════════════════════════════════════════════

supervisor_orchestrator_resilient_caller = ResilientCaller(
    config=ResilienceConfig(
        timeout=TimeoutConfig(default_timeout=15.0),
        bulkhead=BulkheadConfig(max_concurrent=10, max_queue=20),
    ),
    agent_name="ClinicalWorkflowSupervisorOrchestrator",
    circuit_breaker=shared_llm_api_circuit_breaker,
    token_manager=per_workflow_token_budget_manager,
)

medical_triage_extraction_resilient_caller = ResilientCaller(
    config=ResilienceConfig(
        timeout=TimeoutConfig(default_timeout=20.0),
        bulkhead=BulkheadConfig(max_concurrent=5, max_queue=10),
    ),
    agent_name="MedicalSymptomTriageAndExtractionAgent",
    circuit_breaker=shared_llm_api_circuit_breaker,
    token_manager=per_workflow_token_budget_manager,
)

cardiovascular_diagnosis_resilient_caller = ResilientCaller(
    config=ResilienceConfig(
        timeout=TimeoutConfig(default_timeout=30.0),
        bulkhead=BulkheadConfig(max_concurrent=3, max_queue=5),
    ),
    agent_name="CardiovascularDifferentialDiagnosisAgent",
    circuit_breaker=shared_llm_api_circuit_breaker,
    token_manager=per_workflow_token_budget_manager,
)

# One shared LLM instance for all agents
# Temperature=0.0 for deterministic, reproducible outputs in a medical context
clinical_llm_instance = get_llm(temperature=0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CORE LLM INVOCATION FUNCTION
#
# invoke_llm_through_complete_resilience_stack is called by every agent node.
# It handles the complete lifecycle of one LLM call:
#   1. Builds message list (system + human messages)
#   2. Estimates input tokens for Layer 1 pre-check
#   3. Runs the ResilienceStackPreCallInspector (prints all 6 layers)
#   4. Calls caller.call(llm.invoke, messages, estimated_tokens=N)
#   5. Extracts actual token usage from response metadata
#   6. Records actual usage in the shared TokenManager
#   7. Prints the full response and post-call accounting
#   8. Returns (response_text, error_message_or_none, elapsed_seconds)
# ═══════════════════════════════════════════════════════════════════════════════

def invoke_llm_through_complete_resilience_stack(
    system_role_instruction: str,
    human_turn_prompt_content: str,
    resilient_caller_for_this_agent: ResilientCaller,
    agent_display_name_for_logging: str,
    call_purpose_description_for_logging: str,
    workflow_stage_number: int,
    substage_number_to_start_at: int,
    per_call_timeout_override_seconds: float | None = None,
) -> tuple[str | None, str | None, float]:
    """
    Executes a single LLM call through the full 6-layer resilience stack.

    Before making the call, this function prints the complete stack inspection
    so you can observe what each layer checks and what decision it makes.
    After the call, it prints the full response and updated token accounting.

    Args:
        system_role_instruction:               System prompt defining the agent's role.
        human_turn_prompt_content:             The actual task/question for this call.
        resilient_caller_for_this_agent:       The caller with this agent's resilience config.
        agent_display_name_for_logging:        Human-readable name shown in console output.
        call_purpose_description_for_logging:  Brief description of why this call is made.
        workflow_stage_number:                 Current major stage (for [N.M] labelling).
        substage_number_to_start_at:           Starting substage number for layer labels.
        per_call_timeout_override_seconds:     Optional per-call deadline override.

    Returns:
        Tuple of (response_text_or_none, error_message_or_none, elapsed_seconds).
        response_text is None if a resilience layer blocked or the call failed.
        error_message is None on success; contains exception description on failure.
    """
    messages_for_llm = []
    if system_role_instruction:
        messages_for_llm.append(SystemMessage(content=system_role_instruction))
    messages_for_llm.append(HumanMessage(content=human_turn_prompt_content))

    # Estimate input token count for Layer 1 (Token Budget Pre-Check)
    # We estimate from the serialized message content. Output tokens are unknown before the call.
    combined_message_text_for_estimation = system_role_instruction + " " + human_turn_prompt_content
    estimated_input_token_count = prompt_token_estimator.count(combined_message_text_for_estimation)

    # Run the pre-call stack inspection (prints all 6 layers)
    stack_inspector = ResilienceStackPreCallInspector(
        resilient_caller_to_inspect=resilient_caller_for_this_agent,
        workflow_stage_number=workflow_stage_number,
        starting_substage_number=substage_number_to_start_at,
        agent_display_name=agent_display_name_for_logging,
    )
    next_available_substage_number = stack_inspector.inspect_and_print_all_six_layers(
        estimated_input_token_count=estimated_input_token_count,
        call_purpose_description=call_purpose_description_for_logging,
        per_call_timeout_override_seconds=per_call_timeout_override_seconds,
    )

    # ── Execute the LLM call through the resilience stack ──────────────────
    call_start_wall_time = time.perf_counter()

    try:
        llm_response_object = resilient_caller_for_this_agent.call(
            clinical_llm_instance.invoke,
            messages_for_llm,
            estimated_tokens=estimated_input_token_count,
            timeout=per_call_timeout_override_seconds,
        )

        elapsed_seconds = time.perf_counter() - call_start_wall_time

        # ── Extract actual token counts from provider response metadata ──
        # LangChain models return usage_metadata with 'input_tokens' and 'output_tokens'.
        # If the model does not return this (some providers don't), fall back to estimates.
        response_usage_metadata = getattr(llm_response_object, "usage_metadata", {}) or {}
        actual_input_tokens_consumed = response_usage_metadata.get(
            "input_tokens", estimated_input_token_count
        )
        actual_output_tokens_consumed = response_usage_metadata.get(
            "output_tokens", max(50, len(str(llm_response_object.content)) // 4)
        )

        # ── Record actual usage in the shared TokenManager ──────────────
        # This updates the cumulative counter that Layer 1 reads on the NEXT call.
        if resilient_caller_for_this_agent.token_manager is not None:
            resilient_caller_for_this_agent.token_manager.record_usage(
                agent_name=agent_display_name_for_logging,
                tokens_in=actual_input_tokens_consumed,
                tokens_out=actual_output_tokens_consumed,
            )

        response_text_content = str(llm_response_object.content)

        print_llm_call_execution_and_result(
            stage_number=workflow_stage_number,
            substage_number=next_available_substage_number,
            agent_display_name=agent_display_name_for_logging,
            full_llm_response_text=response_text_content,
            actual_input_tokens=actual_input_tokens_consumed,
            actual_output_tokens=actual_output_tokens_consumed,
            elapsed_wall_time_seconds=elapsed_seconds,
            token_manager=resilient_caller_for_this_agent.token_manager,
        )

        return response_text_content, None, elapsed_seconds

    except CircuitBreakerOpen as circuit_breaker_open_exception:
        elapsed_seconds = time.perf_counter() - call_start_wall_time
        error_description = (
            f"CircuitBreakerOpen: {circuit_breaker_open_exception.message} | "
            f"Breaker details: {circuit_breaker_open_exception.details}"
        )
        print(f"\n  [{workflow_stage_number}.{next_available_substage_number}] RESILIENCE LAYER BLOCKED — Circuit Breaker OPEN")
        print(f"         The circuit breaker rejected this call in <1ms.")
        print(f"         No API call was made. No timeout was incurred.")
        print(f"         Error: {circuit_breaker_open_exception.message}")
        print(f"         Details: {circuit_breaker_open_exception.details}")
        print(f"         Recovery: The breaker resets after its reset_timeout. Do not retry manually.")
        return None, error_description, elapsed_seconds

    except TimeoutExceeded as timeout_exceeded_exception:
        elapsed_seconds = time.perf_counter() - call_start_wall_time
        error_description = f"TimeoutExceeded: {timeout_exceeded_exception.message}"
        print(f"\n  [{workflow_stage_number}.{next_available_substage_number}] RESILIENCE LAYER BLOCKED — Timeout Guard")
        print(f"         The call exceeded its {per_call_timeout_override_seconds or resilient_caller_for_this_agent._timeout_guard._default_timeout}s deadline.")
        print(f"         Error: {timeout_exceeded_exception.message}")
        print(f"         Details: {timeout_exceeded_exception.details}")
        return None, error_description, elapsed_seconds

    except TokenBudgetExceeded as token_budget_exceeded_exception:
        elapsed_seconds = time.perf_counter() - call_start_wall_time
        error_description = f"TokenBudgetExceeded: {token_budget_exceeded_exception.message}"
        print(f"\n  [{workflow_stage_number}.{next_available_substage_number}] RESILIENCE LAYER BLOCKED — Token Budget")
        print(f"         The token budget pre-check (Layer 1) rejected this call.")
        print(f"         No API call was made. No tokens consumed.")
        print(f"         Error: {token_budget_exceeded_exception.message}")
        print(f"         Details: {token_budget_exceeded_exception.details}")
        return None, error_description, elapsed_seconds

    except Exception as unexpected_exception:
        elapsed_seconds = time.perf_counter() - call_start_wall_time
        error_description = f"UnexpectedError: {type(unexpected_exception).__name__}: {str(unexpected_exception)}"
        print(f"\n  [{workflow_stage_number}.{next_available_substage_number}] UNEXPECTED ERROR")
        print(f"         {error_description}")
        import traceback
        traceback.print_exc()
        return None, error_description, elapsed_seconds


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — LANGGRAPH NODE DEFINITIONS
#
# Three nodes, each representing one agent in the workflow:
#
#   ClinicalWorkflowSupervisorOrchestrator:
#       Reads the current state and decides which agent should run next.
#       Routing logic: Triage first → Cardiovascular if triage done → FINISH.
#       This node runs BEFORE and AFTER every specialist agent.
#
#   MedicalSymptomTriageAndExtractionAgent:
#       Extracts structured clinical facts from the raw patient query:
#       symptoms, onset, duration, severity, relevant history, urgency level.
#       Outputs a structured triage summary for the specialist to reason over.
#
#   CardiovascularDifferentialDiagnosisAgent:
#       Receives the structured triage data and performs deep clinical analysis:
#       differential diagnosis (most to least likely), recommended immediate
#       workup, red flags, and suggested disposition.
# ═══════════════════════════════════════════════════════════════════════════════

def clinical_workflow_supervisor_orchestrator_node(
    state: ClinicalWorkflowGraphState,
) -> dict:
    """
    LangGraph node for the ClinicalWorkflowSupervisorOrchestrator.

    The supervisor has three possible routing outcomes:
        → MedicalSymptomTriageAndExtractionAgent   (triage not done yet)
        → CardiovascularDifferentialDiagnosisAgent (triage done, specialist not done)
        → FINISH                                    (all agents completed or error)

    RESILIENCE BEHAVIOUR:
        If CircuitBreakerOpen:  route to FINISH — cannot make any LLM decisions
        If TokenBudgetExceeded: route to FINISH — budget exhausted
        If TimeoutExceeded:     route to FINISH — supervisor hung, abort workflow
    """
    current_stage_number = state.get("current_workflow_execution_stage_number", 1)

    print_major_workflow_stage_banner(
        stage_number=current_stage_number,
        stage_title="ClinicalWorkflowSupervisorOrchestrator — Routing Decision",
        stage_description=(
            "The supervisor reads the current workflow state and decides which\n"
            "  agent should execute next. It makes this decision via an LLM call,\n"
            "  which means it ALSO goes through the full 6-layer resilience stack.\n"
            "  This demonstrates that resilience protects orchestration logic too,\n"
            "  not just the specialist agents."
        ),
    )

    # ── Abort if a resilience error already occurred upstream ───────────────
    if state.get("workflow_resilience_error_state"):
        print(f"  [SUPERVISOR] Resilience error detected in state. Routing to FINISH.")
        print(f"  [SUPERVISOR] Error: {state['workflow_resilience_error_state']}")
        return {
            "supervisor_next_agent_routing_decision": "FINISH",
            "current_workflow_execution_stage_number": current_stage_number + 1,
        }

    print_workflow_state_snapshot(
        state_snapshot_label="Before Supervisor Routing Decision",
        workflow_state=state,
    )

    # ── Build the supervisor's routing prompt ────────────────────────────────
    patient_query_text = state["accumulated_conversation_messages"][0].content
    has_triage = "NO" if not state.get("medical_triage_classification_result") else "YES"
    has_cardio = "NO" if not state.get("cardiovascular_differential_diagnosis_result") else "YES"

    supervisor_routing_prompt = f"""
You are the ClinicalWorkflowSupervisorOrchestrator. Your job is to decide which agent should execute next based on the current workflow state.

CURRENT WORKFLOW STATE:
  Medical Triage Completed: {has_triage}
  Cardiovascular Analysis Completed: {has_cardio}

ROUTING RULES (apply in order):
  Rule 1: If Medical Triage Completed is NO → route to MedicalSymptomTriageAndExtractionAgent
  Rule 2: If Medical Triage Completed is YES AND Cardiovascular Analysis Completed is NO → route to CardiovascularDifferentialDiagnosisAgent
  Rule 3: If both are YES → route to FINISH

IMPORTANT: Reply with EXACTLY ONE of these three options, and nothing else:
  MedicalSymptomTriageAndExtractionAgent
  CardiovascularDifferentialDiagnosisAgent
  FINISH
""".strip()

    supervisor_system_instruction = (
        "You are ClinicalWorkflowSupervisorOrchestrator. "
        "You make routing decisions for a medical multi-agent workflow. "
        "You reply with exactly one routing target and nothing else."
    )

    response_text, error_message, elapsed_seconds = invoke_llm_through_complete_resilience_stack(
        system_role_instruction=supervisor_system_instruction,
        human_turn_prompt_content=supervisor_routing_prompt,
        resilient_caller_for_this_agent=supervisor_orchestrator_resilient_caller,
        agent_display_name_for_logging="ClinicalWorkflowSupervisorOrchestrator",
        call_purpose_description_for_logging="Determine which specialist agent should run next",
        workflow_stage_number=current_stage_number,
        substage_number_to_start_at=1,
    )

    if error_message:
        print(f"\n  [SUPERVISOR] Resilience error during routing decision: {error_message}")
        print(f"  [SUPERVISOR] Cannot determine routing safely. Sending workflow to FINISH.")
        return {
            "supervisor_next_agent_routing_decision": "FINISH",
            "workflow_resilience_error_state": error_message,
            "current_workflow_execution_stage_number": current_stage_number + 1,
        }

    # ── Parse the routing decision from the response ─────────────────────────
    # The LLM was instructed to reply with exactly one routing target.
    # We do a substring check to be robust against minor formatting variations.
    routing_decision_text = response_text.strip() if response_text else ""
    if "CardiovascularDifferentialDiagnosisAgent" in routing_decision_text:
        next_routing_destination = "CardiovascularDifferentialDiagnosisAgent"
    elif "MedicalSymptomTriageAndExtractionAgent" in routing_decision_text:
        next_routing_destination = "MedicalSymptomTriageAndExtractionAgent"
    else:
        next_routing_destination = "FINISH"

    print(f"\n  [SUPERVISOR DECISION] Routing to: {next_routing_destination}")
    print(f"  [SUPERVISOR] Elapsed for routing call: {elapsed_seconds:.2f}s")

    return {
        "supervisor_next_agent_routing_decision": next_routing_destination,
        "current_workflow_execution_stage_number": current_stage_number + 1,
    }


def medical_symptom_triage_and_extraction_agent_node(
    state: ClinicalWorkflowGraphState,
) -> dict:
    """
    LangGraph node for the MedicalSymptomTriageAndExtractionAgent.

    Extracts structured clinical information from the raw patient query:
        - Primary presenting symptom(s) with onset, duration, severity
        - Associated symptoms
        - Relevant past medical history / risk factors
        - Urgency classification (Immediate / Urgent / Semi-urgent / Non-urgent)
        - Key clinical concerns for specialist review

    This structured output is stored in `medical_triage_classification_result`
    and becomes the primary input for the CardiovascularDifferentialDiagnosisAgent.

    RESILIENCE NOTE:
        This agent shares the circuit breaker with the Supervisor.
        If this agent caused failures that opened the circuit, the SUPERVISOR's
        next routing call would also fail fast — demonstrating cross-agent
        resilience sharing.
    """
    current_stage_number = state.get("current_workflow_execution_stage_number", 2)

    print_major_workflow_stage_banner(
        stage_number=current_stage_number,
        stage_title="MedicalSymptomTriageAndExtractionAgent — Clinical Triage",
        stage_description=(
            "The Triage Agent extracts structured clinical facts from the raw patient\n"
            "  query. Its output — symptoms, onset, severity, risk factors, urgency —\n"
            "  is stored in state and passed to the Cardiovascular Specialist.\n"
            "  Note: same circuit breaker as Supervisor. Token budget continues\n"
            "  accumulating from Supervisor's earlier call."
        ),
    )

    patient_query_text = state["accumulated_conversation_messages"][0].content

    triage_extraction_prompt = f"""
You are the MedicalSymptomTriageAndExtractionAgent. Extract and structure all clinically relevant information from the patient presentation below.

PATIENT PRESENTATION:
{patient_query_text}

Provide your triage output in the following structured format:

PRESENTING COMPLAINT:
  [Primary symptom with onset, duration, character, severity, radiation, timing, aggravating/relieving factors]

ASSOCIATED SYMPTOMS:
  [List each associated symptom with duration and severity]

VITAL CONTEXT:
  [Patient demographics, relevant history, current medications if mentioned]

RISK FACTORS IDENTIFIED:
  [List each cardiovascular, metabolic, or relevant risk factor]

URGENCY CLASSIFICATION:
  [Immediate Life-Threatening / Urgent / Semi-Urgent / Non-Urgent]
  [Rationale for classification]

KEY CLINICAL CONCERNS FOR SPECIALIST:
  [Numbered list of the most important clinical questions this presentation raises]

DIFFERENTIAL PROMPTS:
  [Top 3–5 diagnoses that should be considered, briefly listed for the specialist]
""".strip()

    triage_system_instruction = (
        "You are MedicalSymptomTriageAndExtractionAgent. "
        "You are a senior emergency medicine triage nurse/physician assistant. "
        "Extract and structure ALL clinically relevant information from patient presentations. "
        "Be thorough and precise. Use standard clinical terminology."
    )

    response_text, error_message, elapsed_seconds = invoke_llm_through_complete_resilience_stack(
        system_role_instruction=triage_system_instruction,
        human_turn_prompt_content=triage_extraction_prompt,
        resilient_caller_for_this_agent=medical_triage_extraction_resilient_caller,
        agent_display_name_for_logging="MedicalSymptomTriageAndExtractionAgent",
        call_purpose_description_for_logging="Extract structured clinical triage data from patient query",
        workflow_stage_number=current_stage_number,
        substage_number_to_start_at=1,
    )

    if error_message:
        print(f"\n  [TRIAGE AGENT] Resilience error: {error_message}")
        print(f"  [TRIAGE AGENT] Storing error in state. Supervisor will route to FINISH.")
        return {
            "workflow_resilience_error_state": f"Triage Agent Failed: {error_message}",
            "current_workflow_execution_stage_number": current_stage_number + 1,
        }

    print(f"\n  [TRIAGE AGENT] Completed in {elapsed_seconds:.2f}s. Triage data stored in state.")
    print(f"  [TRIAGE AGENT] Returning to Supervisor for next routing decision.")
    print_agent_transition("MedicalSymptomTriageAndExtractionAgent", "ClinicalWorkflowSupervisorOrchestrator")

    return {
        "medical_triage_classification_result": response_text,
        "accumulated_conversation_messages": [
            AIMessage(
                content=f"[MedicalSymptomTriageAndExtractionAgent completed triage in {elapsed_seconds:.2f}s]"
            )
        ],
        "current_workflow_execution_stage_number": current_stage_number + 1,
    }


def cardiovascular_differential_diagnosis_agent_node(
    state: ClinicalWorkflowGraphState,
) -> dict:
    """
    LangGraph node for the CardiovascularDifferentialDiagnosisAgent.

    Receives the structured triage data from MedicalSymptomTriageAndExtractionAgent
    and performs deep cardiovascular clinical analysis:
        - Systematic differential diagnosis (most to least likely, with reasoning)
        - Immediate workup recommendations (ECG, troponins, imaging, etc.)
        - Time-sensitive interventions
        - Red flag features that warrant immediate escalation
        - Suggested disposition (admit to CCU / ED workup / outpatient)

    RESILIENCE NOTE:
        This agent has a 30s timeout (vs 15s for Supervisor, 20s for Triage)
        because differential diagnosis reasoning is more computationally intensive
        and requires more generation time from the model.

        It also has a smaller bulkhead (max_concurrent=3) because deep analysis
        should not be parallelised aggressively — quality over throughput.
    """
    current_stage_number = state.get("current_workflow_execution_stage_number", 3)

    print_major_workflow_stage_banner(
        stage_number=current_stage_number,
        stage_title="CardiovascularDifferentialDiagnosisAgent — Deep Analysis",
        stage_description=(
            "The Cardiovascular Specialist Agent receives the structured triage data\n"
            "  and performs comprehensive differential diagnosis reasoning.\n"
            "  Note: 30s timeout (vs 15s for Supervisor) — longer reasoning time needed.\n"
            "  Note: Bulkhead max_concurrent=3 (smaller pool) — quality over throughput.\n"
            "  Token budget Layer 1 check will reflect both Supervisor + Triage usage."
        ),
    )

    patient_query_text = state["accumulated_conversation_messages"][0].content
    structured_triage_data = state.get("medical_triage_classification_result") or "No triage data available"

    cardiovascular_analysis_prompt = f"""
You are the CardiovascularDifferentialDiagnosisAgent. Perform a comprehensive cardiovascular clinical analysis.

ORIGINAL PATIENT PRESENTATION:
{patient_query_text}

STRUCTURED TRIAGE DATA (from MedicalSymptomTriageAndExtractionAgent):
{structured_triage_data}

Provide your complete clinical analysis in the following structured format:

DIFFERENTIAL DIAGNOSIS (Most to Least Likely):
  For each diagnosis provide:
  1. Diagnosis Name
     - Clinical probability: High / Moderate / Low
     - Supporting features from this presentation
     - Features that argue against this diagnosis
     - Why this ranks at this position in the differential

IMMEDIATE WORKUP REQUIRED:
  Priority 1 (Within 10 minutes):
    [Test name] — Rationale
  Priority 2 (Within 30 minutes):
    [Test name] — Rationale
  Priority 3 (Within 60 minutes):
    [Test name] — Rationale

TIME-SENSITIVE INTERVENTIONS:
  [List any immediate therapeutic interventions warranted while awaiting test results]

RED FLAG FEATURES IN THIS PRESENTATION:
  [List specific features that indicate high acuity or time-sensitive diagnosis]

DISPOSITION RECOMMENDATION:
  [CCU / Monitored ED bed / Standard ED / Outpatient]
  [Rationale for disposition]

CLINICAL SUMMARY FOR HANDOFF:
  [2–3 sentence concise summary suitable for verbal handoff to the receiving physician]
""".strip()

    cardiovascular_system_instruction = (
        "You are CardiovascularDifferentialDiagnosisAgent. "
        "You are a consultant cardiologist with emergency medicine expertise. "
        "Provide thorough, evidence-based cardiovascular differential diagnosis. "
        "Be systematic, comprehensive, and clinically precise. "
        "This analysis will inform urgent clinical decision-making."
    )

    response_text, error_message, elapsed_seconds = invoke_llm_through_complete_resilience_stack(
        system_role_instruction=cardiovascular_system_instruction,
        human_turn_prompt_content=cardiovascular_analysis_prompt,
        resilient_caller_for_this_agent=cardiovascular_diagnosis_resilient_caller,
        agent_display_name_for_logging="CardiovascularDifferentialDiagnosisAgent",
        call_purpose_description_for_logging="Deep cardiovascular differential diagnosis from structured triage data",
        workflow_stage_number=current_stage_number,
        substage_number_to_start_at=1,
        per_call_timeout_override_seconds=30.0,
    )

    if error_message:
        print(f"\n  [CARDIOVASCULAR AGENT] Resilience error: {error_message}")
        print(f"  [CARDIOVASCULAR AGENT] Storing error in state. Supervisor will route to FINISH.")
        return {
            "workflow_resilience_error_state": f"Cardiovascular Agent Failed: {error_message}",
            "current_workflow_execution_stage_number": current_stage_number + 1,
        }

    print(f"\n  [CARDIOVASCULAR AGENT] Analysis completed in {elapsed_seconds:.2f}s.")
    print(f"  [CARDIOVASCULAR AGENT] Specialist analysis stored in state.")
    print(f"  [CARDIOVASCULAR AGENT] Returning to Supervisor for final routing decision.")
    print_agent_transition("CardiovascularDifferentialDiagnosisAgent", "ClinicalWorkflowSupervisorOrchestrator")

    return {
        "cardiovascular_differential_diagnosis_result": response_text,
        "accumulated_conversation_messages": [
            AIMessage(
                content=f"[CardiovascularDifferentialDiagnosisAgent completed analysis in {elapsed_seconds:.2f}s]"
            )
        ],
        "current_workflow_execution_stage_number": current_stage_number + 1,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — ROUTING FUNCTION AND WORKFLOW GRAPH ASSEMBLY
#
# The routing function reads `supervisor_next_agent_routing_decision` from state
# and returns the string key that LangGraph uses to select the next node.
#
# WHY CONDITIONAL EDGES FROM SUPERVISOR:
#   The supervisor node sets `supervisor_next_agent_routing_decision` to one of
#   three values. The routing function maps those values to node names.
#   This is the Supervisor pattern: one central orchestrator, workers report back.
#
# GRAPH TOPOLOGY:
#   Supervisor ──► MedicalSymptomTriageAndExtractionAgent ──► Supervisor
#   Supervisor ──► CardiovascularDifferentialDiagnosisAgent ──► Supervisor
#   Supervisor ──► END
# ═══════════════════════════════════════════════════════════════════════════════

def supervisor_routing_function(state: ClinicalWorkflowGraphState) -> str:
    """
    LangGraph routing function called after the Supervisor node.

    Reads the supervisor's routing decision from state and returns the
    corresponding node name or END.

    Returns one of:
        "MedicalSymptomTriageAndExtractionAgent"
        "CardiovascularDifferentialDiagnosisAgent"
        END  (the LangGraph sentinel value for workflow termination)
    """
    routing_decision = state.get("supervisor_next_agent_routing_decision", "FINISH")

    if state.get("workflow_resilience_error_state"):
        print_agent_transition("ClinicalWorkflowSupervisorOrchestrator", "FINISH (Error)")
        return END

    if routing_decision == "MedicalSymptomTriageAndExtractionAgent":
        print_agent_transition("ClinicalWorkflowSupervisorOrchestrator", "MedicalSymptomTriageAndExtractionAgent")
        return "MedicalSymptomTriageAndExtractionAgent"
    elif routing_decision == "CardiovascularDifferentialDiagnosisAgent":
        print_agent_transition("ClinicalWorkflowSupervisorOrchestrator", "CardiovascularDifferentialDiagnosisAgent")
        return "CardiovascularDifferentialDiagnosisAgent"
    else:
        print_agent_transition("ClinicalWorkflowSupervisorOrchestrator", "FINISH (Success)")
        return END


def build_and_compile_clinical_workflow_graph() -> StateGraph:
    """
    Assembles and compiles the full LangGraph StateGraph for the
    clinical multi-agent workflow.

    Node registration:
        "ClinicalWorkflowSupervisorOrchestrator"       → supervisor node function
        "MedicalSymptomTriageAndExtractionAgent"        → triage node function
        "CardiovascularDifferentialDiagnosisAgent"      → specialist node function

    Edge structure:
        Entry point: ClinicalWorkflowSupervisorOrchestrator
        Conditional edges from Supervisor → {Triage, Cardiovascular, END}
        Direct edges from specialist agents → Supervisor (always report back)

    Returns:
        A compiled LangGraph workflow ready for streaming execution.
    """
    clinical_workflow_state_graph = StateGraph(ClinicalWorkflowGraphState)

    clinical_workflow_state_graph.add_node(
        "ClinicalWorkflowSupervisorOrchestrator",
        clinical_workflow_supervisor_orchestrator_node,
    )
    clinical_workflow_state_graph.add_node(
        "MedicalSymptomTriageAndExtractionAgent",
        medical_symptom_triage_and_extraction_agent_node,
    )
    clinical_workflow_state_graph.add_node(
        "CardiovascularDifferentialDiagnosisAgent",
        cardiovascular_differential_diagnosis_agent_node,
    )

    clinical_workflow_state_graph.set_entry_point("ClinicalWorkflowSupervisorOrchestrator")

    clinical_workflow_state_graph.add_conditional_edges(
        "ClinicalWorkflowSupervisorOrchestrator",
        supervisor_routing_function,
        {
            "MedicalSymptomTriageAndExtractionAgent": "MedicalSymptomTriageAndExtractionAgent",
            "CardiovascularDifferentialDiagnosisAgent": "CardiovascularDifferentialDiagnosisAgent",
            END: END,
        },
    )

    # All specialist agents always return to the Supervisor
    clinical_workflow_state_graph.add_edge(
        "MedicalSymptomTriageAndExtractionAgent",
        "ClinicalWorkflowSupervisorOrchestrator",
    )
    clinical_workflow_state_graph.add_edge(
        "CardiovascularDifferentialDiagnosisAgent",
        "ClinicalWorkflowSupervisorOrchestrator",
    )

    return clinical_workflow_state_graph.compile()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Main entry point. Initialises infrastructure, compiles the workflow,
    streams execution, and prints the final resilience metrics report.

    EXPECTED EXECUTION SEQUENCE FOR THIS PATIENT CASE:
        Stage 1:  Initialization
        Stage 2:  Supervisor (1st call) → routes to Triage
        Stage 3:  MedicalSymptomTriageAndExtractionAgent
        Stage 4:  Supervisor (2nd call) → routes to Cardiovascular Specialist
        Stage 5:  CardiovascularDifferentialDiagnosisAgent
        Stage 6:  Supervisor (3rd call) → routes to FINISH
        Stage 7:  Final resilience and token usage report
    """
    workflow_start_wall_time = time.perf_counter()

    # ── Stage 1: Initialization ────────────────────────────────────────────
    print_major_workflow_stage_banner(
        stage_number=1,
        stage_title="SYSTEM INITIALIZATION",
        stage_description=(
            "Setting up all shared resilience infrastructure and compiling the\n"
            "  LangGraph workflow. Objects created here persist for the workflow's\n"
            "  lifetime. The circuit breaker and token manager are particularly\n"
            "  important — they are SHARED across all three agents."
        ),
    )

    print_substage_header(
        stage_number=1, substage_number=1,
        substage_title="Shared Circuit Breaker — registered in CircuitBreakerRegistry",
        substage_explanation=(
            "Key: 'clinical_llm_api'  |  fail_max=5  |  reset_timeout=60s\n"
            "All three agents (Supervisor, Triage, Cardiovascular) share this breaker.\n"
            "One shared failure count: if ANY agent causes 5 failures, ALL agents fail fast."
        ),
    )

    print_substage_header(
        stage_number=1, substage_number=2,
        substage_title="Per-Workflow Token Budget Manager",
        substage_explanation=(
            f"Budget: {per_workflow_token_budget_manager._max_per_workflow:,} tokens total for this workflow run\n"
            "Shared across ALL agents. Cumulative counter tracks Supervisor + Triage + Specialist.\n"
            "Layer 1 of every call reads this counter to enforce the ceiling."
        ),
    )

    print_substage_header(
        stage_number=1, substage_number=3,
        substage_title="Per-Agent ResilientCallers (3 total)",
        substage_explanation=(
            "Supervisor:       timeout=15s  bulkhead_max=10  (routing calls, many, fast)\n"
            "Triage Agent:     timeout=20s  bulkhead_max=5   (extraction, moderate volume)\n"
            "Cardiovascular:   timeout=30s  bulkhead_max=3   (deep analysis, fewer, longer)\n"
            "All share: shared_llm_api_circuit_breaker + per_workflow_token_budget_manager"
        ),
    )

    print_substage_header(
        stage_number=1, substage_number=4,
        substage_title="LangGraph Workflow Graph — Compiling",
        substage_explanation=(
            "Graph topology: Supervisor → {Triage, Cardiovascular, END}\n"
            "Specialist agents always edge back to Supervisor (report-back pattern).\n"
            "Entry point: ClinicalWorkflowSupervisorOrchestrator"
        ),
    )

    compiled_clinical_workflow_graph = build_and_compile_clinical_workflow_graph()
    print(f"  [1.4] Workflow graph compiled successfully.\n")

    # ── Define the patient case ────────────────────────────────────────────
    patient_clinical_presentation_query = (
        "A 55-year-old male presents to the emergency department with sudden onset "
        "crushing chest pain radiating to his left arm and jaw, rated 9/10 in severity. "
        "He reports associated diaphoresis, nausea, and progressive shortness of breath "
        "that started approximately 30 minutes ago at rest. He has a 15-year history of "
        "hypertension managed with lisinopril, is a former smoker (20 pack-year history, "
        "quit 5 years ago), and his father had a myocardial infarction at age 60. "
        "He denies fever, cough, trauma, or recent travel. BP on arrival: 150/95 mmHg, "
        "HR: 102 bpm, RR: 22/min, SpO2: 94% on room air."
    )

    initial_workflow_state: ClinicalWorkflowGraphState = {
        "accumulated_conversation_messages": [HumanMessage(content=patient_clinical_presentation_query)],
        "supervisor_next_agent_routing_decision": "",
        "medical_triage_classification_result": None,
        "cardiovascular_differential_diagnosis_result": None,
        "workflow_resilience_error_state": None,
        "current_workflow_execution_stage_number": 2,
    }

    print(f"\n  Patient Case:")
    for case_line in patient_clinical_presentation_query.split(". "):
        print(f"  | {case_line.strip()}.")
    print()

    print_agent_transition("Patient Query (Input)", "ClinicalWorkflowSupervisorOrchestrator")

    # ── Execute the workflow via streaming ─────────────────────────────────
    # LangGraph's stream() yields a dict for each node execution.
    # We stream rather than invoke() so we can observe each node as it completes.
    final_workflow_state = {}
    for state_update_from_node in compiled_clinical_workflow_graph.stream(initial_workflow_state):
        # state_update_from_node is {node_name: {field: value, ...}}
        for node_name_key, node_output_dict in state_update_from_node.items():
            final_workflow_state.update(node_output_dict)

    # ── Final Stage: Resilience Metrics Report ─────────────────────────────
    total_workflow_elapsed_seconds = time.perf_counter() - workflow_start_wall_time

    print_major_workflow_stage_banner(
        stage_number=final_workflow_state.get("current_workflow_execution_stage_number", 7),
        stage_title="FINAL RESILIENCE AND OBSERVABILITY REPORT",
        stage_description=(
            "Workflow completed. The following report shows the final health state\n"
            "  of every resilience component and the complete token usage breakdown\n"
            "  across all three agents for this workflow execution."
        ),
    )

    print_substage_header(
        stage_number=final_workflow_state.get("current_workflow_execution_stage_number", 7),
        substage_number=1,
        substage_title="Workflow Completion Summary",
        substage_explanation=(
            f"Total wall-clock time: {total_workflow_elapsed_seconds:.2f}s\n"
            f"Error state: {final_workflow_state.get('workflow_resilience_error_state') or 'None — workflow completed successfully'}\n"
            f"Triage collected: {'YES' if final_workflow_state.get('medical_triage_classification_result') else 'NO'}\n"
            f"Cardiovascular analysis collected: {'YES' if final_workflow_state.get('cardiovascular_differential_diagnosis_result') else 'NO'}"
        ),
    )

    print_final_resilience_and_token_usage_report(
        per_workflow_token_budget_manager=per_workflow_token_budget_manager,
    )

    print(f"\n  Total workflow wall-clock time: {total_workflow_elapsed_seconds:.2f}s\n")


if __name__ == "__main__":
    main()