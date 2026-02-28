"""
Token Manager — Budget Tracking & Enforcement
===============================================
Tracks token consumption per agent and per workflow, enforcing hard
budget limits to prevent runaway costs.

════════════════════════════════════════
  *** WHERE TOKEN MANAGER SHOULD BE USED IN MAS ***
════════════════════════════════════════

TOKEN MANAGER RATIONALE:
    Multi-agent workflows amplify costs. In supervisor pattern: triage → pulm
    → cardio → nephro → synthesis = 5 LLM calls. If each uses 500 tokens,
    workflow consumes 2,500 tokens. Without budget enforcement, one runaway
    loop (e.g., reflection, debate) can cost $5–10 before anyone notices.

WHERE IT SHOULD BE USED (but currently IS NOT):
    scripts/orchestration/* — All 4 patterns (supervisor, peer_to_peer,
    dynamic_router, hybrid) invoke 3+ specialists + synthesis = 4–5 LLM calls
    per workflow. Token budget should track cumulative usage and fail fast.

    scripts/MAS_architectures/reflection_self_critique.py — Generator → critic
    → revise loop; max_iterations guard exists but token budget adds cost ceiling.

    scripts/MAS_architectures/parallel_voting.py — 3+ specialists in parallel;
    cumulative tokens from all branches.

    scripts/MAS_architectures/adversarial_debate.py — Pro/con opening, rebuttals,
    judge = 5 LLM calls; token budget caps debate cost.

WHERE IT IS CURRENTLY USED:
    *** NOWHERE in production scripts. ***
    - resilience/langgraph_integration_example.py: reference implementation only.
    - scripts/script_09_observability_resilience.py: demo section_5_token_budget().

INTEGRATION PATTERN (from langgraph_integration_example.py):
    1. Create ONE TokenManager per workflow run (not per agent).
    2. Pass token_manager to ResilientCaller constructor (shared budget).
    3. In each node: check_budget(agent_name, estimated_tokens) BEFORE call.
    4. Record actual usage AFTER call: record_usage(agent_name, tokens_in, tokens_out).
    5. ResilientCaller can auto-check if estimated_tokens param is provided.

════════════════════════════════════════
  WHY TOKEN BUDGETS MATTER IN A MAS
════════════════════════════════════════
Token costs multiply fast in multi-agent systems:

  Single agent call:      ~500 tokens = ~$0.001
  5-agent voting pattern: ~500 × 5    = ~$0.005
  10-step reasoning loop: ~500 × 10   = ~$0.05
  3 reasoning loops × 5 agents: ~500 × 30 = ~$0.15

Without a budget, one runaway reasoning loop can cost $5–10 before
anyone notices. The TokenManager tracks cumulative usage and raises
TokenBudgetExceeded BEFORE making the call that would breach the limit.

════════════════════════════════════════
  HOW TO USE IN A LANGGRAPH NODE
════════════════════════════════════════

    def my_agent_node(state: GraphState) -> GraphState:
        manager = state["token_manager"]  # passed via state

        # 1. Estimate and check BEFORE calling
        estimated = manager.counter.count(state["prompt"])
        manager.check_budget("my_agent", estimated_tokens=estimated)

        # 2. Make the actual call
        response = llm.invoke(state["prompt"])

        # 3. Record AFTER the call completes
        usage = response.usage_metadata          # LangChain standard
        manager.record_usage(
            "my_agent",
            tokens_in=usage["input_tokens"],
            tokens_out=usage["output_tokens"],
        )

        return {**state, "response": response}

════════════════════════════════════════
  SOLID — SINGLE RESPONSIBILITY PRINCIPLE
════════════════════════════════════════
The original code had `TokenManager.count_tokens()` — token counting
was the manager's concern. This violates SRP:

  TokenManager's responsibility: TRACK AND ENFORCE token budgets.
  TokenCounter's responsibility:  COUNT tokens in a string.

These are different things. They change for different reasons:
  - TokenManager changes when budget rules change.
  - TokenCounter changes when a new model/tokenizer is supported.

Separating them means you can swap the counting strategy (tiktoken →
provider-reported counts) without touching budget enforcement logic.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from .config import TokenBudgetConfig
from .exceptions import TokenBudgetExceeded

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Counts tokens in a text string using tiktoken.

    Responsibility (SRP): estimate prompt size ONLY.
    Does not track, enforce, or manage budgets — that is TokenManager's job.

    Args:
        model: Tiktoken model name for the tokenizer.
               Defaults to "gpt-4o" (cl100k_base encoding).
               Use your actual deployment model for accurate estimates.

    Example:
        counter = TokenCounter(model="gpt-4o")
        n       = counter.count("What is the patient's creatinine level?")
        print(n)  # → 10
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        self._model = model
        self._encoding = self._load_encoding(model)

    @staticmethod
    def _load_encoding(model: str):
        """Load tiktoken encoding, falling back gracefully if unavailable."""
        try:
            import tiktoken

            try:
                return tiktoken.encoding_for_model(model)
            except KeyError:
                logger.warning(
                    "Unknown model for tiktoken — using cl100k_base fallback",
                    extra={"model": model},
                )
                return tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning(
                "tiktoken not installed — token counting will use character estimate. "
                "Install it: pip install tiktoken"
            )
            return None

    def count(self, text: str) -> int:
        """
        Return the number of tokens in `text`.

        If tiktoken is unavailable, returns a rough estimate (~4 chars/token).
        This estimate is intentionally conservative to avoid under-counting.

        Args:
            text: The string to tokenize.

        Returns:
            Token count (exact if tiktoken available, estimated otherwise).
        """
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        # Fallback: ~4 characters per token is a commonly used rule of thumb
        return max(1, len(text) // 4)


class TokenManager:
    """
    Tracks token consumption and enforces budgets per agent and per workflow.

    Responsibility (SRP): budget TRACKING and ENFORCEMENT only.
    Token counting is delegated to TokenCounter.

    Args:
        config: TokenBudgetConfig with per-agent and per-workflow limits.

    LangGraph integration:
        Create ONE TokenManager per workflow execution and store it in the
        graph state so all nodes share the same budget counter:

            initial_state = {
                "token_manager": TokenManager(TokenBudgetConfig()),
                ...
            }

    Example:
        manager = TokenManager(TokenBudgetConfig(max_tokens_per_workflow=16_000))

        manager.check_budget("triage", estimated_tokens=300)
        manager.record_usage("triage", tokens_in=200, tokens_out=500)

        print(manager.get_workflow_summary())
    """

    def __init__(self, config: TokenBudgetConfig) -> None:
        self._max_per_workflow = config.max_tokens_per_workflow
        self._max_per_agent = config.max_tokens_per_agent

        # Per-agent counters: {agent_name: {tokens_in, tokens_out, calls}}
        self._agent_usage: dict[str, dict[str, int]] = defaultdict(
            lambda: {"tokens_in": 0, "tokens_out": 0, "calls": 0}
        )
        self._total_in: int = 0
        self._total_out: int = 0

    # ── Budget Enforcement ───────────────────────────────────────────────────

    def check_budget(self, agent_name: str, estimated_tokens: int = 0) -> None:
        """
        Assert that the upcoming call would not breach the budget.

        *** TRIGGER: This is where budget enforcement occurs. Called from:
        - ResilientCaller.call() if estimated_tokens param is provided.
        - Node code directly (see langgraph_integration_example.py).

        CURRENTLY NOT TRIGGERED in scripts/ — token_manager=None in orchestrator. ***

        Call this BEFORE invoking the LLM to prevent wasted calls.
        If it raises, the call should not be made.

        Args:
            agent_name:       The requesting agent (for error context).
            estimated_tokens: Rough token count for the upcoming call.
                              Use TokenCounter.count() to estimate.

        Raises:
            TokenBudgetExceeded: if the budget would be exceeded.
        """
        total_used = self._total_in + self._total_out

        if total_used + estimated_tokens > self._max_per_workflow:
            raise TokenBudgetExceeded(
                f"Workflow token budget would be exceeded by agent '{agent_name}'.",
                details={
                    "scope": "workflow",
                    "agent_name": agent_name,
                    "used": total_used,
                    "estimated_add": estimated_tokens,
                    "limit": self._max_per_workflow,
                },
            )

        logger.debug(
            "Token budget check passed",
            extra={
                "agent": agent_name,
                "used": total_used,
                "limit": self._max_per_workflow,
                "remaining": self.remaining_budget,
            },
        )

    # ── Usage Recording ──────────────────────────────────────────────────────

    def record_usage(
        self,
        agent_name: str,
        tokens_in: int,
        tokens_out: int,
    ) -> dict[str, int]:
        """
        Record actual token usage from a completed LLM call.

        *** SHOULD BE CALLED after every LLM invoke to track cumulative cost.
        CURRENTLY NOT CALLED in scripts/ — token_manager=None in orchestrator. ***

        Args:
            agent_name: Which agent made the call.
            tokens_in:  Input/prompt tokens consumed.
            tokens_out: Output/completion tokens consumed.

        Returns:
            Dict with call_tokens, workflow_total, remaining.
        """
        self._agent_usage[agent_name]["tokens_in"] += tokens_in
        self._agent_usage[agent_name]["tokens_out"] += tokens_out
        self._agent_usage[agent_name]["calls"] += 1

        self._total_in += tokens_in
        self._total_out += tokens_out

        total = self._total_in + self._total_out
        remaining = self.remaining_budget

        logger.debug(
            "Token usage recorded",
            extra={
                "agent": agent_name,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "workflow_total": total,
                "remaining": remaining,
            },
        )

        return {
            "call_tokens": tokens_in + tokens_out,
            "workflow_total": total,
            "remaining": remaining,
        }

    # ── Observability Queries ────────────────────────────────────────────────

    @property
    def remaining_budget(self) -> int:
        """Tokens remaining in the workflow budget."""
        used = self._total_in + self._total_out
        return max(0, self._max_per_workflow - used)

    def get_workflow_summary(self) -> dict:
        """
        Return a full summary of workflow-level token consumption.

        Useful for logging at the end of a workflow or for a cost report.
        """
        total = self._total_in + self._total_out
        return {
            "total_tokens_in": self._total_in,
            "total_tokens_out": self._total_out,
            "total_tokens": total,
            "budget_limit": self._max_per_workflow,
            "remaining": self.remaining_budget,
            "utilization_pct": round(total / self._max_per_workflow * 100, 1)
            if self._max_per_workflow > 0
            else 0.0,
        }

    def get_agent_summary(self, agent_name: str) -> dict:
        """Return token usage for a specific agent."""
        usage = self._agent_usage[agent_name]
        total = usage["tokens_in"] + usage["tokens_out"]
        return {
            "agent": agent_name,
            "tokens_in": usage["tokens_in"],
            "tokens_out": usage["tokens_out"],
            "total_tokens": total,
            "calls": usage["calls"],
        }

    def get_all_agents_summary(self) -> list[dict]:
        """Return per-agent usage sorted by total tokens (descending)."""
        return sorted(
            [self.get_agent_summary(name) for name in self._agent_usage],
            key=lambda x: x["total_tokens"],
            reverse=True,
        )

    def reset(self) -> None:
        """
        Reset all counters to zero.

        Call this between workflow executions if the TokenManager instance
        is reused (not recommended — prefer creating a fresh instance per
        workflow to avoid cross-workflow budget contamination).
        """
        self._agent_usage.clear()
        self._total_in = 0
        self._total_out = 0
        logger.info("Token manager reset — all counters cleared")
