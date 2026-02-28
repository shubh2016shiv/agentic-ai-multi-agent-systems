"""
Observability Metrics
========================
Tracks token usage, latency, and cost across the multi-agent system.
Aggregates metrics per agent, per workflow, and per model for cost
optimization and performance analysis.

Usage:
    from observability.metrics import MetricsCollector

    metrics = MetricsCollector()
    metrics.record_llm_call("triage_agent", tokens_in=150, tokens_out=300, model="gemini-2.5-flash")
    metrics.record_tool_call("drug_lookup", latency_ms=45)
    print(metrics.get_workflow_summary())
"""

import time
import logging
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================
# Model pricing (per 1M tokens, USD) — updated for 2026
# ============================================================
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash-preview-05-20": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}


@dataclass
class LLMCallMetric:
    """Individual LLM call measurement."""
    agent_name: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolCallMetric:
    """Individual tool call measurement."""
    tool_name: str
    agent_name: str
    latency_ms: float
    success: bool
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """
    Collects and aggregates observability metrics for a workflow.

    Create one MetricsCollector per workflow execution to track all
    LLM calls, tool invocations, and their associated costs.

    This data can be:
        1. Logged locally for development debugging
        2. Sent to Langfuse as trace metadata for production monitoring
        3. Used by the token_manager (resilience/) for budget enforcement

    Example:
        >>> metrics = MetricsCollector()
        >>> metrics.record_llm_call("triage", tokens_in=200, tokens_out=500, model="gemini-2.5-flash-preview-05-20")
        >>> summary = metrics.get_workflow_summary()
        >>> print(f"Total cost: ${summary['total_cost_usd']:.4f}")
    """

    def __init__(self):
        self.llm_calls: list[LLMCallMetric] = []
        self.tool_calls: list[ToolCallMetric] = []
        self._start_time = time.time()

    def record_llm_call(
        self,
        agent_name: str,
        tokens_in: int,
        tokens_out: int,
        model: str,
        latency_ms: float = 0.0,
    ) -> LLMCallMetric:
        """
        Record a single LLM API call with token usage.

        Args:
            agent_name: Which agent made this call.
            tokens_in: Number of input/prompt tokens.
            tokens_out: Number of output/completion tokens.
            model: Model identifier for cost calculation.
            latency_ms: Response time in milliseconds.

        Returns:
            The recorded metric object.
        """
        cost = self._calculate_cost(model, tokens_in, tokens_out)

        metric = LLMCallMetric(
            agent_name=agent_name,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=cost,
        )
        self.llm_calls.append(metric)

        logger.debug(
            f"LLM call: {agent_name} | {model} | "
            f"in={tokens_in} out={tokens_out} | "
            f"${cost:.6f} | {latency_ms:.0f}ms"
        )
        return metric

    def record_tool_call(
        self,
        tool_name: str,
        agent_name: str = "unknown",
        latency_ms: float = 0.0,
        success: bool = True,
    ) -> ToolCallMetric:
        """Record a tool invocation with latency and success status."""
        metric = ToolCallMetric(
            tool_name=tool_name,
            agent_name=agent_name,
            latency_ms=latency_ms,
            success=success,
        )
        self.tool_calls.append(metric)
        return metric

    def get_agent_summary(self) -> dict[str, dict]:
        """
        Get per-agent aggregated metrics.

        Returns:
            Dict mapping agent names to their aggregated metrics:
            {
                "triage_agent": {
                    "llm_calls": 3,
                    "total_tokens": 1500,
                    "total_cost_usd": 0.002,
                    "avg_latency_ms": 450.0,
                    "tool_calls": 2,
                },
                ...
            }
        """
        agent_metrics: dict[str, dict] = defaultdict(
            lambda: {
                "llm_calls": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_cost_usd": 0.0,
                "total_latency_ms": 0.0,
                "tool_calls": 0,
                "tool_failures": 0,
            }
        )

        for call in self.llm_calls:
            m = agent_metrics[call.agent_name]
            m["llm_calls"] += 1
            m["total_tokens_in"] += call.tokens_in
            m["total_tokens_out"] += call.tokens_out
            m["total_cost_usd"] += call.cost_usd
            m["total_latency_ms"] += call.latency_ms

        for call in self.tool_calls:
            m = agent_metrics[call.agent_name]
            m["tool_calls"] += 1
            if not call.success:
                m["tool_failures"] += 1

        # Calculate averages
        for name, m in agent_metrics.items():
            if m["llm_calls"] > 0:
                m["avg_latency_ms"] = m["total_latency_ms"] / m["llm_calls"]
            else:
                m["avg_latency_ms"] = 0.0

        return dict(agent_metrics)

    def get_workflow_summary(self) -> dict:
        """
        Get aggregated summary for the entire workflow.

        Returns:
            {
                "total_llm_calls": 12,
                "total_tokens_in": 5000,
                "total_tokens_out": 8000,
                "total_cost_usd": 0.015,
                "total_tool_calls": 8,
                "workflow_duration_ms": 12500.0,
                "agents_involved": ["triage", "pharmacology", "safety"],
            }
        """
        return {
            "total_llm_calls": len(self.llm_calls),
            "total_tokens_in": sum(c.tokens_in for c in self.llm_calls),
            "total_tokens_out": sum(c.tokens_out for c in self.llm_calls),
            "total_cost_usd": sum(c.cost_usd for c in self.llm_calls),
            "total_tool_calls": len(self.tool_calls),
            "tool_failure_count": sum(1 for c in self.tool_calls if not c.success),
            "workflow_duration_ms": (time.time() - self._start_time) * 1000,
            "agents_involved": list(set(c.agent_name for c in self.llm_calls)),
        }

    def print_summary(self) -> None:
        """Print a formatted summary to stdout (useful in educational scripts)."""
        summary = self.get_workflow_summary()
        agent_summary = self.get_agent_summary()

        print("\n" + "=" * 60)
        print("📊 WORKFLOW METRICS SUMMARY")
        print("=" * 60)
        print(f"  Total LLM Calls:    {summary['total_llm_calls']}")
        print(f"  Total Tokens (in):  {summary['total_tokens_in']:,}")
        print(f"  Total Tokens (out): {summary['total_tokens_out']:,}")
        print(f"  Total Cost:         ${summary['total_cost_usd']:.4f}")
        print(f"  Total Tool Calls:   {summary['total_tool_calls']}")
        print(f"  Workflow Duration:  {summary['workflow_duration_ms']:.0f}ms")
        print(f"  Agents Involved:    {', '.join(summary['agents_involved'])}")

        if agent_summary:
            print("\n  Per-Agent Breakdown:")
            print("  " + "-" * 56)
            for agent, metrics in agent_summary.items():
                print(f"    {agent}:")
                print(f"      LLM calls: {metrics['llm_calls']} | "
                      f"Tokens: {metrics['total_tokens_in']}→{metrics['total_tokens_out']} | "
                      f"Cost: ${metrics['total_cost_usd']:.4f}")
        print("=" * 60 + "\n")

    @staticmethod
    def _calculate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
        """Calculate USD cost for an LLM call based on model pricing."""
        pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost_in = (tokens_in / 1_000_000) * pricing["input"]
        cost_out = (tokens_out / 1_000_000) * pricing["output"]
        return cost_in + cost_out
