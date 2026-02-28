"""
Fallback Chain — Provider Failover
=====================================
Tries LLM providers in priority order. If the primary provider fails
(any exception), it transparently falls through to the next provider.

════════════════════════════════════════
  THE PROBLEM: SINGLE-PROVIDER BRITTLENESS
════════════════════════════════════════
When your entire MAS is wired to one LLM provider, that provider's
outage IS your outage. OpenAI, Anthropic, and Google all have SLA
incidents. Even a 99.9% uptime means ~8.7 hours of downtime per year.

A fallback chain decouples your system from any single provider's
availability:

    Primary: GPT-4o          (fastest, highest quality)
    Fallback-1: Claude Sonnet (slightly different strengths)
    Fallback-2: Gemini Pro    (redundant geographic provider)
    Fallback-3: cached response or static rule-based answer

════════════════════════════════════════
  DESIGN PATTERN: CHAIN OF RESPONSIBILITY
════════════════════════════════════════
This is a textbook Chain of Responsibility implementation:
- Each provider is a "handler" that either handles the request or
  passes it to the next handler.
- No handler knows about the others — each simply declares what to
  do and whether it succeeded.
- New providers are added to the chain without changing any existing code
  (Open/Closed Principle).

════════════════════════════════════════
  WHEN TO USE FALLBACKS VS CIRCUIT BREAKERS
════════════════════════════════════════
These two patterns are COMPLEMENTARY, not alternatives:

  Circuit Breaker → stops calling a KNOWN-DOWN service quickly
  Fallback Chain  → routes to an ALTERNATIVE when any call fails

Best practice: wrap EACH provider in the chain with its own circuit
breaker. The breaker fast-fails a downed provider so the chain can
skip to the next one in <1ms instead of waiting 30s for a timeout.

════════════════════════════════════════
  EXCEPTIONS TO CATCH vs PASS THROUGH
════════════════════════════════════════
The fallback chain catches "provider failure" exceptions (5xx, timeouts,
rate limits, circuit-open) and falls through to the next provider.

It should NOT catch "input error" exceptions (invalid prompt, content
policy violation) because those will fail on every provider — falling
through wastes time and quota.

Configure `non_fallback_exceptions` with your provider's input-error
exception types to prevent this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .exceptions import AllFallbacksFailed, CircuitBreakerOpen

logger = logging.getLogger(__name__)


@dataclass
class Provider:
    """
    A single LLM provider entry in the fallback chain.

    Attributes:
        name:        Display name for logging (e.g., "openai-gpt4o").
        invoke:      The callable to invoke this provider
                     (e.g., `openai_llm.invoke`).
        weight:      Priority weight — lower number = tried first.
                     Providers with equal weight are tried in definition order.

    Example:
        Provider(name="openai", invoke=openai_llm.invoke, weight=1)
        Provider(name="anthropic", invoke=anthropic_llm.invoke, weight=2)
    """

    name: str
    invoke: Callable
    weight: int = 1


class FallbackChain:
    """
    Tries a prioritized list of LLM providers, returning the first success.

    PATTERN: Chain of Responsibility
    - Providers are sorted by weight (ascending).
    - Each is tried in order until one succeeds.
    - If all fail, raises AllFallbacksFailed with the full error log.

    Args:
        providers:               List of Provider instances (any order — sorted by weight).
        non_fallback_exceptions: Exception types that should propagate immediately
                                 without trying the next provider. Default: none.

    Example:
        chain = FallbackChain([
            Provider("gpt4o",   openai_llm.invoke,    weight=1),
            Provider("claude",  anthropic_llm.invoke,  weight=2),
            Provider("gemini",  gemini_llm.invoke,     weight=3),
        ])

        # In your LangGraph node:
        response = chain.call(prompt)

    Example — with non-fallback exceptions (don't waste time on input errors):
        from openai import BadRequestError

        chain = FallbackChain(
            providers=[...],
            non_fallback_exceptions=(BadRequestError,),
        )
    """

    def __init__(
        self,
        providers: list[Provider],
        non_fallback_exceptions: tuple[type[Exception], ...] = (),
    ) -> None:
        if not providers:
            raise ValueError("FallbackChain requires at least one provider.")

        # Sort by weight so callers don't need to pre-order the list
        self._providers = sorted(providers, key=lambda p: p.weight)
        self._non_fallback = non_fallback_exceptions

        logger.info(
            "FallbackChain initialized",
            extra={"providers": [p.name for p in self._providers]},
        )

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """
        Try each provider in priority order, returning the first success.

        Args:
            *args:    Positional args forwarded to each provider's `invoke`.
            **kwargs: Keyword args forwarded to each provider's `invoke`.

        Returns:
            The response from the first provider that succeeds.

        Raises:
            AllFallbacksFailed: if every provider fails.
            Any exception in `non_fallback_exceptions` immediately.
        """
        errors: list[str] = []
        providers_tried: list[str] = []

        for provider in self._providers:
            providers_tried.append(provider.name)
            try:
                logger.debug(
                    "Attempting provider",
                    extra={"provider": provider.name},
                )
                result = provider.invoke(*args, **kwargs)
                logger.info(
                    "Provider succeeded",
                    extra={"provider": provider.name},
                )
                return result

            except self._non_fallback as exc:
                # Input-level error — won't succeed on any other provider either.
                # Re-raise immediately.
                logger.error(
                    "Non-fallback exception — aborting chain",
                    extra={"provider": provider.name, "error": str(exc)},
                )
                raise

            except CircuitBreakerOpen as exc:
                # The circuit breaker already knows this service is down.
                # Skip quickly — no need to wait for a real call to fail.
                logger.warning(
                    "Provider circuit is open — skipping",
                    extra={"provider": provider.name},
                )
                errors.append(f"{provider.name}: [circuit open] {exc.message}")

            except Exception as exc:  # noqa: BLE001 — intentional broad catch in fallback
                logger.warning(
                    "Provider failed — trying next",
                    extra={"provider": provider.name, "error": str(exc)},
                )
                errors.append(f"{provider.name}: {exc}")

        raise AllFallbacksFailed(
            f"All {len(providers_tried)} providers failed: "
            + "; ".join(errors),
            details={
                "providers_tried": providers_tried,
                "last_error": errors[-1] if errors else "unknown",
                "errors": errors,
            },
        )

    @property
    def provider_names(self) -> list[str]:
        """Ordered list of provider names (by priority weight)."""
        return [p.name for p in self._providers]
