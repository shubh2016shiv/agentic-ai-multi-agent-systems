"""
LLM Provider Factory
======================
Central registry that maps a provider name (``"gemini"``, ``"openai"``,
``"lmstudio"``) to the concrete ``BaseLLMProvider`` implementation.

Adding a new provider:
    1. Create ``core/llm_providers/<name>.py`` with a class that extends
       ``BaseLLMProvider``.
    2. Import it below and add a branch in ``get_llm_provider()``.
    3. Add the name to the ``Literal`` type in ``core/config.py → Settings``.

No other files need to change.
"""

from core.llm_providers.base import BaseLLMProvider

_SUPPORTED_PROVIDERS = ("gemini", "openai", "lmstudio")


def get_llm_provider(provider_name: str) -> BaseLLMProvider:
    """
    Factory that returns the correct provider implementation.

    Imports are deferred so that only the selected provider's SDK is
    loaded — this is essential for Gemini, whose import hangs on Windows
    when ADC is absent.

    Args:
        provider_name: One of ``"gemini"``, ``"openai"``, ``"lmstudio"``.

    Returns:
        A ``BaseLLMProvider`` instance.

    Raises:
        ValueError: If the provider name is not recognized.
    """
    if provider_name == "gemini":
        from core.llm_providers.gemini import GeminiProvider
        return GeminiProvider()

    if provider_name == "openai":
        from core.llm_providers.openai import OpenAIProvider
        return OpenAIProvider()

    if provider_name == "lmstudio":
        from core.llm_providers.lmstudio import LMStudioProvider
        return LMStudioProvider()

    raise ValueError(
        f"Unknown LLM provider: '{provider_name}'. "
        f"Supported providers: {', '.join(_SUPPORTED_PROVIDERS)}"
    )


__all__ = ["BaseLLMProvider", "get_llm_provider"]
