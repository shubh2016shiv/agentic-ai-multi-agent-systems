"""
Base LLM Provider
===================
Abstract base class for all LLM provider implementations.

Every provider must implement three methods:
    - get_llm()         -> returns a LangChain ChatModel
    - get_embeddings()  -> returns a LangChain Embeddings model
    - validate_config() -> checks that required settings are present

The Strategy Pattern here lets the rest of the codebase depend on the
abstract interface, not on any concrete LLM SDK. Adding a new provider
(Anthropic, Cohere, Ollama, etc.) is a single-file addition with zero
changes to existing code.
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings


class BaseLLMProvider(ABC):
    """
    Abstract interface that every LLM provider must satisfy.

    Subclasses are responsible for:
        1. Importing the correct SDK (lazily, inside get_llm / get_embeddings).
        2. Mapping generic parameters (temperature, max_tokens) to the
           provider-specific constructor arguments.
        3. Validating that required API keys / URLs are configured.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name used in logs and error messages."""
        ...

    @abstractmethod
    def get_llm(
        self,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> BaseChatModel:
        """
        Return a ready-to-use LangChain ChatModel.

        Args:
            temperature: Sampling temperature (0 = deterministic).
            max_tokens:  Maximum output tokens.
            **kwargs:    Provider-specific overrides.

        Returns:
            A LangChain BaseChatModel instance.
        """
        ...

    @abstractmethod
    def get_embeddings(self, **kwargs: Any) -> Embeddings:
        """
        Return a ready-to-use LangChain Embeddings model from the SAME provider.

        This is the companion to get_llm(). It ensures the entire system
        (chat completion + RAG embeddings) uses a single provider,
        with NO external ONNX / sentence-transformer model downloads.

        WHY THIS MATTERS:
            ChromaDB's default embedding function downloads the
            all-MiniLM-L6-v2 ONNX model (79MB) on first use. By wiring
            an explicit LangChain embedding model here — using the same
            API key that is already configured — we avoid that download
            entirely and keep all network calls to the same provider.

        Args:
            **kwargs: Provider-specific overrides (e.g., model name).

        Returns:
            A LangChain Embeddings instance.
        """
        ...

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Return True if the provider's required config is present.

        Implementations should check API keys, URLs, etc. and raise
        a descriptive ValueError when something is missing.
        """
        ...
