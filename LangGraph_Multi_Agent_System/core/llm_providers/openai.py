"""
OpenAI LLM Provider
=====================
Wraps ``langchain-openai`` ChatOpenAI for chat completion and
OpenAIEmbeddings for vector embeddings.

Both models use the same OPENAI_API_KEY — no additional credentials needed.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from core.llm_providers.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI GPT models (gpt-4o, gpt-4o-mini, etc.)."""

    @property
    def provider_name(self) -> str:
        return "openai"

    def validate_config(self) -> bool:
        from core.config import settings

        if not settings.openai_api_key:
            raise ValueError(
                "OpenAI API key is not configured. "
                "Set OPENAI_API_KEY in your .env file."
            )
        return True

    def get_llm(
        self,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> BaseChatModel:
        from core.config import settings
        from langchain_openai import ChatOpenAI

        self.validate_config()

        return ChatOpenAI(
            model=settings.openai_model_name,
            api_key=settings.openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def get_embeddings(self, **kwargs: Any) -> Embeddings:
        """
        Return OpenAI text-embedding-3-small via langchain-openai.

        WHY text-embedding-3-small?
            - 1536 dimensions, strong retrieval performance.
            - Cheapest OpenAI embedding model ($0.02 / 1M tokens).
            - Same API key as the chat model — no extra configuration.

        The model name can be overridden via OPENAI_EMBEDDING_MODEL in .env.
        """
        from core.config import settings
        from langchain_openai import OpenAIEmbeddings

        self.validate_config()

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
            **kwargs,
        )
