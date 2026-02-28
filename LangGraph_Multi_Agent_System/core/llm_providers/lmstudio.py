"""
LM Studio LLM Provider
========================
Connects to a locally-running LM Studio server via its OpenAI-compatible
REST API.

LM Studio exposes an endpoint at ``http://localhost:1234/v1`` that accepts
the same request format as the OpenAI Chat Completions API. This means we
can reuse ``langchain-openai`` ChatOpenAI and OpenAIEmbeddings with a
custom ``base_url`` — no extra dependencies required.

Advantages for development:
    - Zero cloud API costs
    - Works completely offline
    - Models run on local GPU (fast iteration)
    - Supports any GGUF / HuggingFace model loaded in LM Studio

EMBEDDING NOTE:
    LM Studio also exposes an OpenAI-compatible /v1/embeddings endpoint.
    Load a dedicated embedding model in LM Studio (e.g., nomic-embed-text
    or text-embedding-3-small GGUF) and set LMSTUDIO_EMBEDDING_MODEL in
    .env to use it. If not set, the embedding model name defaults to the
    LM Studio placeholder "lm-studio" (which uses whatever is loaded).
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from core.llm_providers.base import BaseLLMProvider


class LMStudioProvider(BaseLLMProvider):
    """Provider for locally-hosted models via LM Studio's OpenAI-compatible API."""

    @property
    def provider_name(self) -> str:
        return "lmstudio"

    def validate_config(self) -> bool:
        from core.config import settings

        if not settings.lmstudio_base_url:
            raise ValueError(
                "LM Studio base URL is not configured. "
                "Set LMSTUDIO_BASE_URL in your .env file "
                "(default: http://localhost:1234/v1)."
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
            base_url=settings.lmstudio_base_url,
            api_key="lm-studio",
            model=settings.lmstudio_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def get_embeddings(self, **kwargs: Any) -> Embeddings:
        """
        Return embeddings via LM Studio's /v1/embeddings endpoint.

        LM Studio exposes the same OpenAI Embeddings API at its base_url,
        so OpenAIEmbeddings with a custom base_url works out of the box.

        To use this:
            1. Load an embedding model in LM Studio
               (e.g., nomic-embed-text, all-MiniLM-L6-v2 GGUF).
            2. Set LMSTUDIO_EMBEDDING_MODEL in .env to the model's
               identifier shown in LM Studio.

        WHY NOT OpenAI cloud embeddings when using LM Studio?
            LM Studio is typically used for fully-offline, zero-cost
            development. Using a cloud embedding model would defeat
            that purpose.
        """
        from core.config import settings
        from langchain_openai import OpenAIEmbeddings

        self.validate_config()

        return OpenAIEmbeddings(
            base_url=settings.lmstudio_base_url,
            api_key="lm-studio",
            model=settings.lmstudio_embedding_model,
            check_embedding_ctx_length=False,  # LM Studio models vary
            **kwargs,
        )
