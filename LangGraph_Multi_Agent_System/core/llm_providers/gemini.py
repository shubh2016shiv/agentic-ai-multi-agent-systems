"""
Google Gemini LLM Provider
============================
Wraps ``langchain-google-genai`` ChatGoogleGenerativeAI for chat and
GoogleGenerativeAIEmbeddings for vector embeddings.

Both models use the same GEMINI_API_KEY — no additional credentials needed.

IMPORTANT — Lazy Import
    ``langchain-google-genai`` 4.x attempts ADC (Application Default
    Credentials) discovery at *import* time, which hangs on Windows when
    ADC is not configured. All imports are therefore deferred to inside
    the method bodies so the rest of the system boots instantly regardless
    of whether Gemini is the active provider.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from core.llm_providers.base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """Provider for Google Gemini models via the Gemini Developer API."""

    @property
    def provider_name(self) -> str:
        return "gemini"

    def validate_config(self) -> bool:
        from core.config import settings

        if not settings.gemini_api_key:
            raise ValueError(
                "Gemini API key is not configured. "
                "Set GEMINI_API_KEY in your .env file."
            )
        return True

    def get_llm(
        self,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> BaseChatModel:
        from core.config import settings

        self.validate_config()

        # Lazy import — avoids the ADC-hang-on-import issue in v4.x
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model_name,
            api_key=settings.gemini_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def get_embeddings(self, **kwargs: Any) -> Embeddings:
        """
        Return GoogleGenerativeAIEmbeddings via langchain-google-genai.

        WHY models/text-embedding-004?
            - Google's latest embedding model (768 dims).
            - Same GEMINI_API_KEY as the chat model — no extra configuration.
            - Free tier available (1500 req/min, more than enough for RAG).

        The model name can be overridden via GEMINI_EMBEDDING_MODEL in .env.

        NOTE: uses lazy import to avoid the same ADC-hang issue as get_llm().
        """
        from core.config import settings

        self.validate_config()

        # Lazy import — same ADC-hang prevention as get_llm()
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.gemini_api_key,
            **kwargs,
        )
