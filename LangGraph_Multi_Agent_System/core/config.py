"""
Centralized Configuration Module
==================================
This module is the SINGLE SOURCE OF TRUTH for all configuration in the
LangGraph Multi-Agent Medical System. It follows the Twelve-Factor App
methodology: configuration is loaded from environment variables (via .env),
validated with Pydantic, and exposed as a singleton Settings object.

Design Decisions:
    - Pydantic Settings for type-safe, validated configuration
    - Factory pattern (get_llm / get_embeddings) to abstract provider selection
    - All magic numbers are named constants with documentation
    - Environment variables override defaults for production flexibility

Usage:
    from core.config import settings, get_llm, get_embeddings

    llm        = get_llm()          # Get the configured chat model
    embeddings = get_embeddings()   # Get the configured embedding model
    print(settings.llm_provider)   # "gemini", "openai", or "lmstudio"
"""

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

# ---------------------------------------------------------------------------
# Load .env file from the project root
# ---------------------------------------------------------------------------
# We resolve the path relative to THIS file so it works regardless of the
# working directory from which the script is executed.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables.

    Every field has a sensible default so the system can start with minimal
    configuration. In production, override via .env or environment variables.

    Attributes:
        llm_provider: Which LLM provider to use ("openai" or "gemini").
        openai_api_key: OpenAI API key (required if llm_provider is "openai").
        openai_model_name: OpenAI model identifier.
        gemini_api_key: Google Gemini API key (required if llm_provider is "gemini").
        gemini_model_name: Gemini model identifier.
        langfuse_secret_key: Langfuse secret for observability tracing.
        langfuse_public_key: Langfuse public key for observability tracing.
        langfuse_host: Langfuse server URL.
        max_tokens_per_agent_call: Hard cap on tokens per single LLM invocation.
        max_tokens_per_workflow: Hard cap on total tokens per multi-agent workflow.
        rate_limit_calls_per_minute: Max LLM API calls per minute (per process).
        circuit_breaker_fail_max: Consecutive failures before circuit opens.
        circuit_breaker_reset_timeout: Seconds before a tripped breaker retries.
    """

    # --- LLM Provider ---
    llm_provider: Literal["openai", "gemini", "lmstudio"] = Field(
        default="openai",
        description="Which LLM provider to use. Determines which API key and model are active.",
    )

    # --- OpenAI ---
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model_name: str = Field(default="gpt-4o-mini", description="OpenAI chat model name")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model (text-embedding-3-small or text-embedding-3-large)",
    )

    # --- Gemini ---
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    gemini_model_name: str = Field(
        default="gemini-2.5-flash", description="Gemini chat model name"
    )
    gemini_embedding_model: str = Field(
        default="models/text-embedding-004",
        description="Gemini embedding model (models/text-embedding-004)",
    )

    # --- LM Studio (Local) ---
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        description="LM Studio OpenAI-compatible API endpoint",
    )
    lmstudio_model_name: str = Field(
        default="deepseek/deepseek-r1-0528-qwen3-8b",
        description="Chat model identifier loaded in LM Studio",
    )
    lmstudio_embedding_model: str = Field(
        default="nomic-embed-text",
        description=(
            "Embedding model loaded in LM Studio for /v1/embeddings. "
            "Load a dedicated embedding model (nomic-embed-text, "
            "all-MiniLM-L6-v2, etc.) in LM Studio before use."
        ),
    )

    # --- Langfuse Observability ---
    langfuse_secret_key: str = Field(default="", description="Langfuse secret key")
    langfuse_public_key: str = Field(default="", description="Langfuse public key")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", description="Langfuse host URL"
    )

    # --- Token Budget ---
    max_tokens_per_agent_call: int = Field(
        default=4096,
        description="Maximum tokens allowed per single agent LLM call",
    )
    max_tokens_per_workflow: int = Field(
        default=32000,
        description="Maximum total tokens allowed across an entire workflow execution",
    )

    # --- Rate Limiting ---
    rate_limit_calls_per_minute: int = Field(
        default=30,
        description="Maximum LLM API calls per minute to prevent quota exhaustion",
    )

    # --- Circuit Breaker ---
    circuit_breaker_fail_max: int = Field(
        default=5,
        description="Number of consecutive failures before the circuit breaker opens",
    )
    circuit_breaker_reset_timeout: int = Field(
        default=60,
        description="Seconds to wait before attempting to close an open circuit",
    )

    # --- MongoDB ---
    mongodb_uri: str = Field(
        default="mongodb://admin:adminpassword@localhost:27017",
        description="MongoDB connection URI for the medical knowledge base",
    )
    mongodb_database_name: str = Field(
        default="medical_knowledge_base",
        description="Default MongoDB database name for storing medical guidelines and drugs",
    )

    # --- Paths ---
    project_root: str = Field(default=_PROJECT_ROOT, description="Project root directory")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Allow env vars like OPENAI_API_KEY to map to openai_api_key
        env_prefix = ""
        case_sensitive = False
        extra = "ignore"


# ---------------------------------------------------------------------------
# Singleton settings instance
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_settings() -> Settings:
    """Create and cache the settings singleton."""
    return Settings()


settings = _get_settings()


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------
def get_llm(
    provider: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    **kwargs,
):
    """
    Factory function that returns the configured LLM instance.

    This is the ONLY place in the codebase where LLM objects are created.
    All agents, tools, and scripts call this function to get their LLM,
    ensuring consistent configuration and easy provider switching.

    Delegates to the provider abstraction in ``core.llm_providers`` so
    that SDK imports happen lazily — only the active provider's library
    is loaded at runtime.

    Args:
        provider: Override the default provider ("openai", "gemini", or
                  "lmstudio"). If None, uses settings.llm_provider.
        temperature: LLM temperature (0.0 = deterministic, 1.0 = creative).
                     Medical domain defaults to 0.0 for consistency.
        max_tokens: Maximum output tokens. Defaults to settings value.
        **kwargs: Additional provider-specific arguments.

    Returns:
        A LangChain ChatModel instance.

    Raises:
        ValueError: If the provider is not recognized.

    Example:
        >>> from core.config import get_llm
        >>> llm = get_llm()                        # Uses .env default
        >>> llm = get_llm(provider="openai")        # Force OpenAI
        >>> llm = get_llm(provider="lmstudio")      # Force LM Studio
        >>> llm = get_llm(temperature=0.7)          # More creative
    """
    from core.llm_providers import get_llm_provider

    active_provider = provider or settings.llm_provider
    effective_max_tokens = max_tokens or settings.max_tokens_per_agent_call

    llm_provider = get_llm_provider(active_provider)
    return llm_provider.get_llm(
        temperature=temperature,
        max_tokens=effective_max_tokens,
        **kwargs,
    )


def get_llm_model_name(provider: str | None = None) -> str:
    """Return the configured model name for the given provider (for metrics/tracing)."""
    active = provider or settings.llm_provider
    if active == "openai":
        return settings.openai_model_name
    if active == "gemini":
        return settings.gemini_model_name
    return settings.lmstudio_model_name


def get_embeddings(provider: str | None = None, **kwargs):
    """
    Factory function that returns the configured embedding model instance.

    This mirrors get_llm() for embeddings — the ONLY place in the codebase
    where embedding objects are created. All RAG components call this to
    get their embeddings, ensuring no ONNX / sentence-transformer models
    are ever downloaded.

    By default, uses the same provider as get_llm() (settings.llm_provider),
    so chat and embedding models stay on the same provider and API key.

    Args:
        provider: Override the default provider ("openai", "gemini",
                  or "lmstudio"). If None, uses settings.llm_provider.
        **kwargs: Additional provider-specific arguments.

    Returns:
        A LangChain Embeddings instance.

    Raises:
        ValueError: If the provider is not recognized.

    Example:
        >>> from core.config import get_embeddings
        >>> emb = get_embeddings()                 # Uses .env default provider
        >>> emb = get_embeddings(provider="openai") # Force OpenAI
        >>> vectors = emb.embed_documents(["text"]) # Embed a list of strings
    """
    from core.llm_providers import get_llm_provider

    active_provider = provider or settings.llm_provider
    llm_provider = get_llm_provider(active_provider)
    return llm_provider.get_embeddings(**kwargs)
