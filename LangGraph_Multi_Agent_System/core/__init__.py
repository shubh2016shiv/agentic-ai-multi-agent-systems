"""
Core Infrastructure Package
============================
Provides centralized configuration, shared data models, and custom exceptions
used across all modules in the LangGraph Multi-Agent Medical System.

Modules:
    - config: Pydantic Settings, environment loading, LLM factory
    - models: Shared Pydantic models (PatientCase, DrugInfo, etc.)
    - exceptions: Custom exception hierarchy for graceful error handling
"""

from core.config import settings, get_llm
from core.models import PatientCase, DrugInfo, GuidelineReference, HandoffContext
from core.exceptions import (
    MASBaseException,
    GuardrailTripped,
    TokenBudgetExceeded,
    CircuitBreakerOpen,
    HandoffLimitReached,
)

__all__ = [
    "settings",
    "get_llm",
    "PatientCase",
    "DrugInfo",
    "GuidelineReference",
    "HandoffContext",
    "MASBaseException",
    "GuardrailTripped",
    "TokenBudgetExceeded",
    "CircuitBreakerOpen",
    "HandoffLimitReached",
]
