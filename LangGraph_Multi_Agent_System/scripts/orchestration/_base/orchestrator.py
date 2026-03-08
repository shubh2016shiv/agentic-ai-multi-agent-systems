"""
============================================================
_base/orchestrator.py — Backwards-compatibility redirect
============================================================
This module re-exports everything from the root orchestration/orchestrator.py.

The BaseOrchestrator and SPECIALIST_SYSTEM_PROMPTS have moved to:
    orchestration/orchestrator.py  (root component module)

All pattern scripts should import from:
    from orchestration.orchestrator import BaseOrchestrator, SPECIALIST_SYSTEM_PROMPTS

This file keeps existing imports working during the transition.
============================================================
"""

# Re-export everything from the root component module
# CONNECTION: orchestration/orchestrator.py is the authoritative source.
from orchestration.orchestrator import (
    BaseOrchestrator,
    SPECIALIST_SYSTEM_PROMPTS,
    _ORCHESTRATION_LLM_BREAKER,
    _ORCHESTRATION_CALLER,
    _TOKEN_COUNTER,
)

__all__ = [
    "BaseOrchestrator",
    "SPECIALIST_SYSTEM_PROMPTS",
    "_ORCHESTRATION_LLM_BREAKER",
    "_ORCHESTRATION_CALLER",
    "_TOKEN_COUNTER",
]
