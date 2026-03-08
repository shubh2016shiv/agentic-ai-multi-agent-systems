"""
_base — Backwards-compatibility redirect
=========================================
All shared orchestration code has moved to the root orchestration/ module.

Import from orchestration/ directly:
    from orchestration import BaseOrchestrator, OrchestrationResult
    from orchestration.models import SHARED_PATIENT
    from orchestration.orchestrator import SPECIALIST_SYSTEM_PROMPTS
"""

# Re-export from root for backwards compatibility
from orchestration.models import OrchestrationResult, PatientWorkload
from orchestration.orchestrator import BaseOrchestrator

__all__ = ["OrchestrationResult", "PatientWorkload", "BaseOrchestrator"]
