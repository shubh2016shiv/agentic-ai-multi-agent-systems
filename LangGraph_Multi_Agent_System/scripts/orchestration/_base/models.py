"""
============================================================
_base/models.py — Backwards-compatibility redirect
============================================================
This module re-exports everything from the root orchestration/models.py.

The shared orchestration models have moved to:
    orchestration/models.py  (root component module)

All pattern scripts should import from:
    from orchestration.models import OrchestrationResult, ...

This file keeps existing imports working during the transition.
============================================================
"""

# Re-export everything from the root component module
# CONNECTION: orchestration/models.py is the authoritative source.
from orchestration.models import (
    OrchestrationResult,
    PatientWorkload,
    SHARED_PATIENT,
    SHARED_WORKLOAD,
    format_patient_for_prompt,
)

__all__ = [
    "OrchestrationResult",
    "PatientWorkload",
    "SHARED_PATIENT",
    "SHARED_WORKLOAD",
    "format_patient_for_prompt",
]
