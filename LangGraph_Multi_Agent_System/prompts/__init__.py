"""
Prompts Package Initialization
==============================
Exporting the specialized agent prompt templates to Centralize asset management.
"""

from prompts.base_prompts import ENTERPRISE_CONSTRAINTS_SUFFIX
from prompts.triage_prompts import TRIAGE_SYSTEM_PROMPT
from prompts.pharmacology_prompts import PHARMACOLOGY_SYSTEM_PROMPT
from prompts.guidelines_prompts import GUIDELINES_SYSTEM_PROMPT

__all__ = [
    "ENTERPRISE_CONSTRAINTS_SUFFIX",
    "TRIAGE_SYSTEM_PROMPT",
    "PHARMACOLOGY_SYSTEM_PROMPT",
    "GUIDELINES_SYSTEM_PROMPT",
]
