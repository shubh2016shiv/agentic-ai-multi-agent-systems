"""
Agents Package
================
Agent definitions and base class for the clinical decision support system.
"""

from agents.base_agent import BaseAgent
from agents.triage import TriageAgent
from agents.pharmacology import PharmacologyAgent
from agents.guidelines import GuidelinesAgent

__all__ = [
    "BaseAgent",
    "TriageAgent",
    "PharmacologyAgent",
    "GuidelinesAgent",
]
