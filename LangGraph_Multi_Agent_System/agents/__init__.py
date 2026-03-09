"""
Agents Package
================
Agent definitions and base class for the clinical decision support system.

All concrete agents are re-exported here so scripts can use:
    from agents import TriageAgent, DiagnosticAgent, PharmacistAgent
"""

from agents.base_agent import BaseAgent
from agents.triage import TriageAgent
from agents.pharmacology import PharmacologyAgent
from agents.guidelines import GuidelinesAgent
from agents.diagnostic import DiagnosticAgent
from agents.pharmacist import PharmacistAgent

__all__ = [
    "BaseAgent",
    "TriageAgent",
    "PharmacologyAgent",
    "GuidelinesAgent",
    "DiagnosticAgent",
    "PharmacistAgent",
]

