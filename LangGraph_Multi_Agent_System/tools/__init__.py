"""
Medical Tools Package — Public API
====================================
This is the public interface for the tools package. All tool functions
are re-exported here so that external code can import from a single,
stable location:

    from tools import analyze_symptoms, check_drug_interactions

Internal structure (not imported directly by external code):

    tools/
    ├── __init__.py                  # THIS FILE — public API
    ├── _clinical_knowledge_base.py  # Static data: symptom maps, drug DBs, guidelines
    ├── triage_tools.py              # Triage domain: analyze_symptoms, assess_patient_risk
    ├── pharmacology_tools.py        # Pharmacology: check_drug_interactions, lookup_drug_info,
    │                                #               calculate_dosage_adjustment
    └── guidelines_tools.py          # Guidelines: lookup_clinical_guideline

Design principle — Context Scoping (see Script 03):
    Each domain module contains ONLY the tools for that agent's role.
    This enforces the context scoping principle at the file-system level:
    a triage agent imports from triage_tools, a pharmacology agent from
    pharmacology_tools. The __init__.py re-exports everything for
    convenience, but the underlying separation is what matters.
"""

# ── Triage domain tools ─────────────────────────────────────────────────
# Used by: Triage Agent (analyze symptoms, score patient risk)
from tools.triage_tools import (
    analyze_symptoms,
    assess_patient_risk,
)

# ── Pharmacology domain tools ───────────────────────────────────────────
# Used by: Pharmacology Agent (drug interactions, drug info, renal dosing)
from tools.pharmacology_tools import (
    check_drug_interactions,
    lookup_drug_info,
    calculate_dosage_adjustment,
)

# ── Guidelines domain tools ─────────────────────────────────────────────
# Used by: Guidelines Agent (clinical guideline lookup)
from tools.guidelines_tools import (
    lookup_clinical_guideline,
)

__all__ = [
    # Triage domain
    "analyze_symptoms",
    "assess_patient_risk",
    # Pharmacology domain
    "check_drug_interactions",
    "lookup_drug_info",
    "calculate_dosage_adjustment",
    # Guidelines domain
    "lookup_clinical_guideline",
]
