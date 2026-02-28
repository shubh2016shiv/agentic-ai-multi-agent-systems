"""
Pharmacology Domain Tools
==========================
Tool functions scoped to the PHARMACOLOGY AGENT domain.

What is a "Pharmacology Agent"?
    A clinical pharmacologist reviews a patient's medication regimen for:
    - Drug-drug interactions (e.g., two drugs that both raise potassium)
    - Renal/hepatic dosing adjustments (e.g., metformin at low eGFR)
    - Contraindications and side effect profiles
    They provide specific, actionable recommendations — not general advice.

Why are these tools in their own module?
    The Pharmacology Agent in Script 03 receives ONLY these tools via
    llm.bind_tools(pharma_tools). By physically separating them into
    this file, the domain boundary is enforced at the module level:
    a developer cannot accidentally bind a triage tool to the pharmacology
    agent without an explicit cross-module import.

    In this module you will find:
        check_drug_interactions      — pairwise interaction checker
        lookup_drug_info             — drug class, indications, monitoring
        calculate_dosage_adjustment  — renal dose adjustment by eGFR

Data source:
    All static reference data is imported from _clinical_knowledge_base.py.
"""

import json
import logging

from langchain_core.tools import tool

from observability.decorators import observe_tool
from tools._clinical_knowledge_base import (
    DRUG_INTERACTION_DATABASE,
    DRUG_INFORMATION_DATABASE,
    RENAL_DOSING_ADJUSTMENTS,
)

logger = logging.getLogger(__name__)


# ============================================================
# Tool 1: Drug-Drug Interaction Checker
# ============================================================

@tool
@observe_tool(tool_name="check_drug_interactions")
def check_drug_interactions(medications: list[str]) -> str:
    """
    Check for drug-drug interactions among a list of medications.

    Compares all medication pairs against a known interactions database
    and returns any identified interactions with severity and recommendations.

    Args:
        medications: List of medication names the patient is currently taking.

    Returns:
        JSON string with identified interactions, severity levels, and recommendations.
    """
    if len(medications) < 2:
        return json.dumps({
            "interactions_found": 0,
            "message": "Need at least 2 medications to check interactions",
            "interactions": [],
        })

    interactions_found = []
    checked_pairs: set[tuple[str, str]] = set()

    for i, drug_a in enumerate(medications):
        for j, drug_b in enumerate(medications):
            if i >= j:
                continue

            # Normalize names to lowercase for lookup
            a_lower = drug_a.lower().strip()
            b_lower = drug_b.lower().strip()

            # Strip dosage info (e.g., "Lisinopril 20mg daily" → "lisinopril")
            a_base = a_lower.split()[0] if a_lower else a_lower
            b_base = b_lower.split()[0] if b_lower else b_lower

            canonical_pair = tuple(sorted([a_base, b_base]))
            if canonical_pair in checked_pairs:
                continue
            checked_pairs.add(canonical_pair)

            # Check both orderings in the database
            interaction = (
                DRUG_INTERACTION_DATABASE.get((a_base, b_base))
                or DRUG_INTERACTION_DATABASE.get((b_base, a_base))
            )

            if interaction:
                interactions_found.append({
                    "drug_a": drug_a,
                    "drug_b": drug_b,
                    "severity": interaction["severity"],
                    "effect": interaction["effect"],
                    "recommendation": interaction["recommendation"],
                })

    result = {
        "medications_checked": medications,
        "interactions_found": len(interactions_found),
        "interactions": interactions_found,
        "all_clear": len(interactions_found) == 0,
        "message": (
            "No interactions detected"
            if not interactions_found
            else f"⚠️ {len(interactions_found)} interaction(s) found — review recommendations"
        ),
    }

    logger.info(
        f"Drug interaction check: {len(medications)} drugs → "
        f"{len(interactions_found)} interactions"
    )
    return json.dumps(result, indent=2)


# ============================================================
# Tool 2: Drug Information Lookup
# ============================================================

@tool
@observe_tool(tool_name="lookup_drug_info")
def lookup_drug_info(drug_name: str) -> str:
    """
    Look up comprehensive information about a specific drug.

    Returns drug class, indications, contraindications, side effects,
    and monitoring requirements from a clinical knowledge base.

    Args:
        drug_name: Name of the drug to look up (e.g., "lisinopril").

    Returns:
        JSON string with drug information or a not-found message.
    """
    drug_lower = drug_name.lower().strip()

    # Try exact match first, then try matching just the base name
    info = DRUG_INFORMATION_DATABASE.get(drug_lower)
    if not info:
        # Strip dosage (e.g., "Lisinopril 20mg daily" → "lisinopril")
        base_name = drug_lower.split()[0]
        info = DRUG_INFORMATION_DATABASE.get(base_name)

    if info:
        result = {
            "drug_name": drug_name,
            "found": True,
            **info,
        }
    else:
        result = {
            "drug_name": drug_name,
            "found": False,
            "message": (
                f"Drug '{drug_name}' not found in knowledge base. "
                f"Available: {list(DRUG_INFORMATION_DATABASE.keys())}"
            ),
        }

    logger.info(f"Drug lookup: {drug_name} → {'found' if info else 'not found'}")
    return json.dumps(result, indent=2)


# ============================================================
# Tool 3: Renal Dosage Adjustment Calculator
# ============================================================

@tool
@observe_tool(tool_name="calculate_dosage_adjustment")
def calculate_dosage_adjustment(
    drug_name: str,
    current_dose: str,
    egfr: float,
    weight_kg: float = 70.0,
) -> str:
    """
    Calculate renal dose adjustment for a medication based on eGFR.

    Many drugs require dose reduction in patients with impaired kidney
    function. This tool provides evidence-based dose adjustments using
    established CKD staging thresholds.

    Args:
        drug_name: Name of the medication (e.g., "metformin").
        current_dose: Current dosage string (e.g., "1000mg BID").
        egfr: Estimated GFR in mL/min/1.73m².
        weight_kg: Patient weight in kilograms (default 70).

    Returns:
        JSON string with adjusted dosage recommendation and rationale.
    """
    drug_lower = drug_name.lower().strip()

    # Try exact match, then base name (strip dosage info)
    adjustments = RENAL_DOSING_ADJUSTMENTS.get(drug_lower)
    if not adjustments:
        base_name = drug_lower.split()[0]
        adjustments = RENAL_DOSING_ADJUSTMENTS.get(base_name)

    if not adjustments:
        return json.dumps({
            "drug": drug_name,
            "current_dose": current_dose,
            "egfr": egfr,
            "adjustment_needed": "UNKNOWN",
            "message": f"No renal dosing data available for {drug_name}. Consult pharmacy.",
        })

    # ── Determine CKD severity tier based on eGFR ────────────────────────
    if egfr >= 60:
        tier = adjustments.get("normal", adjustments.get("moderate"))
        ckd_stage_label = "Normal/Mild (≥60)"
    elif egfr >= 30:
        tier = adjustments.get("moderate", adjustments["normal"])
        ckd_stage_label = "Moderate (30-59)"
    else:
        tier = adjustments.get("severe", adjustments["moderate"])
        ckd_stage_label = "Severe (<30)"

    result = {
        "drug": drug_name,
        "current_dose": current_dose,
        "egfr": egfr,
        "ckd_stage": ckd_stage_label,
        "adjustment": tier["dose"],
        "max_dose": tier["max"],
        "egfr_range": tier["range"],
        "recommendation": (
            f"For {drug_name} with eGFR {egfr}: {tier['dose']}. "
            f"Maximum dose: {tier['max']}."
        ),
    }

    logger.info(f"Dose adjustment: {drug_name} eGFR={egfr} → {tier['dose']}")
    return json.dumps(result, indent=2)
