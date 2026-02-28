"""
Triage Domain Tools
====================
Tool functions scoped to the TRIAGE AGENT domain.

What is a "Triage Agent"?
    In a hospital, the triage nurse is the FIRST clinician you see.
    Their job is to evaluate your symptoms, assign an urgency level,
    and route you to the right specialist. The triage agent does the
    same thing with patient data: symptom analysis + risk scoring.

Why are these tools in their own module?
    Context scoping (see Script 03, Section 1): each agent gets ONLY
    the tools for its domain. Putting triage tools in a separate file
    makes this separation physically enforced, not just conceptual.

    In this module you will find:
        analyze_symptoms     — maps symptoms to differential diagnoses
        assess_patient_risk  — computes a multi-factor risk score

    Neither of these functions knows anything about drug interactions,
    dosing adjustments, or clinical guidelines. That information belongs
    to the pharmacology and guidelines tools respectively.

Data source:
    All static reference data is imported from _clinical_knowledge_base.py.
    The tool functions contain ONLY logic (matching, scoring, formatting).
"""

import json
import logging

from langchain_core.tools import tool

from observability.decorators import observe_tool
from tools._clinical_knowledge_base import SYMPTOM_CONDITION_MAP

logger = logging.getLogger(__name__)


# ============================================================
# Tool 1: Symptom Analysis (Differential Diagnosis)
# ============================================================

@tool
@observe_tool(tool_name="analyze_symptoms")
def analyze_symptoms(symptoms: list[str], patient_age: int = 0, patient_sex: str = "") -> str:
    """
    Analyze a list of symptoms and return possible differential diagnoses.

    This tool maps patient symptoms to potential conditions using a medical
    knowledge base. Results are ranked by the number of overlapping symptoms
    to suggest the most likely diagnoses.

    Args:
        symptoms: List of reported symptoms (e.g., ["chronic cough", "dyspnea"]).
        patient_age: Patient age for age-specific considerations.
        patient_sex: Patient sex for sex-specific conditions.

    Returns:
        JSON string with ranked differential diagnoses and next steps.
    """
    if not symptoms:
        return json.dumps({"error": "No symptoms provided", "diagnoses": []})

    # ── Match symptoms to conditions ──────────────────────────────────────
    # For each symptom, find all conditions it maps to and count how many
    # symptoms each condition matches. Higher count = more likely diagnosis.
    condition_scores: dict[str, int] = {}
    matched_symptoms: dict[str, list[str]] = {}

    for symptom in symptoms:
        symptom_lower = symptom.lower().strip()
        conditions = SYMPTOM_CONDITION_MAP.get(symptom_lower, [])

        for condition in conditions:
            condition_scores[condition] = condition_scores.get(condition, 0) + 1
            if condition not in matched_symptoms:
                matched_symptoms[condition] = []
            matched_symptoms[condition].append(symptom_lower)

    # ── Rank by matching symptom count (descending) ───────────────────────
    ranked = sorted(condition_scores.items(), key=lambda x: x[1], reverse=True)

    diagnoses = []
    for condition, score in ranked[:5]:  # Top 5 differentials
        diagnoses.append({
            "condition": condition,
            "matching_symptoms": matched_symptoms[condition],
            "confidence": f"{(score / len(symptoms)) * 100:.0f}%",
            "matching_count": f"{score}/{len(symptoms)} symptoms",
        })

    result = {
        "input_symptoms": symptoms,
        "differential_diagnoses": diagnoses,
        "recommendation": (
            "Further workup recommended. Consider labs and imaging "
            "based on the most likely diagnoses."
        ),
    }

    logger.info(f"Symptom analysis: {len(symptoms)} symptoms → {len(diagnoses)} differentials")
    return json.dumps(result, indent=2)


# ============================================================
# Tool 2: Multi-Factor Patient Risk Assessment
# ============================================================

@tool
@observe_tool(tool_name="assess_patient_risk")
def assess_patient_risk(
    age: int,
    conditions: list[str],
    medications: list[str],
    vitals: dict[str, str],
) -> str:
    """
    Perform a multi-factor risk assessment for a patient.

    Evaluates cardiovascular risk, fall risk, polypharmacy risk, and
    generates an overall risk profile with actionable recommendations.

    Args:
        age: Patient age in years.
        conditions: List of active medical conditions.
        medications: List of current medications.
        vitals: Dict of vital signs (e.g., {"BP": "145/90", "HR": "88"}).

    Returns:
        JSON string with risk scores and recommendations.
    """
    risk_factors = []
    overall_score = 0

    # ── Age-based risk ────────────────────────────────────────────────────
    if age >= 75:
        risk_factors.append({
            "factor": "Advanced Age",
            "level": "HIGH",
            "detail": f"Age {age} — increased risk for all adverse outcomes",
        })
        overall_score += 3
    elif age >= 65:
        risk_factors.append({
            "factor": "Geriatric",
            "level": "MODERATE",
            "detail": f"Age {age} — monitor for age-related complications",
        })
        overall_score += 2

    # ── Polypharmacy risk ─────────────────────────────────────────────────
    if len(medications) >= 5:
        risk_factors.append({
            "factor": "Polypharmacy",
            "level": "HIGH",
            "detail": f"{len(medications)} medications — high risk of interactions and ADEs",
        })
        overall_score += 3
    elif len(medications) >= 3:
        risk_factors.append({
            "factor": "Multiple Medications",
            "level": "MODERATE",
            "detail": f"{len(medications)} medications — review for necessity",
        })
        overall_score += 1

    # ── Comorbidity burden ────────────────────────────────────────────────
    if len(conditions) >= 3:
        risk_factors.append({
            "factor": "Multimorbidity",
            "level": "HIGH",
            "detail": f"{len(conditions)} conditions — complex care needs",
        })
        overall_score += 2

    # ── Vital sign-specific risks ─────────────────────────────────────────
    if "BP" in vitals:
        bp_reading = vitals["BP"]
        try:
            systolic = int(bp_reading.split("/")[0])
            if systolic >= 180:
                risk_factors.append({
                    "factor": "Hypertensive Crisis",
                    "level": "CRITICAL",
                    "detail": f"BP {bp_reading} — immediate intervention needed",
                })
                overall_score += 5
            elif systolic >= 140:
                risk_factors.append({
                    "factor": "Hypertension",
                    "level": "MODERATE",
                    "detail": f"BP {bp_reading} — above target",
                })
                overall_score += 1
            elif systolic < 90:
                risk_factors.append({
                    "factor": "Hypotension",
                    "level": "HIGH",
                    "detail": f"BP {bp_reading} — risk of syncope and falls",
                })
                overall_score += 3
        except (ValueError, IndexError):
            pass  # Unparseable BP format — skip silently

    # ── Overall risk classification ───────────────────────────────────────
    if overall_score >= 8:
        risk_level = "CRITICAL"
    elif overall_score >= 5:
        risk_level = "HIGH"
    elif overall_score >= 3:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    result = {
        "overall_risk_level": risk_level,
        "risk_score": overall_score,
        "risk_factors": risk_factors,
        "recommendations": [
            "Comprehensive medication review by clinical pharmacist"
            if len(medications) >= 5
            else "Continue current medication plan",
            "Consider geriatric assessment"
            if age >= 75
            else "Age-appropriate screening",
            "Cardiology consult recommended"
            if overall_score >= 5
            else "Routine follow-up",
        ],
    }

    logger.info(f"Risk assessment: score={overall_score}, level={risk_level}")
    return json.dumps(result, indent=2)
