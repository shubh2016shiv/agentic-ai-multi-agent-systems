"""
Clinical Knowledge Base — Static Reference Data for Medical Tools
===================================================================
This module is the DATA LAYER for the tools package. It contains only
static dictionaries and constants — no tool functions, no LLM calls,
no business logic.

Why a separate data layer?
--------------------------
In enterprise systems, data and logic are separated for three reasons:

    1. Testability:  Data dictionaries can be validated independently
       (e.g., "does every drug in the interaction DB also have an entry
       in the drug information DB?").

    2. Maintainability:  When a new drug is added to the formulary, a
       clinician edits THIS file — they never need to touch the tool code.

    3. Swapability:  In production, these dictionaries would be replaced
       by database queries, API calls, or FHIR endpoints. Having them in
       one file makes it clear where the swap happens.

Naming Convention:
    All top-level constants use SCREAMING_SNAKE_CASE to signal that they
    are module-level constants, not class attributes or local variables.
    Each constant has a docstring explaining its schema.

Module name prefixed with underscore (_):
    The leading underscore signals that this module is INTERNAL to the
    tools package. External code should never import from here directly;
    it should import the tool functions from tools/__init__.py instead.
    This is a Python packaging convention for "private" modules.

Used by:
    tools/triage_tools.py        → SYMPTOM_CONDITION_MAP
    tools/pharmacology_tools.py  → DRUG_INTERACTION_DATABASE,
                                   DRUG_INFORMATION_DATABASE,
                                   RENAL_DOSING_ADJUSTMENTS
    tools/guidelines_tools.py    → CLINICAL_GUIDELINE_DATABASE
"""


# ============================================================
# 1. SYMPTOM → CONDITION MAPPING
# ============================================================
# Schema:  { symptom_name: [list of possible conditions] }
# Used by: triage_tools.analyze_symptoms()
#
# Each symptom maps to a ranked list of differential diagnoses.
# In production this would be a medical ontology (SNOMED-CT)
# or a vector similarity search against a clinical corpus.
# ============================================================

SYMPTOM_CONDITION_MAP: dict[str, list[str]] = {
    "chronic cough": ["COPD", "Asthma", "Lung Cancer", "Heart Failure"],
    "dyspnea": ["COPD", "Heart Failure", "Pneumonia", "Pulmonary Embolism"],
    "chest pain": ["Angina", "MI", "Pulmonary Embolism", "Costochondritis"],
    "edema": ["Heart Failure", "CKD", "Nephrotic Syndrome", "Liver Disease"],
    "fatigue": ["Anemia", "Hypothyroidism", "Depression", "CKD", "Heart Failure"],
    "wheezing": ["Asthma", "COPD", "Bronchitis", "Allergic Reaction"],
    "polyuria": ["Diabetes Mellitus", "Diabetes Insipidus", "CKD", "Hypercalcemia"],
    "headache": ["Migraine", "Tension Headache", "Hypertension", "Brain Tumor"],
    "weight loss": ["Cancer", "Hyperthyroidism", "Diabetes", "Depression", "COPD"],
    "joint pain": ["Rheumatoid Arthritis", "Osteoarthritis", "Gout", "Lupus"],
    "hypertension": ["Essential Hypertension", "CKD", "Pheochromocytoma"],
    "hypotension": ["Dehydration", "Sepsis", "Heart Failure", "Adrenal Insufficiency"],
    "dizziness": ["Orthostatic Hypotension", "Vertigo", "Anemia", "Dehydration"],
    "bilateral ankle edema": ["Heart Failure", "CKD", "Venous Insufficiency", "Medication Side Effect"],
}


# ============================================================
# 2. DRUG INTERACTION DATABASE
# ============================================================
# Schema:  { (drug_a, drug_b): { severity, effect, recommendation } }
# Used by: pharmacology_tools.check_drug_interactions()
#
# Keys are LOWERCASE tuples. The lookup code checks both orderings
# (A,B) and (B,A) so entries need not be duplicated.
#
# Severity levels follow the FDA classification:
#   CRITICAL — life-threatening, avoid combination
#   HIGH     — serious harm likely, requires close monitoring
#   MODERATE — manageable with dose adjustment or monitoring
#   LOW      — minor clinical significance
# ============================================================

DRUG_INTERACTION_DATABASE: dict[tuple[str, str], dict[str, str]] = {
    ("lisinopril", "spironolactone"): {
        "severity": "HIGH",
        "effect": "Risk of hyperkalemia — both drugs increase potassium levels",
        "recommendation": "Monitor serum potassium closely. Consider alternative.",
    },
    ("warfarin", "aspirin"): {
        "severity": "HIGH",
        "effect": "Increased bleeding risk due to combined anticoagulant/antiplatelet effects",
        "recommendation": "Avoid combination unless specifically indicated. Monitor INR.",
    },
    ("metformin", "contrast dye"): {
        "severity": "HIGH",
        "effect": "Risk of lactic acidosis in patients with renal impairment",
        "recommendation": "Hold metformin 48h before and after contrast. Check eGFR.",
    },
    ("lisinopril", "nsaids"): {
        "severity": "MODERATE",
        "effect": "NSAIDs can reduce antihypertensive effect and worsen renal function",
        "recommendation": "Use the lowest NSAID dose for shortest duration. Monitor BP and renal function.",
    },
    ("ssri", "tramadol"): {
        "severity": "HIGH",
        "effect": "Risk of serotonin syndrome",
        "recommendation": "Avoid combination. Use alternative analgesic.",
    },
    ("tiotropium", "ipratropium"): {
        "severity": "MODERATE",
        "effect": "Both are anticholinergics — additive side effects without added benefit",
        "recommendation": "Avoid concurrent use. Choose one anticholinergic bronchodilator.",
    },
}


# ============================================================
# 3. DRUG INFORMATION DATABASE
# ============================================================
# Schema:  { drug_name: { class, indications, contraindications,
#                          side_effects, monitoring } }
# Used by: pharmacology_tools.lookup_drug_info()
#
# Each entry provides a formulary-level summary of the drug.
# In production this would be backed by a service like First
# Databank, Lexicomp, or a FHIR MedicationKnowledge resource.
# ============================================================

DRUG_INFORMATION_DATABASE: dict[str, dict] = {
    "lisinopril": {
        "class": "ACE Inhibitor",
        "indications": ["Hypertension", "Heart Failure", "Post-MI", "Diabetic Nephropathy"],
        "contraindications": ["Pregnancy", "Angioedema history", "Bilateral renal artery stenosis"],
        "side_effects": ["Cough", "Hyperkalemia", "Dizziness", "Renal impairment"],
        "monitoring": ["Serum potassium", "Renal function (creatinine, eGFR)", "Blood pressure"],
    },
    "metformin": {
        "class": "Biguanide",
        "indications": ["Type 2 Diabetes Mellitus"],
        "contraindications": ["eGFR <30", "Metabolic acidosis", "Severe hepatic impairment"],
        "side_effects": ["GI upset", "Lactic acidosis (rare)", "B12 deficiency"],
        "monitoring": ["eGFR (at least annually)", "Vitamin B12 levels", "HbA1c"],
    },
    "tiotropium": {
        "class": "Long-Acting Muscarinic Antagonist (LAMA)",
        "indications": ["COPD maintenance", "Asthma (add-on)"],
        "contraindications": ["Hypersensitivity to atropine derivatives"],
        "side_effects": ["Dry mouth", "Urinary retention", "Constipation"],
        "monitoring": ["Lung function (FEV1)", "Symptom control", "Exacerbation frequency"],
    },
    "warfarin": {
        "class": "Vitamin K Antagonist",
        "indications": ["Atrial fibrillation", "DVT/PE", "Mechanical heart valves"],
        "contraindications": ["Active bleeding", "Pregnancy", "Severe hepatic disease"],
        "side_effects": ["Bleeding", "Skin necrosis (rare)", "Purple toe syndrome"],
        "monitoring": ["INR (target 2-3 for most indications)", "Signs of bleeding"],
    },
    "amlodipine": {
        "class": "Calcium Channel Blocker (DHP)",
        "indications": ["Hypertension", "Angina"],
        "contraindications": ["Severe aortic stenosis", "Cardiogenic shock"],
        "side_effects": ["Peripheral edema", "Headache", "Flushing", "Dizziness"],
        "monitoring": ["Blood pressure", "Heart rate", "Peripheral edema"],
    },
    "spironolactone": {
        "class": "Potassium-Sparing Diuretic / Mineralocorticoid Receptor Antagonist",
        "indications": ["Heart Failure", "Hypertension", "Primary Aldosteronism", "Edema"],
        "contraindications": ["Hyperkalemia (K+ >5.5)", "Severe renal impairment", "Addison's disease"],
        "side_effects": ["Hyperkalemia", "Gynecomastia", "Dizziness", "GI upset"],
        "monitoring": ["Serum potassium", "Renal function", "Blood pressure"],
    },
}


# ============================================================
# 4. RENAL DOSING ADJUSTMENT RULES
# ============================================================
# Schema:  { drug_name: { severity_tier: { range, dose, max } } }
# Used by: pharmacology_tools.calculate_dosage_adjustment()
#
# Tiers are "normal", "moderate", "severe" based on eGFR ranges.
# In production these would come from a renal dosing API.
# ============================================================

RENAL_DOSING_ADJUSTMENTS: dict[str, dict[str, dict[str, str]]] = {
    "metformin": {
        "normal":   {"range": "eGFR ≥45",  "dose": "No adjustment needed", "max": "2000mg/day"},
        "moderate": {"range": "eGFR 30-44", "dose": "Reduce to 50%",       "max": "1000mg/day"},
        "severe":   {"range": "eGFR <30",   "dose": "CONTRAINDICATED",     "max": "0"},
    },
    "lisinopril": {
        "normal":   {"range": "eGFR ≥30",  "dose": "No adjustment needed", "max": "40mg/day"},
        "moderate": {"range": "eGFR 10-29", "dose": "Start at 50% dose",   "max": "20mg/day"},
        "severe":   {"range": "eGFR <10",   "dose": "Start at 25% dose",   "max": "10mg/day"},
    },
    "gabapentin": {
        "normal":   {"range": "eGFR ≥60",  "dose": "No adjustment needed", "max": "3600mg/day"},
        "moderate": {"range": "eGFR 30-59", "dose": "Reduce dose by 50%",  "max": "1400mg/day"},
        "severe":   {"range": "eGFR 15-29", "dose": "Reduce dose by 75%",  "max": "700mg/day"},
    },
}


# ============================================================
# 5. CLINICAL GUIDELINE DATABASE
# ============================================================
# Schema:  { condition: { source, topic: { ...recommendations } } }
# Used by: guidelines_tools.lookup_clinical_guideline()
#
# Each entry references a real clinical guideline (GOLD, KDIGO,
# AHA/ACC, APA) and provides key recommendations with evidence grades.
# ============================================================

CLINICAL_GUIDELINE_DATABASE: dict[str, dict] = {
    "copd": {
        "source": "GOLD 2024",
        "treatment": {
            "first_line": "Long-acting bronchodilators (LAMA or LABA)",
            "second_line": "LAMA + LABA combination for persistent symptoms",
            "escalation": "Add ICS if frequent exacerbations and eosinophils ≥300",
            "evidence_grade": "Grade A",
        },
        "diagnosis": {
            "criteria": "Post-bronchodilator FEV1/FVC < 0.7",
            "classification": "GOLD 1-4 based on FEV1 percent predicted",
            "evidence_grade": "Grade A",
        },
    },
    "ckd": {
        "source": "KDIGO 2024",
        "treatment": {
            "first_line": "ACE inhibitor or ARB for proteinuric CKD",
            "blood_pressure_target": "<130/80 mmHg",
            "new_therapy": "SGLT2 inhibitor (dapagliflozin/empagliflozin) for eGFR ≥20",
            "evidence_grade": "Grade A",
        },
        "monitoring": {
            "frequency": "eGFR and UACR every 3-6 months based on stage",
            "potassium": "Monitor potassium with ACEi/ARB use",
            "evidence_grade": "Grade B",
        },
    },
    "hypertension": {
        "source": "AHA/ACC 2024",
        "treatment": {
            "threshold": "≥130/80 mmHg with ASCVD risk or ≥140/90 otherwise",
            "first_line": "Thiazide diuretic, CCB, ACEi, or ARB",
            "combination": "Two-drug therapy for Stage 2 (≥140/90)",
            "evidence_grade": "Grade A",
        },
    },
    "depression": {
        "source": "APA Guidelines 2024",
        "treatment": {
            "first_line": "SSRI or SNRI for moderate-severe depression",
            "augmentation": "Add bupropion or atypical antipsychotic if partial response",
            "psychotherapy": "CBT recommended as monotherapy for mild or adjunct for moderate-severe",
            "evidence_grade": "Grade A",
        },
    },
}
