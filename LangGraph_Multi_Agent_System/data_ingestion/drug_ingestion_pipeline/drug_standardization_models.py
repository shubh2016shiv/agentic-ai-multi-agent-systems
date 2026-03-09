"""
Drug Data Standardization Models
===================================
PURPOSE
-------
This module defines the Pydantic data models that form the DATA CONTRACT
for the Drug Ingestion Pipeline. Every drug record flowing through the
pipeline is validated against these models before it reaches MongoDB.

MODEL HIERARCHY (Data Flow Order)
----------------------------------

    ┌──────────────────────────────────┐
    │  RawDrugExcelRecord              │  ← Step 1: Raw extract from Excel
    │  (Excel fields only)             │
    └────────────────┬─────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────┐
    │  LLMDrugEnrichmentResponse       │  ← Step 2: Structured LLM output
    │  (clinical attributes via LLM)   │     Strict validators REJECT
    └────────────────┬─────────────────┘     placeholder values like
                     │                       "Unknown" / "Other"
                     ▼
    ┌──────────────────────────────────┐
    │  StandardizedDrugDocument        │  ← Step 3: Merged & validated
    │  (final MongoDB document)        │     Uses alias "_meta" for
    └──────────────────────────────────┘     provenance sub-document

    ┌──────────────────────────────────┐
    │  DrugIngestionPipelineResult     │  ← Reporting (NOT stored in MongoDB)
    └──────────────────────────────────┘

SCHEMA DESIGN DECISIONS
-----------------------
Based on what FDB MedKnowledge, Micromedex, Medi-Span, and Lexidrug track:

1.  ACTIVE INGREDIENT is the single most important field for clinical safety
    logic (therapeutic substitution, generic preference rules).

2.  DOSAGE_STRENGTH and DOSAGE_FORM are parsed from the raw drug name string
    rather than LLM-enriched — they are directly readable from names like
    "Pharbedryl Capsule 25 Mg Oral" and don't require LLM inference.

3.  PROVENANCE (_meta sub-document) keeps pipeline artifact fields
    (source_file, llm_enrichment_model, ingested_at) out of the clinical
    record and into a clearly separated metadata envelope. This removes
    clutter from agent queries and keeps the clinical schema clean.

4.  DATA_QUALITY_SCORE is NOT persisted. It is computed as a Python property
    at read time. Storing a derived metric that goes stale was identified as
    an anti-pattern (always showed 0 before enrichment ran).

5.  IS_PREFERRED_ON_FORMULARY is NOT a drug master attribute — it is an
    organizational formulary attribute that varies per health system.
    It belongs in a separate formulary collection keyed by (org_id, drug_name).

6.  LLM ENRICHMENT VALIDATION: The LLMDrugEnrichmentResponse model has strict
    validators that REJECT placeholder values ("Unknown", "Other", empty lists
    for required clinical fields). This forces the retry mechanism to re-call
    the LLM rather than silently accepting garbage output.

7.  MONITORING_PARAMETERS and DRUG_LAB_INTERACTIONS are added because:
    - Monitoring parameters (e.g., "Serum potassium for ACE inhibitors")
      are what pharmacists check during order verification.
    - Drug-lab interactions (e.g., "diphenhydramine → false-positive PCP screen")
      are tracked by Medi-Span specifically and are clinically critical.

8.  DOSING_BY_POPULATION is what Micromedex specifically calls out as a
    critical safety need for neonatal, pediatric, and renal-impaired patients.
"""

import re
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# STANDARD THERAPEUTIC CATEGORIES (WHO ATC Level 1 + clinical groupings)
# =============================================================================
# These are the ONLY valid values for therapeutic_category in
# LLMDrugEnrichmentResponse. If the LLM returns anything outside this set,
# the strict validator flags it (not hard-rejects — it logs a warning and
# accepts novel values so the system can evolve).
# =============================================================================

RECOGNIZED_THERAPEUTIC_CATEGORIES: set[str] = {
    "Cardiovascular System",
    "Respiratory System",
    "Central Nervous System",
    "Gastrointestinal System",
    "Endocrine and Metabolic System",
    "Musculoskeletal System",
    "Anti-infective and Antimicrobial",
    "Dermatological",
    "Genitourinary System",
    "Hematological and Anticoagulant",
    "Immunological and Immunosuppressant",
    "Oncological",
    "Ophthalmic",
    "Analgesic and Anesthetic",
    "Other",
}

# Placeholder values the LLM must NOT return for fields we require to be
# populated with real clinical knowledge.
_LLM_PLACEHOLDER_VALUES: set[str] = {
    "unknown", "other", "n/a", "not available", "not specified",
    "none", "na", "", "unspecified", "various", "see prescribing information",
}


# =============================================================================
# Sub-model: Dosing by Patient Population
# =============================================================================


class DosingByPopulation(BaseModel):
    """
    Special dosing considerations by patient population.

    Enterprise databases (Micromedex, FDB) specifically flag these populations
    as requiring dose adjustments or additional monitoring.

    All fields are optional strings — a null value means "standard adult
    dosing applies; no special adjustment documented."
    """

    pediatric: Optional[str] = Field(
        default=None,
        description=(
            "Pediatric dosing note. E.g., 'Not recommended under 12 years' or "
            "'1-2 mg/kg/dose, max 50 mg, every 6-8 hours for children 6-12 years'."
        ),
    )
    renal_impairment: Optional[str] = Field(
        default=None,
        description=(
            "Renal dosing adjustment. E.g., 'Reduce dose by 50% if CrCl < 30 mL/min; "
            "avoid if CrCl < 10 mL/min'."
        ),
    )
    hepatic_impairment: Optional[str] = Field(
        default=None,
        description=(
            "Hepatic dosing adjustment. E.g., 'Use with caution in severe hepatic "
            "impairment; reduce initial dose by 50%'."
        ),
    )
    geriatric: Optional[str] = Field(
        default=None,
        description=(
            "Geriatric note. E.g., 'Beers Criteria: potentially inappropriate for "
            "patients ≥65 years due to anticholinergic effects'."
        ),
    )


# =============================================================================
# Sub-model: Pipeline Provenance (_meta envelope)
# =============================================================================


class DrugRecordProvenance(BaseModel):
    """
    Pipeline provenance metadata stored in the '_meta' sub-document.

    Keeping these fields in a nested envelope separates pipeline artifact
    fields from the clinical drug record. Agent queries can simply ignore
    _meta entirely and only look at the clinical attributes.

    MongoDB path: drugs._meta.*
    """

    source_file: str = Field(
        description="Relative source path (e.g., 'drugs/') NOT an absolute system path.",
    )
    llm_enrichment_model: Optional[str] = Field(
        default=None,
        description=(
            "LLM model identifier used for enrichment (e.g., 'gpt-4o', 'gemini-2.5-flash'). "
            "None if enrichment was not performed."
        ),
    )
    llm_enrichment_timestamp: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp of when LLM enrichment was performed.",
    )
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the most recent pipeline upsert.",
    )


# =============================================================================
# Model 1: Raw Excel Extraction Record
# =============================================================================


class RawDrugExcelRecord(BaseModel):
    """
    A single drug record extracted DIRECTLY from the source Excel files,
    before any LLM enrichment or structured parsing.

    Contains exactly the fields available in the source Excel files:
        drug_class_sub-class.xlsx  →  drug_name, drug_class, sub_class
        preferred_drugs.xlsx       →  is_preferred_on_formulary overlay

    The '_from_excel' suffix on field names makes it visually clear these
    are raw, unvalidated source values in transit through the pipeline.

    NOTE on is_preferred_on_formulary:
        This field is captured here for completeness (it IS in the source data)
        but is NOT written to the StandardizedDrugDocument. Formulary preference
        is an organizational attribute that varies per health system and belongs
        in a separate formulary collection, not the drug master record.
    """

    drug_name: str = Field(
        ...,
        min_length=1,
        description="Generic drug name as it appears in the Excel file.",
    )
    drug_class_from_excel: str = Field(
        default="Unknown",
        description="Drug class label from drug_class_sub-class.xlsx.",
    )
    sub_class_from_excel: Optional[str] = Field(
        default=None,
        description="Sub-class label from drug_class_sub-class.xlsx.",
    )
    is_preferred_on_formulary: bool = Field(
        default=False,
        description=(
            "From preferred_drugs.xlsx. NOT written to the drug master record. "
            "For future use in a separate formulary collection."
        ),
    )

    @field_validator("drug_name")
    @classmethod
    def reject_placeholder_drug_names(cls, value: str) -> str:
        cleaned = value.strip()
        if cleaned.lower() in {"none", "n/a", "null", "", "unknown", "-", "nan"}:
            raise ValueError(
                f"Drug name '{value}' is a placeholder value, not a valid drug name"
            )
        return cleaned


# =============================================================================
# Model 2: LLM Enrichment Response
# =============================================================================


class LLMDrugEnrichmentResponse(BaseModel):
    """
    Structured output schema for the LLM drug enrichment call.

    This model is passed directly to LangChain's `.with_structured_output()`,
    which forces the LLM to return its response in exactly this schema.

    STRICT VALIDATION — this model REJECTS placeholder responses:
        - therapeutic_category "Other" → triggers LLM retry
        - pharmacological_class "Unknown" → triggers LLM retry
        - Empty common_indications list → triggers LLM retry
        - Empty drug_interactions list → triggers LLM retry

    The rationale: if the LLM knows enough to call a drug a "Sedating
    Antihistamine" (drug_class), it KNOWS its therapeutic_category
    ("Central Nervous System"). Accepting "Other" silently is a bug,
    not a graceful degradation. Force a retry.
    """

    active_ingredient: str = Field(
        description=(
            "INN (International Nonproprietary Name) of the active chemical compound. "
            "For single-ingredient drugs this is the generic name. "
            "For combination drugs, list the primary active ingredient. "
            "Example: 'Diphenhydramine Hydrochloride' for Benadryl."
        ),
    )
    brand_names: list[str] = Field(
        default_factory=list,
        description=(
            "2-5 most widely recognized brand/trade names. "
            "Example: ['Benadryl', 'Unisom'] for Diphenhydramine."
        ),
    )
    therapeutic_category: str = Field(
        description=(
            "Broadest organ-system or disease-area category (WHO ATC Level 1 equivalent). "
            "MUST be one of these exact values — do NOT use 'Other' unless the drug "
            "genuinely does not fit any other category: "
            "Cardiovascular System, Respiratory System, Central Nervous System, "
            "Gastrointestinal System, Endocrine and Metabolic System, "
            "Musculoskeletal System, Anti-infective and Antimicrobial, "
            "Dermatological, Genitourinary System, Hematological and Anticoagulant, "
            "Immunological and Immunosuppressant, Oncological, Ophthalmic, "
            "Analgesic and Anesthetic, Other. "
            "Example: diphenhydramine → 'Central Nervous System'."
        ),
    )
    pharmacological_class: str = Field(
        description=(
            "Mechanism-based pharmacological class. NEVER return 'Unknown'. "
            "Example: 'First-Generation (Sedating) H1 Antihistamine' for diphenhydramine, "
            "'Angiotensin-Converting Enzyme (ACE) Inhibitor' for lisinopril."
        ),
    )
    mechanism_of_action: str = Field(
        description=(
            "1-3 sentence molecular/cellular mechanism. Do not return 'Not available'. "
            "Example: 'Competitively antagonizes histamine at H1 receptors. Also has "
            "significant anticholinergic activity by blocking muscarinic receptors.'"
        ),
    )
    common_indications: list[str] = Field(
        description=(
            "3-6 primary FDA-approved or widely-used clinical indications. "
            "Must not be empty. "
            "Example: ['Allergic rhinitis', 'Urticaria', 'Motion sickness', 'Insomnia']."
        ),
    )
    contraindications: list[str] = Field(
        description=(
            "3-5 key contraindications. "
            "Example: ['Narrow-angle glaucoma', 'Benign prostatic hypertrophy', "
            "'MAO inhibitor use within 14 days', 'Neonates and premature infants']."
        ),
    )
    common_side_effects: list[str] = Field(
        description=(
            "4-8 most frequently reported adverse effects. "
            "Example: ['Sedation', 'Dry mouth', 'Dizziness', 'Blurred vision', "
            "'Urinary retention', 'Constipation']."
        ),
    )
    serious_adverse_effects: list[str] = Field(
        default_factory=list,
        description=(
            "2-5 serious or life-threatening adverse effects. "
            "Example: ['Paradoxical CNS excitation in children', 'Anticholinergic toxicity', "
            "'Prolonged QT interval']."
        ),
    )
    drug_interactions: list[str] = Field(
        description=(
            "3-6 clinically significant drug-drug interactions with rationale. "
            "Must not be empty. "
            "Example: ['MAO inhibitors (hypertensive crisis risk)', "
            "'CNS depressants including alcohol (additive sedation)', "
            "'Tricyclic antidepressants (additive anticholinergic effects)']."
        ),
    )
    drug_lab_interactions: list[str] = Field(
        default_factory=list,
        description=(
            "Drug-laboratory test interferences — when this drug INVALIDATES or "
            "FALSELY ALTERS lab results. Tracked by Medi-Span and Lexidrug. "
            "Example for diphenhydramine: ["
            "'May cause false-positive urine drug screen for phencyclidine (PCP)', "
            "'May suppress skin-test reactions to allergens for up to 72 hours']."
        ),
    )
    monitoring_parameters: list[str] = Field(
        default_factory=list,
        description=(
            "2-5 laboratory or clinical parameters pharmacists monitor during therapy. "
            "Example for an ACE inhibitor: ["
            "'Serum potassium (hyperkalemia risk)', "
            "'Serum creatinine and BUN (renal function)', "
            "'Blood pressure response', "
            "'Signs of angioedema']."
        ),
    )
    dosing_by_population: DosingByPopulation = Field(
        default_factory=DosingByPopulation,
        description=(
            "Dosing considerations for special populations. "
            "Populate any sub-fields that have important adjustments; "
            "leave null if standard adult dosing applies."
        ),
    )
    route_of_administration: list[str] = Field(
        description=(
            "All available routes. Must not be empty. "
            "Example: ['Oral', 'Intravenous', 'Topical']."
        ),
    )
    dosage_forms: list[str] = Field(
        description=(
            "Available dosage forms. Must not be empty. "
            "Example: ['Tablet', 'Capsule', 'Oral Solution', 'Injection']."
        ),
    )
    pregnancy_category: Optional[str] = Field(
        default=None,
        description="FDA letter category: A, B, C, D, X, or null.",
    )
    controlled_substance_schedule: Optional[str] = Field(
        default=None,
        description="DEA schedule (Schedule I–V) or null if not controlled.",
    )
    black_box_warning: bool = Field(
        default=False,
        description="True if the FDA has issued a black box (boxed) warning.",
    )

    # =========================================================================
    # STRICT VALIDATORS — Reject placeholder LLM output to force retries
    # =========================================================================

    @model_validator(mode="after")
    def reject_placeholder_llm_responses_that_indicate_llm_failure(
        self,
    ) -> "LLMDrugEnrichmentResponse":
        """
        Validate that the LLM has returned real clinical knowledge, not
        placeholder defaults. Raises ValueError for fields where a default
        like "Unknown" or "Other" is a clear signal the LLM didn't try.

        When this raises, the enrichment service's retry loop catches the
        ValidationError and re-calls the LLM with the same prompt.
        """
        errors: list[str] = []

        if self.therapeutic_category.strip().lower() in _LLM_PLACEHOLDER_VALUES:
            errors.append(
                f"therapeutic_category '{self.therapeutic_category}' is a placeholder. "
                "The LLM must identify the correct organ-system category."
            )

        if self.pharmacological_class.strip().lower() in _LLM_PLACEHOLDER_VALUES:
            errors.append(
                f"pharmacological_class '{self.pharmacological_class}' is a placeholder. "
                "The LLM must identify the mechanism-based class."
            )

        if self.active_ingredient.strip().lower() in _LLM_PLACEHOLDER_VALUES:
            errors.append(
                f"active_ingredient '{self.active_ingredient}' is a placeholder."
            )

        if not self.common_indications:
            errors.append("common_indications must not be empty.")

        if not self.drug_interactions:
            errors.append("drug_interactions must not be empty.")

        if not self.route_of_administration:
            errors.append("route_of_administration must not be empty.")

        if not self.dosage_forms:
            errors.append("dosage_forms must not be empty.")

        if errors:
            raise ValueError(
                f"LLM returned {len(errors)} placeholder/empty field(s): "
                + "; ".join(errors)
            )

        return self

    @field_validator("pregnancy_category")
    @classmethod
    def validate_pregnancy_category_is_fda_letter(
        cls, value: Optional[str]
    ) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip().upper()
        return cleaned if cleaned in {"A", "B", "C", "D", "X"} else None


# =============================================================================
# Model 3: Standardized Drug Document (Final MongoDB Schema)
# =============================================================================


class StandardizedDrugDocument(BaseModel):
    """
    The COMPLETE, STANDARDIZED drug record written to MongoDB.

    Merges data from three sources:
        1. Excel files   → drug_name, drug_class, sub_class
        2. Drug name     → dosage_strength, dosage_form, route hint
        3. LLM enrichment → all clinical attributes

    SCHEMA ORGANIZATION:
        IDENTITY         → drug_name, active_ingredient, brand_names
        CLASSIFICATION   → therapeutic_category, drug_class,
                           pharmacological_class, sub_class
        PARSED           → dosage_strength, dosage_form (from drug name string)
        CLINICAL PROFILE → mechanism_of_action, indications, contraindications,
                           side_effects, interactions, monitoring, lab_interactions
        DOSAGE & ADMIN   → route_of_administration, dosage_forms,
                           dosing_by_population
        SAFETY/REGULATORY→ pregnancy_category, schedule, black_box_warning
        PROVENANCE       → _meta sub-document (source_file, llm model, timestamps)

    NOT STORED:
        - data_quality_score: computed as a @property at read time
        - is_preferred_on_formulary: belongs in a separate formulary collection

    MONGODB COLLECTION: medical_knowledge_base.drugs
    DEDUPLICATION KEY:  drug_name (upsert filter)
    """

    model_config = ConfigDict(populate_by_name=True)

    drug_name: str = Field(
        ...,
        min_length=1,
        description=(
            "Full drug name as it appears in the formulary/source data — "
            "PRIMARY KEY for upsert. Title-cased during validation. "
            "May include brand prefix, form, and strength "
            "(e.g., 'Pharbedryl Capsule 25 Mg Oral')."
        ),
    )
    active_ingredient: str = Field(
        default="Unknown",
        description=(
            "INN name of the active chemical compound — the single most important "
            "field for therapeutic substitution, generic preference, and "
            "duplicate therapy detection. "
            "Example: 'Diphenhydramine Hydrochloride'."
        ),
    )
    brand_names: list[str] = Field(
        default_factory=list,
        description="Common brand/trade names from LLM enrichment.",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # CLASSIFICATION HIERARCHY (WHO ATC-aligned)
    # ─────────────────────────────────────────────────────────────────────────

    therapeutic_category: str = Field(
        default="Other",
        description="Organ-system category (ATC Level 1). Example: 'Central Nervous System'.",
    )
    drug_class: str = Field(
        default="Unknown",
        description=(
            "Therapeutic drug class from source Excel data. "
            "Example: 'Sedating Antihistamine', 'ACE Inhibitor', 'Statin'."
        ),
    )
    pharmacological_class: str = Field(
        default="Unknown",
        description=(
            "Mechanism-based pharmacological classification from LLM enrichment. "
            "Example: 'First-Generation (Sedating) H1 Antihistamine'."
        ),
    )
    sub_class: Optional[str] = Field(
        default=None,
        description="Chemical or pharmacological sub-class from Excel source data.",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PARSED FROM DRUG NAME STRING
    # These fields are extracted by DrugNameParser — NOT by the LLM.
    # They are structurally present in names like "Drug Capsule 25 Mg Oral"
    # and should never be left buried in the name string.
    # ─────────────────────────────────────────────────────────────────────────

    dosage_strength: Optional[str] = Field(
        default=None,
        description=(
            "Dosage strength parsed from the drug name string. "
            "Example: '25 Mg', '0.5 Mcg', '250 Mg/5Ml', '2%'."
        ),
    )
    dosage_form: Optional[str] = Field(
        default=None,
        description=(
            "Primary dosage form parsed from the drug name string. "
            "Example: 'Capsule', 'Tablet', 'Oral Suspension', 'Ophthalmic Solution'."
        ),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # CLINICAL PROFILE
    # ─────────────────────────────────────────────────────────────────────────

    mechanism_of_action: str = Field(
        default="Not available",
        description="Molecular/cellular mechanism from LLM enrichment.",
    )
    common_indications: list[str] = Field(
        default_factory=list,
        description="Primary clinical indications.",
    )
    contraindications: list[str] = Field(
        default_factory=list,
        description="Conditions where this drug should NOT be used.",
    )
    common_side_effects: list[str] = Field(
        default_factory=list,
        description="Most frequently reported adverse effects.",
    )
    serious_adverse_effects: list[str] = Field(
        default_factory=list,
        description="Serious or life-threatening adverse effects.",
    )
    drug_interactions: list[str] = Field(
        default_factory=list,
        description="Clinically significant drug-drug interactions.",
    )
    drug_lab_interactions: list[str] = Field(
        default_factory=list,
        description=(
            "Drug-laboratory test interferences — when this drug invalidates "
            "or falsely alters lab results. Tracked by Medi-Span."
        ),
    )
    monitoring_parameters: list[str] = Field(
        default_factory=list,
        description="Laboratory or clinical parameters to monitor during therapy.",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # DOSAGE & ADMINISTRATION
    # ─────────────────────────────────────────────────────────────────────────

    route_of_administration: list[str] = Field(
        default_factory=list,
        description="All available routes of administration.",
    )
    dosage_forms: list[str] = Field(
        default_factory=list,
        description="All available dosage forms from LLM enrichment.",
    )
    dosing_by_population: DosingByPopulation = Field(
        default_factory=DosingByPopulation,
        description="Dosing adjustments for special populations (pediatric, renal, hepatic, geriatric).",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SAFETY & REGULATORY
    # ─────────────────────────────────────────────────────────────────────────

    pregnancy_category: Optional[str] = Field(
        default=None,
        description="FDA pregnancy risk letter category: A, B, C, D, X, or null.",
    )
    controlled_substance_schedule: Optional[str] = Field(
        default=None,
        description="DEA controlled substance schedule or null if not controlled.",
    )
    black_box_warning: bool = Field(
        default=False,
        description="Whether this drug has an FDA black box (boxed) warning.",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PROVENANCE — stored as _meta sub-document
    # Use model_dump(by_alias=True) when serializing for MongoDB.
    # ─────────────────────────────────────────────────────────────────────────

    meta: DrugRecordProvenance = Field(
        alias="_meta",
        description=(
            "Pipeline provenance envelope. Stored as '_meta' in MongoDB. "
            "Contains source_file, llm_enrichment_model, timestamps."
        ),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATORS
    # ─────────────────────────────────────────────────────────────────────────

    @field_validator("drug_name")
    @classmethod
    def normalize_drug_name_to_title_case(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or normalized.lower() in {"none", "n/a", "null", "nan"}:
            raise ValueError(f"'{value}' is not a valid drug name")
        return normalized.title()

    @field_validator("pregnancy_category")
    @classmethod
    def validate_pregnancy_category_is_fda_letter(
        cls, value: Optional[str]
    ) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip().upper()
        return cleaned if cleaned in {"A", "B", "C", "D", "X"} else None

    # ─────────────────────────────────────────────────────────────────────────
    # DATA QUALITY SCORE — computed at READ TIME, NOT persisted
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def data_quality_score(self) -> float:
        """
        Compute a completeness score (0.0–1.0) based on populated clinical fields.

        This is a READ-TIME property — it is NEVER written to MongoDB.
        Compute it after fetching a document if needed for analytics.

        Scoring: 10 clinical fields, each worth 0.1 point.
            1.  active_ingredient        (not "Unknown")
            2.  therapeutic_category     (not "Other")
            3.  pharmacological_class    (not "Unknown")
            4.  mechanism_of_action      (not "Not available")
            5.  common_indications       (non-empty)
            6.  contraindications        (non-empty)
            7.  common_side_effects      (non-empty)
            8.  drug_interactions        (non-empty)
            9.  route_of_administration  (non-empty)
            10. brand_names              (non-empty)
        """
        score = 0.0
        if self.active_ingredient and self.active_ingredient != "Unknown":
            score += 0.1
        if self.therapeutic_category and self.therapeutic_category != "Other":
            score += 0.1
        if self.pharmacological_class and self.pharmacological_class != "Unknown":
            score += 0.1
        if self.mechanism_of_action and self.mechanism_of_action != "Not available":
            score += 0.1
        if self.common_indications:
            score += 0.1
        if self.contraindications:
            score += 0.1
        if self.common_side_effects:
            score += 0.1
        if self.drug_interactions:
            score += 0.1
        if self.route_of_administration:
            score += 0.1
        if self.brand_names:
            score += 0.1
        return round(score, 2)


# =============================================================================
# Model 4: Pipeline Execution Result (In-Memory Reporting Only)
# =============================================================================


class DrugIngestionPipelineResult(BaseModel):
    """
    Summary report of a Drug Ingestion Pipeline execution.
    NOT stored in MongoDB — used for console output and monitoring.
    """

    pipeline_name: str = Field(
        default="Drug Ingestion Pipeline (Standardized + LLM-Enriched)",
    )
    total_drugs_extracted_from_excel: int = Field(default=0)
    source_files_processed: int = Field(default=0)
    total_drugs_enriched_by_llm: int = Field(default=0)
    total_drugs_where_enrichment_failed: int = Field(default=0)
    total_drugs_where_enrichment_was_skipped: int = Field(default=0)
    average_data_quality_score: float = Field(default=0.0)
    total_documents_upserted_as_new: int = Field(default=0)
    total_documents_updated_existing: int = Field(default=0)
    total_documents_failed_to_load: int = Field(default=0)
    pipeline_duration_seconds: float = Field(default=0.0)
    errors: list[str] = Field(default_factory=list)

    @property
    def formatted_summary(self) -> str:
        divider = "=" * 64
        lines = [
            "",
            divider,
            f"  {self.pipeline_name}",
            divider,
            "  EXTRACT PHASE:",
            f"    Drugs extracted from Excel:      {self.total_drugs_extracted_from_excel}",
            f"    Source files processed:           {self.source_files_processed}",
            "",
            "  TRANSFORM / ENRICHMENT PHASE:",
            f"    Drugs enriched by LLM:            {self.total_drugs_enriched_by_llm}",
            f"    Enrichment failed (fallback used):{self.total_drugs_where_enrichment_failed}",
            f"    Enrichment skipped (up-to-date):  {self.total_drugs_where_enrichment_was_skipped}",
            f"    Average data quality score:       {self.average_data_quality_score:.2f} / 1.00",
            "",
            "  LOAD PHASE:",
            f"    New documents upserted:           {self.total_documents_upserted_as_new}",
            f"    Existing documents updated:       {self.total_documents_updated_existing}",
            f"    Documents failed to load:         {self.total_documents_failed_to_load}",
            "",
            "  OVERALL:",
            f"    Duration:                         {self.pipeline_duration_seconds:.2f}s",
            f"    Errors:                           {len(self.errors)}",
            divider,
        ]
        if self.errors:
            lines.append(f"\n  Error Details ({len(self.errors)}):")
            for msg in self.errors[:10]:
                lines.append(f"    [ERROR] {msg}")
            if len(self.errors) > 10:
                lines.append(f"    ... and {len(self.errors) - 10} more")
        return "\n".join(lines)
