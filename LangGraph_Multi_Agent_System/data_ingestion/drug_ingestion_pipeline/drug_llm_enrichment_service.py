"""
Drug LLM Enrichment Service
================================
Calls the LLM to enrich raw drug records with comprehensive clinical attributes.

HOW ENRICHMENT WORKS
---------------------
1. Receives drug_name + known Excel classification context.
2. Constructs a strongly-worded prompt that explicitly prohibits placeholder
   responses ("Unknown", "Other", empty lists).
3. Uses LangChain `.with_structured_output(LLMDrugEnrichmentResponse)` to force
   the LLM to return parseable, Pydantic-validated structured data.
4. The LLMDrugEnrichmentResponse model has strict validators that REJECT
   placeholder values and raise ValueError — this triggers the retry loop.
5. Retries up to max_retries_per_drug times with exponential backoff.
6. Returns None on permanent failure — the transformer creates a partial document.

ANTI-PLACEHOLDER STRATEGY
--------------------------
The core bug in the previous version was that the LLM could return
therapeutic_category="Other" for well-known drugs like diphenhydramine
(Benadryl) and it would silently pass validation.

The fix has TWO layers:
    Layer 1 — Prompt:   Explicitly tells the LLM NEVER to use placeholder values
                        and gives specific examples of correct values for common drugs.
    Layer 2 — Validator: LLMDrugEnrichmentResponse.model_validator raises ValueError
                         if any required field contains a placeholder. The retry loop
                         catches this ValidationError and re-calls the LLM.

Together these layers ensure "Other" / "Unknown" only reach the final document
if the drug genuinely cannot be classified after all retry attempts.

Usage:
    service = DrugLLMEnrichmentService(max_retries_per_drug=3)
    result = service.enrich_single_drug_record(
        drug_name="Pharbedryl Capsule 25 Mg Oral",
        known_drug_class="Sedating Antihistamine",
    )
    if result:
        print(result.therapeutic_category)   # "Central Nervous System"
        print(result.active_ingredient)       # "Diphenhydramine Hydrochloride"
"""

import logging
import time
from datetime import UTC, datetime

from langchain_core.prompts import ChatPromptTemplate

from core.config import get_llm, settings
from data_ingestion.drug_ingestion_pipeline.drug_standardization_models import (
    LLMDrugEnrichmentResponse,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENRICHMENT PROMPT — Two layers of anti-placeholder enforcement
# =============================================================================

DRUG_ENRICHMENT_SYSTEM_PROMPT = """You are a licensed clinical pharmacist and pharmacology expert \
with access to FDB MedKnowledge, Micromedex, and Medi-Span reference databases. \
You are populating a clinical decision support drug master database used by \
prescribers and pharmacists at order entry.

ABSOLUTE RULES — violations will cause the record to be REJECTED and retried:

1. NEVER return "Unknown", "Other", "N/A", "Not available", or any placeholder \
   for therapeutic_category, pharmacological_class, active_ingredient, \
   common_indications, drug_interactions, route_of_administration, or dosage_forms.

2. For therapeutic_category, you MUST pick the BEST matching category from this \
   exact list. "Other" is only valid for drugs with no identifiable organ system target:
   - Cardiovascular System         (heart, vessels, antihypertensives, statins)
   - Respiratory System            (bronchodilators, corticosteroids, antihistamines for allergic rhinitis)
   - Central Nervous System        (sedatives, antidepressants, antihistamines with CNS activity, antiepileptics)
   - Gastrointestinal System       (PPIs, antiemetics, laxatives, antidiarrheals)
   - Endocrine and Metabolic System (insulin, thyroid drugs, diabetes drugs)
   - Musculoskeletal System        (NSAIDs, muscle relaxants, gout drugs)
   - Anti-infective and Antimicrobial (antibiotics, antivirals, antifungals)
   - Dermatological                (topical steroids, topical antifungals, wound care)
   - Genitourinary System          (BPH drugs, OAB drugs, erectile dysfunction)
   - Hematological and Anticoagulant (anticoagulants, antiplatelets, iron)
   - Immunological and Immunosuppressant (biologics, immunosuppressants, vaccines)
   - Oncological                   (chemotherapy, targeted therapy, supportive care)
   - Ophthalmic                    (eye drops, ophthalmic antibiotics, glaucoma drugs)
   - Analgesic and Anesthetic      (opioids, NSAIDs for pain, local anesthetics)
   - Other                         (ONLY for drugs with no clear organ system target)

   EXAMPLE: diphenhydramine (Benadryl) → "Central Nervous System" (NOT "Other")
   RATIONALE: Despite antihistamine action, its primary clinical effects are CNS-mediated.

3. For pharmacological_class, use the SPECIFIC mechanism-based class:
   - diphenhydramine → "First-Generation (Sedating) H1 Antihistamine"
   - lisinopril      → "Angiotensin-Converting Enzyme (ACE) Inhibitor"
   - atorvastatin    → "HMG-CoA Reductase Inhibitor (Statin)"
   NEVER just return the drug_class you were given. Provide the mechanism.

4. For drug_lab_interactions, think about:
   - Does this drug interfere with urine drug screens?
   - Does it affect serum electrolyte measurements?
   - Does it suppress skin/allergy test reactions?
   - Does it falsely elevate or lower any common lab value?

5. For monitoring_parameters, list what a pharmacist would actually check:
   lab values, vital signs, clinical signs specific to this drug's risks.

6. Populate dosing_by_population with specific notes wherever applicable. \
   Many drugs require pediatric weight-based dosing, renal adjustment, \
   or are on the Beers Criteria for geriatric patients."""

DRUG_ENRICHMENT_HUMAN_PROMPT = """Provide complete pharmaceutical information for this drug record:

DRUG NAME: {drug_name}
DRUG CLASS (from source formulary): {known_drug_class}
SUB-CLASS (from source formulary): {known_sub_class}

Fill ALL fields with accurate clinical information. Do NOT use placeholder values."""


class DrugLLMEnrichmentService:
    """
    Enriches drug records with clinical attributes via LLM structured output.

    RETRY LOGIC:
        Uses a manual retry loop (not tenacity) for maximum transparency.
        Each failure is logged with the exact error so it is debuggable.
        Exponential backoff: 2^attempt seconds between retries (2s, 4s, 8s...).
        After all retries exhausted, returns None → transformer uses partial doc.

    AVAILABILITY:
        The service self-disables gracefully if the LLM cannot be initialized
        (missing API key, model not found, network error). When disabled,
        is_enrichment_available returns False and all enrich calls return None.

    Args:
        max_retries_per_drug: Max LLM call attempts per drug. Default: 3.
        delay_between_calls_seconds: Throttle between sequential enrichment calls.
        llm_temperature: 0.0 = deterministic (required for clinical data).
    """

    def __init__(
        self,
        max_retries_per_drug: int = 3,
        delay_between_calls_seconds: float = 0.5,
        llm_temperature: float = 0.0,
    ):
        self._max_retries = max_retries_per_drug
        self._delay_seconds = delay_between_calls_seconds
        self._llm_temperature = llm_temperature
        self._llm_model_name = self._resolve_active_model_name()
        self._chain = self._build_structured_output_chain()
        self._is_available = self._chain is not None

        if self._is_available:
            logger.info(
                f"[ENRICHMENT] Service initialized "
                f"(model={self._llm_model_name}, "
                f"max_retries={self._max_retries}, "
                f"delay={self._delay_seconds}s)"
            )
        else:
            logger.warning(
                "[ENRICHMENT] Service UNAVAILABLE — LLM could not be initialized. "
                "Records will be ingested without enrichment."
            )

    @property
    def is_enrichment_available(self) -> bool:
        return self._is_available

    @property
    def active_llm_model_name(self) -> str:
        return self._llm_model_name

    def enrich_single_drug_record(
        self,
        drug_name: str,
        known_drug_class: str = "Unknown",
        known_sub_class: str | None = None,
    ) -> LLMDrugEnrichmentResponse | None:
        """
        Call the LLM to enrich one drug record with clinical attributes.

        Returns LLMDrugEnrichmentResponse on success, None on permanent failure.
        After returning, sleeps for delay_between_calls_seconds to throttle
        sequential API calls.
        """
        if not self._is_available:
            return None

        sub_class_display = known_sub_class or "Not specified"

        logger.debug(
            f"[ENRICHMENT] Enriching: '{drug_name}' "
            f"(class={known_drug_class}, sub_class={sub_class_display})"
        )

        result = self._call_llm_with_retry_and_backoff(
            drug_name=drug_name,
            known_drug_class=known_drug_class,
            known_sub_class=sub_class_display,
        )

        if result is not None:
            logger.debug(
                f"[ENRICHMENT] OK: '{drug_name}' → "
                f"category='{result.therapeutic_category}', "
                f"active_ingredient='{result.active_ingredient}'"
            )
        else:
            logger.warning(f"[ENRICHMENT] FAILED after all retries: '{drug_name}'")

        if self._delay_seconds > 0:
            time.sleep(self._delay_seconds)

        return result

    def get_enrichment_timestamp(self) -> datetime:
        return datetime.now(UTC)

    # =========================================================================
    # PRIVATE
    # =========================================================================

    def _build_structured_output_chain(self):
        """
        Build: ChatPromptTemplate → LLM.with_structured_output(LLMDrugEnrichmentResponse).
        Returns None if LLM initialization fails.
        """
        try:
            llm = get_llm(temperature=self._llm_temperature)
            structured_llm = llm.with_structured_output(
                LLMDrugEnrichmentResponse,
                method="json_schema",
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", DRUG_ENRICHMENT_SYSTEM_PROMPT),
                ("human", DRUG_ENRICHMENT_HUMAN_PROMPT),
            ])
            return prompt | structured_llm
        except Exception as error:
            logger.error(
                f"[ENRICHMENT] Failed to build chain: "
                f"{type(error).__name__}: {error}"
            )
            return None

    def _call_llm_with_retry_and_backoff(
        self,
        drug_name: str,
        known_drug_class: str,
        known_sub_class: str,
    ) -> LLMDrugEnrichmentResponse | None:
        """
        Call the enrichment chain with up to max_retries attempts.

        Catches ALL exceptions including Pydantic ValidationError (which is
        raised by LLMDrugEnrichmentResponse's strict model_validator when the
        LLM returns placeholder values). Each caught exception triggers the
        next retry with exponential backoff.
        """
        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug(
                    f"[ENRICHMENT] Attempt {attempt}/{self._max_retries} "
                    f"for '{drug_name}'"
                )
                raw_result = self._chain.invoke({
                    "drug_name": drug_name,
                    "known_drug_class": known_drug_class,
                    "known_sub_class": known_sub_class,
                })

                if isinstance(raw_result, LLMDrugEnrichmentResponse):
                    return raw_result

                if isinstance(raw_result, dict):
                    return LLMDrugEnrichmentResponse(**raw_result)

                logger.warning(
                    f"[ENRICHMENT] Unexpected return type "
                    f"{type(raw_result).__name__} for '{drug_name}'"
                )
                return None

            except Exception as error:
                logger.warning(
                    f"[ENRICHMENT] Attempt {attempt}/{self._max_retries} FAILED "
                    f"for '{drug_name}': {type(error).__name__}: {error}"
                )
                if attempt < self._max_retries:
                    backoff = 2 ** attempt
                    logger.debug(f"[ENRICHMENT] Backing off {backoff}s...")
                    time.sleep(backoff)

        logger.error(
            f"[ENRICHMENT] Exhausted all {self._max_retries} retries for '{drug_name}'"
        )
        return None

    @staticmethod
    def _resolve_active_model_name() -> str:
        provider = settings.llm_provider
        model_map = {
            "openai": settings.openai_model_name,
            "gemini": settings.gemini_model_name,
            "lmstudio": settings.lmstudio_model_name,
        }
        return model_map.get(provider, f"unknown-provider-{provider}")
