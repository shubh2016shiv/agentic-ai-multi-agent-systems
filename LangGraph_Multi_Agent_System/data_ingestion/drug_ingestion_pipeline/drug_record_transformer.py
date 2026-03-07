"""
Drug Record Transformer — TRANSFORM Phase
=============================================
Converts raw Excel drug records into fully standardized, LLM-enriched
StandardizedDrugDocument objects ready for MongoDB upsert.

WHAT HAPPENS IN THIS PHASE
----------------------------
For EACH raw drug record, the transformer applies three operations in sequence:

    1. PARSE the drug name string for embedded structured fields:
           "Pharbedryl Capsule 25 Mg Oral"
            → dosage_form="Capsule", dosage_strength="25 Mg", route hint="Oral"
       This uses DrugNameParser — no LLM needed, pure regex.

    2. ENRICH via LLM for clinical attributes NOT in the Excel data:
           active_ingredient, mechanism_of_action, indications, contraindications,
           drug_interactions, drug_lab_interactions, monitoring_parameters,
           dosing_by_population, brand_names, etc.
       The LLM enrichment service handles its own retries with exponential backoff.

    3. MERGE everything into a StandardizedDrugDocument with a _meta provenance
       sub-document containing source_file, llm model name, and timestamps.

GRACEFUL DEGRADATION
---------------------
If LLM enrichment fails after all retries, the transformer creates a PARTIAL
document using only the Excel fields + parsed drug name fields. The document
is still valid and gets loaded into MongoDB — just with lower data quality.
No drug record is ever dropped.

PROVENANCE (_meta)
------------------
The _meta sub-document is populated for EVERY document:
    - source_file:             relative path (e.g., "drugs/") — NOT an absolute system path
    - llm_enrichment_model:    model name if enrichment succeeded, else None
    - llm_enrichment_timestamp: UTC timestamp if enrichment succeeded, else None
    - ingested_at:             current UTC timestamp

is_preferred_on_formulary NOTE
--------------------------------
This field IS present in the RawDrugExcelRecord (it comes from preferred_drugs.xlsx).
It is intentionally NOT written to StandardizedDrugDocument because:
    - Formulary preference is an organizational attribute, not a drug attribute.
    - It varies per health system.
    - It belongs in a separate formulary collection keyed by (org_id, drug_name).
"""

import logging
import sys
import time
from typing import Optional

from pydantic import ValidationError

from data_ingestion.drug_ingestion_pipeline.drug_llm_enrichment_service import (
    DrugLLMEnrichmentService,
)
from data_ingestion.drug_ingestion_pipeline.drug_name_parser import DrugNameParser
from data_ingestion.drug_ingestion_pipeline.drug_standardization_models import (
    DrugRecordProvenance,
    LLMDrugEnrichmentResponse,
    RawDrugExcelRecord,
    StandardizedDrugDocument,
)

logger = logging.getLogger(__name__)

# Relative path label written to _meta.source_file — NOT an absolute system path.
_RELATIVE_SOURCE_LABEL = "drugs/"


class _ConsoleProgressBar:
    """
    Lightweight terminal progress bar with ETA and counters.

    Designed for long-running enrichment loops where each record may take
    multiple seconds due to LLM calls.
    """

    def __init__(self, total: int):
        self._total = max(total, 1)
        self._start_time = time.time()

    def update(
        self,
        current: int,
        enriched_count: int,
        failed_count: int,
        skipped_count: int,
    ) -> None:
        elapsed = max(time.time() - self._start_time, 1e-6)
        rate = current / elapsed
        remaining = self._total - current
        eta_seconds = int(remaining / rate) if rate > 0 else 0
        bar_width = 30
        completed = int((current / self._total) * bar_width)
        bar = "#" * completed + "-" * (bar_width - completed)
        percent = (current / self._total) * 100

        sys.stdout.write(
            "\r"
            f"[TRANSFORM] |{bar}| {percent:6.2f}% "
            f"({current}/{self._total}) "
            f"enriched={enriched_count} failed={failed_count} skipped={skipped_count} "
            f"eta={eta_seconds}s"
        )
        sys.stdout.flush()

    @staticmethod
    def finish() -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


class DrugRecordTransformer:
    """
    Transforms raw Excel records → StandardizedDrugDocument via parse + enrich + merge.

    FLOW:
        RawDrugExcelRecord
            ↓  DrugNameParser.parse()        (synchronous, no LLM)
            ↓  DrugLLMEnrichmentService      (LLM call with retries)
            ↓  _merge_into_standardized_doc  (Pydantic construction + validation)
            ↓  StandardizedDrugDocument

    COUNTERS (accessible after transform):
        enrichment_succeeded_count   — drugs with full LLM enrichment
        enrichment_failed_count      — drugs where LLM failed, partial doc used
        enrichment_skipped_count     — drugs skipped per caller instruction

    Args:
        enrichment_service: Initialized DrugLLMEnrichmentService.
        source_directory_path: Used as the source_file label in _meta.
    """

    def __init__(
        self,
        enrichment_service: DrugLLMEnrichmentService,
        source_directory_path: str = _RELATIVE_SOURCE_LABEL,
    ):
        self._enrichment_service = enrichment_service
        self._source_label = source_directory_path

        self._count_enrichment_succeeded = 0
        self._count_enrichment_failed = 0
        self._count_enrichment_skipped = 0
        self._errors: list[str] = []

    @property
    def enrichment_succeeded_count(self) -> int:
        return self._count_enrichment_succeeded

    @property
    def enrichment_failed_count(self) -> int:
        return self._count_enrichment_failed

    @property
    def enrichment_skipped_count(self) -> int:
        return self._count_enrichment_skipped

    @property
    def transformation_errors(self) -> list[str]:
        return list(self._errors)

    # =========================================================================
    # PUBLIC: Single-Record Entry Points (used by stream-upsert orchestrator)
    # =========================================================================

    def transform_single_record_with_enrichment(
        self,
        raw_record: RawDrugExcelRecord,
    ) -> Optional[StandardizedDrugDocument]:
        """
        Enrich a single drug record via LLM, then build a StandardizedDrugDocument.

        Called by the orchestrator in the stream-upsert loop so that each record
        is saved to MongoDB immediately after enrichment — no batch buffering.

        Falls back to a partial (no-enrichment) document if LLM fails after all retries.
        Increments the appropriate counter (succeeded / failed) for the final report.
        """
        return self._transform_with_enrichment(raw_record)

    def transform_single_record_without_enrichment(
        self,
        raw_record: RawDrugExcelRecord,
    ) -> Optional[StandardizedDrugDocument]:
        """
        Build a StandardizedDrugDocument from Excel data + drug name parsing only.

        No LLM call is made. Used for:
            - Drugs already enriched in MongoDB (skip_already_enriched=True).
            - skip_all_enrichment=True runs.
            - Records where the enrichment service is unavailable.

        Increments enrichment_skipped_count.
        """
        self._count_enrichment_skipped += 1
        return self._transform_without_enrichment(raw_record)

    # =========================================================================
    # PUBLIC: Batch Transformation Entry Point (legacy / testing)
    # =========================================================================

    def transform_raw_records_into_standardized_documents(
        self,
        raw_records: list[RawDrugExcelRecord],
        drug_names_to_skip_enrichment: set[str] | None = None,
        skip_all_enrichment: bool = False,
    ) -> list[StandardizedDrugDocument]:
        """
        Transform a batch of raw Excel records into standardized documents.

        For each record:
            1. Always: parse drug name for dosage_form, dosage_strength, route hint.
            2. If enrichment not skipped: call LLM for clinical attributes.
            3. Merge parsed + enriched data into StandardizedDrugDocument.
            4. On any failure: create a partial document (never drop a record).

        Args:
            raw_records: Output from DrugExcelExtractor.
            drug_names_to_skip_enrichment: Lowercase drug names already enriched
                in MongoDB. Skips redundant LLM calls for these.
            skip_all_enrichment: If True, skip LLM for ALL records (fast re-load).

        Returns:
            List of StandardizedDrugDocument ready for MongoDB upsert.
        """
        if drug_names_to_skip_enrichment is None:
            drug_names_to_skip_enrichment = set()

        total = len(raw_records)
        logger.info(
            f"[TRANSFORM] Starting: {total} records, "
            f"skip_all={skip_all_enrichment}, "
            f"pre_enriched={len(drug_names_to_skip_enrichment)}"
        )

        standardized_documents: list[StandardizedDrugDocument] = []
        progress_bar = _ConsoleProgressBar(total)

        for index, raw_record in enumerate(raw_records, start=1):
            progress = f"[TRANSFORM] [{index}/{total}]"

            should_skip = (
                skip_all_enrichment
                or raw_record.drug_name.lower() in drug_names_to_skip_enrichment
                or not self._enrichment_service.is_enrichment_available
            )

            if should_skip:
                logger.debug(f"{progress} Skipping enrichment: '{raw_record.drug_name}'")
                document = self._transform_without_enrichment(raw_record)
                self._count_enrichment_skipped += 1
            else:
                logger.debug(f"{progress} Enriching: '{raw_record.drug_name}'")
                document = self._transform_with_enrichment(raw_record)

            if document is not None:
                standardized_documents.append(document)
            else:
                logger.error(
                    f"{progress} DROPPED: Could not create any document "
                    f"for '{raw_record.drug_name}'"
                )
            progress_bar.update(
                current=index,
                enriched_count=self._count_enrichment_succeeded,
                failed_count=self._count_enrichment_failed,
                skipped_count=self._count_enrichment_skipped,
            )
        progress_bar.finish()

        avg_quality = self._compute_average_quality_score(standardized_documents)
        logger.info(
            f"[TRANSFORM] Complete: {len(standardized_documents)} docs, "
            f"enriched={self._count_enrichment_succeeded}, "
            f"failed={self._count_enrichment_failed}, "
            f"skipped={self._count_enrichment_skipped}, "
            f"avg_quality={avg_quality:.2f}"
        )

        return standardized_documents

    # =========================================================================
    # PRIVATE: Transform With Enrichment
    # =========================================================================

    def _transform_with_enrichment(
        self,
        raw_record: RawDrugExcelRecord,
    ) -> Optional[StandardizedDrugDocument]:
        """
        Call the LLM, then merge raw + enriched data into a document.
        Falls back to _transform_without_enrichment on any failure.
        """
        enrichment = self._enrichment_service.enrich_single_drug_record(
            drug_name=raw_record.drug_name,
            known_drug_class=raw_record.drug_class_from_excel,
            known_sub_class=raw_record.sub_class_from_excel,
        )

        if enrichment is not None:
            document = self._merge_raw_record_with_llm_enrichment(raw_record, enrichment)
            if document is not None:
                self._count_enrichment_succeeded += 1
                return document

        self._count_enrichment_failed += 1
        self._errors.append(
            f"Enrichment failed for '{raw_record.drug_name}' — using partial document"
        )
        return self._transform_without_enrichment(raw_record)

    # =========================================================================
    # PRIVATE: Merge Raw + LLM Enrichment
    # =========================================================================

    def _merge_raw_record_with_llm_enrichment(
        self,
        raw_record: RawDrugExcelRecord,
        enrichment: LLMDrugEnrichmentResponse,
    ) -> Optional[StandardizedDrugDocument]:
        """
        Construct a StandardizedDrugDocument by combining:
            - raw_record:   drug_name, drug_class, sub_class  (from Excel)
            - parser:       dosage_strength, dosage_form, route hint  (from name string)
            - enrichment:   all clinical attributes  (from LLM)
            - provenance:   _meta sub-document  (pipeline metadata)

        Route Merge Logic:
            The LLM returns the complete route_of_administration list.
            If the parser also detected a route from the drug name, and it is NOT
            already in the LLM list, it is prepended (parsed data takes priority
            for the specific product — the LLM knows the general drug, but the
            product name string is authoritative for this specific formulation).
        """
        parsed = DrugNameParser.parse(raw_record.drug_name)

        # Merge parsed route into LLM route list (avoid duplicates)
        merged_routes = list(enrichment.route_of_administration)
        if parsed.parsed_route and parsed.parsed_route not in merged_routes:
            merged_routes.insert(0, parsed.parsed_route)

        # Prefer parsed dosage_form if LLM returned a less specific form
        # (parser is operating on THIS specific product's name string)
        resolved_dosage_form = parsed.dosage_form

        provenance = DrugRecordProvenance(
            source_file=_RELATIVE_SOURCE_LABEL,
            llm_enrichment_model=self._enrichment_service.active_llm_model_name,
            llm_enrichment_timestamp=self._enrichment_service.get_enrichment_timestamp(),
        )

        try:
            return StandardizedDrugDocument(
                drug_name=raw_record.drug_name,
                active_ingredient=enrichment.active_ingredient,
                brand_names=enrichment.brand_names,
                therapeutic_category=enrichment.therapeutic_category,
                drug_class=raw_record.drug_class_from_excel,
                pharmacological_class=enrichment.pharmacological_class,
                sub_class=raw_record.sub_class_from_excel,
                dosage_strength=parsed.dosage_strength,
                dosage_form=resolved_dosage_form,
                mechanism_of_action=enrichment.mechanism_of_action,
                common_indications=enrichment.common_indications,
                contraindications=enrichment.contraindications,
                common_side_effects=enrichment.common_side_effects,
                serious_adverse_effects=enrichment.serious_adverse_effects,
                drug_interactions=enrichment.drug_interactions,
                drug_lab_interactions=enrichment.drug_lab_interactions,
                monitoring_parameters=enrichment.monitoring_parameters,
                route_of_administration=merged_routes,
                dosage_forms=enrichment.dosage_forms,
                dosing_by_population=enrichment.dosing_by_population,
                pregnancy_category=enrichment.pregnancy_category,
                controlled_substance_schedule=enrichment.controlled_substance_schedule,
                black_box_warning=enrichment.black_box_warning,
                **{"_meta": provenance},
            )
        except ValidationError as error:
            logger.error(
                f"[TRANSFORM] Pydantic validation failed after full enrichment "
                f"for '{raw_record.drug_name}': {error}"
            )
            return None

    # =========================================================================
    # PRIVATE: Transform Without Enrichment (Partial Document)
    # =========================================================================

    def _transform_without_enrichment(
        self,
        raw_record: RawDrugExcelRecord,
    ) -> Optional[StandardizedDrugDocument]:
        """
        Create a StandardizedDrugDocument using ONLY Excel data + drug name parsing.

        No LLM is called. Clinical fields default to empty lists / "Unknown".
        The data_quality_score property will return ~0.0 for these records,
        identifying them as candidates for future enrichment.

        _meta.llm_enrichment_model is None (indicating enrichment was NOT done).
        """
        parsed = DrugNameParser.parse(raw_record.drug_name)

        provenance = DrugRecordProvenance(
            source_file=_RELATIVE_SOURCE_LABEL,
            llm_enrichment_model=None,
            llm_enrichment_timestamp=None,
        )

        try:
            return StandardizedDrugDocument(
                drug_name=raw_record.drug_name,
                drug_class=raw_record.drug_class_from_excel,
                sub_class=raw_record.sub_class_from_excel,
                dosage_strength=parsed.dosage_strength,
                dosage_form=parsed.dosage_form,
                route_of_administration=(
                    [parsed.parsed_route] if parsed.parsed_route else []
                ),
                **{"_meta": provenance},
            )
        except ValidationError as error:
            logger.error(
                f"[TRANSFORM] Cannot build even a minimal document "
                f"for '{raw_record.drug_name}': {error}"
            )
            self._errors.append(
                f"Minimal document construction failed for '{raw_record.drug_name}': {error}"
            )
            return None

    # =========================================================================
    # PRIVATE: Utilities
    # =========================================================================

    @staticmethod
    def _compute_average_quality_score(
        documents: list[StandardizedDrugDocument],
    ) -> float:
        """Compute the mean data_quality_score property across a list of documents."""
        if not documents:
            return 0.0
        return round(
            sum(doc.data_quality_score for doc in documents) / len(documents),
            2,
        )
