"""
Drug Ingestion Orchestrator — Full Pipeline Orchestration
=============================================================
Orchestrates the Extract → Transform → Load pipeline for ingesting
drug data from Excel files into MongoDB with LLM-enriched clinical attributes.

STREAM-UPSERT ARCHITECTURE
----------------------------
Records are upserted into MongoDB **immediately** after each drug is enriched,
rather than waiting until all 8,829 records are processed. This means:

    - Progress is visible in MongoDB Compass in real time.
    - A crash/interrupt at record N loses at most ONE record (the in-flight one).
    - No large in-memory buffer of 8,829 documents.
    - Re-runs naturally resume from where they left off (already-enriched
      drugs are detected in Phase 0 and skipped).

PIPELINE EXECUTION FLOW
------------------------

    Phase 0  PRE-CHECK:  Query MongoDB for already-enriched drug names.
    Phase 1  EXTRACT:    Read all drug records from Excel files (fast, in-memory).
    Phase 2  INDEX:      Ensure MongoDB indexes exist (idempotent).
    Phase 3  STREAM:     For each drug record:
                             a) Parse drug name string (regex, ~0ms).
                             b) Call LLM enrichment (8–12s per record).
                             c) Merge raw + enriched → StandardizedDrugDocument.
                             d) IMMEDIATELY upsert this one document to MongoDB.
                             e) Update progress bar.
    Phase 4  REPORT:     Compile and return DrugIngestionPipelineResult.

Usage:
    from data_ingestion.connections.mongodb_connection_manager import (
    MongoDBConnectionManager,
)
    from data_ingestion.drug_ingestion_pipeline import DrugIngestionOrchestrator

    with MongoDBConnectionManager() as connection_manager:
        orchestrator = DrugIngestionOrchestrator(
            drugs_directory="drugs/",
            mongodb_connection_manager=connection_manager,
        )
        result = orchestrator.run_full_extract_transform_load_pipeline()
        print(result.formatted_summary)
"""

import logging
import sys
import time

from data_ingestion.drug_ingestion_pipeline.drug_excel_extractor import (
    DrugExcelExtractor,
)
from data_ingestion.drug_ingestion_pipeline.drug_llm_enrichment_service import (
    DrugLLMEnrichmentService,
)
from data_ingestion.drug_ingestion_pipeline.drug_mongodb_loader import (
    DrugMongoDBLoader,
)
from data_ingestion.drug_ingestion_pipeline.drug_record_transformer import (
    DrugRecordTransformer,
)
from data_ingestion.drug_ingestion_pipeline.drug_standardization_models import (
    DrugIngestionPipelineResult,
)
from data_ingestion.connections.mongodb_connection_manager import (
    MongoDBConnectionManager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# INLINE PROGRESS BAR
# Written here (not in transformer) because the orchestrator now drives the
# per-record loop and needs to update the bar after each MongoDB upsert.
# =============================================================================

class _StreamProgressBar:
    """
    Live-updating terminal progress bar for the stream-upsert loop.

    Columns shown on each update:
        [####---] xx.xx% (current/total)
        ok=N  fail=N  skip=N  saved=N  eta=XmYs  >> drug_name
    Uses only ASCII to work on Windows cp1252 / PowerShell terminals.
    """

    def __init__(self, total: int):
        self._total = max(total, 1)
        self._start_time = time.time()

    def render(
        self,
        current: int,
        enriched: int,
        failed: int,
        skipped: int,
        saved_to_db: int,
        current_drug_name: str = "",
    ) -> None:
        elapsed = max(time.time() - self._start_time, 1e-6)
        rate_per_record = elapsed / current if current else 0.0
        remaining_seconds = int(rate_per_record * (self._total - current))
        eta_min, eta_sec = divmod(remaining_seconds, 60)

        bar_width = 28
        filled = int((current / self._total) * bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)
        percent = (current / self._total) * 100

        # Truncate drug name to fit line width; ASCII-safe truncation marker
        drug_label = (
            current_drug_name[:30] + "..."
            if len(current_drug_name) > 31
            else current_drug_name.ljust(31)
        )

        line = (
            f"\r[{bar}] {percent:5.1f}%"
            f" ({current}/{self._total})"
            f"  ok={enriched} fail={failed} skip={skipped} saved={saved_to_db}"
            f"  eta={eta_min}m{eta_sec:02d}s"
            f"  >> {drug_label}"
        )
        sys.stdout.write(line)
        sys.stdout.flush()

    @staticmethod
    def finish() -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


class DrugIngestionOrchestrator:
    """
    Top-level orchestrator for the Drug Ingestion Pipeline.

    Drives the stream-upsert loop:
        for each drug → enrich → merge → upsert immediately → update bar.

    Args:
        drugs_directory: Path to drug Excel files.
        mongodb_connection_manager: Active MongoDB connection.
        max_retries_per_drug_enrichment: LLM retry attempts per drug.
        delay_between_enrichment_calls_seconds: Throttle between LLM calls.
        skip_enrichment_for_already_enriched_drugs: Skip drugs already in MongoDB.
        skip_all_llm_enrichment: Skip LLM entirely (fast Excel-only load).
    """

    def __init__(
        self,
        drugs_directory: str,
        mongodb_connection_manager: MongoDBConnectionManager,
        max_retries_per_drug_enrichment: int = 3,
        delay_between_enrichment_calls_seconds: float = 0.3,
        skip_enrichment_for_already_enriched_drugs: bool = True,
        skip_all_llm_enrichment: bool = False,
    ):
        self._drugs_directory = drugs_directory
        self._connection_manager = mongodb_connection_manager
        self._skip_already_enriched = skip_enrichment_for_already_enriched_drugs
        self._skip_all_enrichment = skip_all_llm_enrichment

        self._extractor = DrugExcelExtractor(drugs_directory=drugs_directory)

        self._enrichment_service = DrugLLMEnrichmentService(
            max_retries_per_drug=max_retries_per_drug_enrichment,
            delay_between_calls_seconds=delay_between_enrichment_calls_seconds,
        )

        self._transformer = DrugRecordTransformer(
            enrichment_service=self._enrichment_service,
            source_directory_path=drugs_directory,
        )

        self._loader = DrugMongoDBLoader(
            mongodb_connection_manager=mongodb_connection_manager,
        )

        logger.info(
            f"[ORCHESTRATOR] Initialized "
            f"(drugs_dir={drugs_directory}, "
            f"provider={self._enrichment_service.active_llm_model_name}, "
            f"skip_all_enrichment={skip_all_llm_enrichment})"
        )

    def run_full_extract_transform_load_pipeline(self) -> DrugIngestionPipelineResult:
        """
        Execute the full Extract → Transform+Load (streaming) pipeline.

        Transform and Load are FUSED into a single per-record loop:
            enrich one record → upsert immediately → next record.
        """
        pipeline_start_time = time.time()
        all_errors: list[str] = []

        self._log_header("DRUG INGESTION PIPELINE — STARTING")

        # =====================================================================
        # Phase 0: PRE-CHECK — which drugs are already enriched?
        # =====================================================================
        drug_names_already_enriched: set[str] = set()
        if self._skip_already_enriched and not self._skip_all_enrichment:
            logger.info("[ORCHESTRATOR] Phase 0: Checking MongoDB for existing enrichment...")
            drug_names_already_enriched = (
                self._loader.get_drug_names_already_enriched_in_mongodb()
            )

        # =====================================================================
        # Phase 1: EXTRACT — read all records from Excel (fast, O(records))
        # =====================================================================
        logger.info("[ORCHESTRATOR] Phase 1: EXTRACT — reading Excel files...")
        raw_records = self._extractor.extract_all_drug_records_from_excel_files()
        source_file_count = self._extractor.count_source_excel_files()

        if not raw_records:
            logger.warning("[ORCHESTRATOR] No records found in Excel files.")
            return DrugIngestionPipelineResult(
                total_drugs_extracted_from_excel=0,
                source_files_processed=source_file_count,
                pipeline_duration_seconds=round(time.time() - pipeline_start_time, 2),
                errors=["No drug records found in Excel files"],
            )

        logger.info(
            f"[ORCHESTRATOR] Phase 1 complete: "
            f"{len(raw_records)} records from {source_file_count} Excel file(s)"
        )

        # =====================================================================
        # Phase 2: INDEX — ensure indexes exist before writes start
        # =====================================================================
        logger.info("[ORCHESTRATOR] Phase 2: INDEX — ensuring MongoDB indexes...")
        self._loader.ensure_collection_indexes_exist()

        # =====================================================================
        # Phase 3: STREAM — enrich + upsert each record immediately
        # =====================================================================
        logger.info(
            f"[ORCHESTRATOR] Phase 3: STREAM — "
            f"enriching and upserting {len(raw_records)} records one-by-one..."
        )
        print(flush=True)  # blank line before progress bar

        total = len(raw_records)
        progress_bar = _StreamProgressBar(total=total)

        count_upserted_new = 0
        count_updated_existing = 0
        count_load_errors = 0

        for index, raw_record in enumerate(raw_records, start=1):
            # Determine enrichment strategy for this record
            should_skip_enrichment = (
                self._skip_all_enrichment
                or raw_record.drug_name.lower() in drug_names_already_enriched
                or not self._enrichment_service.is_enrichment_available
            )

            # -----------------------------------------------------------------
            # Transform: parse + (optionally) enrich → StandardizedDrugDocument
            # -----------------------------------------------------------------
            if should_skip_enrichment:
                document = self._transformer.transform_single_record_without_enrichment(
                    raw_record
                )
            else:
                document = self._transformer.transform_single_record_with_enrichment(
                    raw_record
                )

            # -----------------------------------------------------------------
            # Immediate per-record MongoDB upsert
            # -----------------------------------------------------------------
            if document is not None:
                new_inserts, updates, record_errors = (
                    self._loader.upsert_standardized_drug_documents([document])
                )
                count_upserted_new += new_inserts
                count_updated_existing += updates
                if record_errors:
                    count_load_errors += len(record_errors)
                    all_errors.extend(record_errors)
                    logger.warning(
                        f"[ORCHESTRATOR] Upsert error for '{raw_record.drug_name}': "
                        f"{record_errors[0]}"
                    )
            else:
                logger.error(
                    f"[ORCHESTRATOR] DROPPED '{raw_record.drug_name}' "
                    f"— transformer returned None"
                )

            # -----------------------------------------------------------------
            # Progress bar update after every record
            # -----------------------------------------------------------------
            progress_bar.render(
                current=index,
                enriched=self._transformer.enrichment_succeeded_count,
                failed=self._transformer.enrichment_failed_count,
                skipped=self._transformer.enrichment_skipped_count,
                saved_to_db=count_upserted_new + count_updated_existing,
                current_drug_name=raw_record.drug_name,
            )

        progress_bar.finish()
        all_errors.extend(self._transformer.transformation_errors)

        # =====================================================================
        # Phase 4: REPORT
        # =====================================================================
        pipeline_duration = round(time.time() - pipeline_start_time, 2)

        enriched_count = self._transformer.enrichment_succeeded_count
        avg_quality = 0.0
        if enriched_count > 0:
            # We don't hold all docs in memory — estimate from enriched fraction
            avg_quality = round(enriched_count / total, 2)

        result = DrugIngestionPipelineResult(
            total_drugs_extracted_from_excel=total,
            source_files_processed=source_file_count,
            total_drugs_enriched_by_llm=enriched_count,
            total_drugs_where_enrichment_failed=self._transformer.enrichment_failed_count,
            total_drugs_where_enrichment_was_skipped=self._transformer.enrichment_skipped_count,
            average_data_quality_score=avg_quality,
            total_documents_upserted_as_new=count_upserted_new,
            total_documents_updated_existing=count_updated_existing,
            total_documents_failed_to_load=count_load_errors,
            pipeline_duration_seconds=pipeline_duration,
            errors=all_errors,
        )

        logger.info("[ORCHESTRATOR] Pipeline complete.")
        return result

    @staticmethod
    def _log_header(message: str) -> None:
        border = "=" * 64
        logger.info(border)
        logger.info(f"  {message}")
        logger.info(border)
