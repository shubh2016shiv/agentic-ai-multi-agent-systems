"""
Drug Ingestion Only — CLI Runner
===================================
Runs ONLY the Drug Ingestion Pipeline (standardized + LLM-enriched), without
the Medical Guideline pipeline. Use this to validate that drug ingestion works
reliably, efficiently, and scales.

Usage:
    # From project root (LangGraph_Multi_Agent_System):
    uv run python -m data_ingestion.run_drug_ingestion_only

    # Skip LLM enrichment (Excel-only load, fast):
    uv run python -m data_ingestion.run_drug_ingestion_only --skip-enrichment

    # Dry-run: extract + transform only, no MongoDB (no DB required):
    uv run python -m data_ingestion.run_drug_ingestion_only --dry-run

Prerequisites:
    - MongoDB running (e.g. docker compose up via infrastructure/manager.py)
    - drugs/*.xlsx present (drug_class_sub-class.xlsx, preferred_drugs.xlsx)
    - For enrichment: LLM provider configured in .env (OPENAI_API_KEY or GEMINI_API_KEY etc.)
"""

import logging
import os
import sys

# Flush stdout so output appears immediately when run in background/CI
def _log(msg: str) -> None:
    print(msg, flush=True)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DRUGS_DIRECTORY = os.path.join(_PROJECT_ROOT, "drugs")


def main() -> int:
    """Run only the drug ingestion pipeline. Returns 0 on success, 1 on failure."""
    print("Drug Ingestion Only — starting.", flush=True)
    skip_enrichment = "--skip-enrichment" in sys.argv
    dry_run = "--dry-run" in sys.argv

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    _log("Loading modules...")
    from data_ingestion.drug_ingestion_pipeline.drug_excel_extractor import DrugExcelExtractor
    from data_ingestion.drug_ingestion_pipeline.drug_ingestion_orchestrator import DrugIngestionOrchestrator
    from data_ingestion.drug_ingestion_pipeline.drug_llm_enrichment_service import DrugLLMEnrichmentService
    from data_ingestion.drug_ingestion_pipeline.drug_record_transformer import DrugRecordTransformer
    from data_ingestion.connections.mongodb_connection_manager import MongoDBConnectionManager

    _log("")
    _log("=" * 64)
    _log("  Drug Ingestion Pipeline — STANDALONE RUN")
    _log("=" * 64)

    # Validate drugs directory
    if not os.path.isdir(DEFAULT_DRUGS_DIRECTORY):
        _log(f"[ERROR] Drugs directory not found: {DEFAULT_DRUGS_DIRECTORY}")
        return 1
    xlsx_count = len(
        [f for f in os.listdir(DEFAULT_DRUGS_DIRECTORY) if f.lower().endswith(".xlsx")]
    )
    _log(f"[CHECK] Drugs directory: {DEFAULT_DRUGS_DIRECTORY} ({xlsx_count} Excel files)")
    if xlsx_count == 0:
        _log("[ERROR] No .xlsx files in drugs directory.")
        return 1

    if dry_run:
        _log("\n[DRY-RUN] Extract + transform only (no MongoDB connection).")
        from data_ingestion.drug_ingestion_pipeline import (
            DrugExcelExtractor,
            DrugRecordTransformer,
            DrugLLMEnrichmentService,
        )
        extractor = DrugExcelExtractor(drugs_directory=DEFAULT_DRUGS_DIRECTORY)
        raw = extractor.extract_all_drug_records_from_excel_files()
        _log(f"[DRY-RUN] Extracted {len(raw)} raw records.")
        enrichment = DrugLLMEnrichmentService()
        transformer = DrugRecordTransformer(enrichment_service=enrichment, source_directory_path=DEFAULT_DRUGS_DIRECTORY)
        docs = transformer.transform_raw_records_into_standardized_documents(raw, skip_all_enrichment=True)
        _log(f"[DRY-RUN] Transformed to {len(docs)} standardized documents.")
        if docs:
            avg = sum(d.data_quality_score for d in docs) / len(docs)
            _log(f"[DRY-RUN] Average data_quality_score: {avg:.2f}")
        _log("[DRY-RUN] Done. Exiting without loading to MongoDB.")
        return 0

    # Validate MongoDB connectivity (uses 5s timeout in MongoDBConnectionManager)
    _log("\n[CHECK] MongoDB connectivity...")
    with MongoDBConnectionManager(server_selection_timeout_ms=5000) as connection_manager:
        if not connection_manager.verify_connection_health():
            _log(
                "[ERROR] MongoDB is not reachable. Start infrastructure first, e.g.:\n"
                "  python -m infrastructure.manager start"
            )
            return 1
        _log("[OK]    MongoDB is reachable.")

        # Run pipeline (reuse same connection via context)
        _log("\n[RUN] Starting Drug Ingestion Pipeline...")
        orchestrator = DrugIngestionOrchestrator(
            drugs_directory=DEFAULT_DRUGS_DIRECTORY,
            mongodb_connection_manager=connection_manager,
            skip_all_llm_enrichment=skip_enrichment,
        )
        result = orchestrator.run_full_extract_transform_load_pipeline()

    # Summary is already logged by the orchestrator; print once for CLI
    _log("")
    _log(result.formatted_summary)

    has_errors = (
        result.total_documents_failed_to_load > 0
        or len(result.errors) > 0
    )
    return 0 if not has_errors else 1


if __name__ == "__main__":
    sys.exit(main())
