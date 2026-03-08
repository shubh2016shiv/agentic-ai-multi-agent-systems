"""
Guidelines Ingestion Runner
==============================
CLI entry point for ingesting all clinical guideline PDFs from
medical_guidelines/ into ChromaDB (vectors) and MongoDB (job registry +
parent chunks) via the guidelines_ingestion_pipeline module.

Usage (from project root):
    python -m data_ingestion.run_full_ingestion

    # Or directly:
    python data_ingestion/run_full_ingestion.py

Environment:
    Reads from .env in the project root.
    Required: GUIDELINES_PIPELINE_MONGODB_URI
    Optional: all other GUIDELINES_PIPELINE_* settings have sensible defaults.

Prerequisites:
    - MongoDB running (e.g. docker compose up)
    - medical_guidelines/*.pdf present
"""

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path — ensures imports work however the script is run.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Load .env before importing PipelineSettings so env vars are present when
# Pydantic validates required fields (e.g. GUIDELINES_PIPELINE_MONGODB_URI).
# ---------------------------------------------------------------------------
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env", override=False)

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
from data_ingestion.guidelines_ingestion_pipeline.application.ingestion_pipeline import (  # noqa: E402
    GuidelinesIngestionPipeline,
)
from data_ingestion.guidelines_ingestion_pipeline.config.pipeline_settings import (  # noqa: E402
    PipelineSettings,
)
from data_ingestion.guidelines_ingestion_pipeline.domain.models.document_metadata import (  # noqa: E402
    GuidelineMetadata,
)
from data_ingestion.guidelines_ingestion_pipeline.domain.models.ingestion_job import (  # noqa: E402
    IngestionStatus,
)
from data_ingestion.guidelines_ingestion_pipeline.utils.logging_utils import (  # noqa: E402
    configure_structlog,
)

configure_structlog()

import structlog  # noqa: E402

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Source directory
# ---------------------------------------------------------------------------
GUIDELINES_DIR = _PROJECT_ROOT / "medical_guidelines"

# ---------------------------------------------------------------------------
# Per-PDF metadata
# Add an entry here whenever a new guideline PDF is placed in medical_guidelines/.
# ---------------------------------------------------------------------------
METADATA_MAP: dict[str, GuidelineMetadata] = {
    "Cancer.pdf": GuidelineMetadata(
        guideline_org="NCCN",
        guideline_year=2024,
        therapeutic_area="oncology",
        condition_focus="cancer",
        pdf_name="Cancer.pdf",
        pdf_source_path=str(GUIDELINES_DIR),
    ),
    "CKD KDIGO 2024.pdf": GuidelineMetadata(
        guideline_org="KDIGO",
        guideline_year=2024,
        therapeutic_area="nephrology",
        condition_focus="chronic kidney disease",
        pdf_name="CKD KDIGO 2024.pdf",
        pdf_source_path=str(GUIDELINES_DIR),
    ),
    "COPD GOLD 2024.pdf": GuidelineMetadata(
        guideline_org="GOLD",
        guideline_year=2024,
        therapeutic_area="pulmonology",
        condition_focus="chronic obstructive pulmonary disease",
        pdf_name="COPD GOLD 2024.pdf",
        pdf_source_path=str(GUIDELINES_DIR),
    ),
    "depression.pdf": GuidelineMetadata(
        guideline_org="APA",
        guideline_year=2023,
        therapeutic_area="psychiatry",
        condition_focus="major depressive disorder",
        pdf_name="depression.pdf",
        pdf_source_path=str(GUIDELINES_DIR),
    ),
    "hyperlipidemia.pdf": GuidelineMetadata(
        guideline_org="ACC/AHA",
        guideline_year=2022,
        therapeutic_area="cardiology",
        condition_focus="hyperlipidemia",
        pdf_name="hyperlipidemia.pdf",
        pdf_source_path=str(GUIDELINES_DIR),
    ),
    "hypertension.pdf": GuidelineMetadata(
        guideline_org="ACC/AHA",
        guideline_year=2022,
        therapeutic_area="cardiology",
        condition_focus="hypertension",
        pdf_name="hypertension.pdf",
        pdf_source_path=str(GUIDELINES_DIR),
    ),
}


def _metadata_resolver(pdf_path: Path) -> GuidelineMetadata:
    """
    Return GuidelineMetadata for a given PDF path.

    Falls back to a minimal default entry for any PDF not listed in
    METADATA_MAP, so that newly dropped PDFs are ingested rather than skipped.
    """
    key = pdf_path.name
    if key not in METADATA_MAP:
        logger.warning("metadata_not_found_using_default", pdf_name=key)
        return GuidelineMetadata(
            guideline_org="Unknown",
            guideline_year=2024,
            therapeutic_area="general",
            condition_focus=key.replace(".pdf", "").replace("_", " "),
            pdf_name=key,
            pdf_source_path=str(pdf_path.parent),
        )
    return METADATA_MAP[key]


def _print_summary(jobs: list) -> None:
    """Print a formatted per-PDF result table to stdout."""
    print("\n" + "=" * 72)
    print("  GUIDELINES INGESTION SUMMARY")
    print("=" * 72)
    print(f"  {'PDF Name':<35} {'Status':<12} {'Chunks':>8}")
    print("-" * 72)

    status_counts: dict[str, int] = {}
    for job in jobs:
        label = job.status.value
        status_counts[label] = status_counts.get(label, 0) + 1
        chunks = job.total_chunks if job.total_chunks else "-"
        print(f"  {job.pdf_name:<35} {label:<12} {str(chunks):>8}")

    print("=" * 72)
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print("=" * 72 + "\n")


def main() -> int:
    """
    Run the guidelines ingestion pipeline.

    Returns:
        0 on success, 1 if any PDF failed to ingest.
    """
    if not GUIDELINES_DIR.exists():
        logger.error("guidelines_directory_not_found", path=str(GUIDELINES_DIR))
        print(f"[ERROR] Directory not found: {GUIDELINES_DIR}", file=sys.stderr)
        return 1

    pdf_count = len(list(GUIDELINES_DIR.glob("*.pdf")))
    logger.info(
        "ingestion_runner_started",
        guidelines_dir=str(GUIDELINES_DIR),
        pdf_count=pdf_count,
    )

    try:
        settings = PipelineSettings()
    except Exception as exc:
        logger.error("settings_validation_failed", error=str(exc))
        print(
            f"\n[ERROR] Could not load PipelineSettings: {exc}\n"
            "Ensure GUIDELINES_PIPELINE_MONGODB_URI is set in .env or environment.\n",
            file=sys.stderr,
        )
        return 1

    logger.info(
        "pipeline_settings_loaded",
        mongodb_database=settings.mongodb_database,
        chroma_collection=settings.chroma_collection_name,
        embedding_model=settings.embedding_model_name,
        pipeline_version=settings.pipeline_version,
    )

    try:
        pipeline = GuidelinesIngestionPipeline(settings)
    except Exception as exc:
        logger.error("pipeline_initialization_failed", error=str(exc))
        print(f"\n[ERROR] Pipeline initialization failed: {exc}\n", file=sys.stderr)
        return 1

    jobs = pipeline.ingest_directory(
        directory_path=GUIDELINES_DIR,
        metadata_resolver=_metadata_resolver,
    )

    _print_summary(jobs)

    failed = [j for j in jobs if j.status == IngestionStatus.FAILED]
    if failed:
        logger.warning("ingestion_completed_with_failures", failed_count=len(failed))
        for j in failed:
            print(f"  [FAILED] {j.pdf_name}: {j.error_message}", file=sys.stderr)
        return 1

    logger.info(
        "ingestion_runner_complete",
        total_jobs=len(jobs),
        successful=sum(1 for j in jobs if j.status == IngestionStatus.INGESTED),
        skipped=sum(1 for j in jobs if j.status == IngestionStatus.SKIPPED),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # suppress verbose third-party noise
    sys.exit(main())
