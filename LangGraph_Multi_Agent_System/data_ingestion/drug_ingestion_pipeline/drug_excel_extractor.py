"""
Drug Excel Extractor — EXTRACT Phase
========================================
Reads drug classification and preferred-drug data from Excel files in the
drugs/ directory, producing a list of RawDrugExcelRecord objects.

This module is the FIRST STAGE of the Drug Ingestion Pipeline:

    ┌──────────────────────────────────────────────────────────────────┐
    │  EXTRACT  ──────►  TRANSFORM (Validate + Enrich)  ──────►   LOAD │
    │  ^^^^^^^^                                                        │
    │  YOU ARE HERE                                                    │
    └──────────────────────────────────────────────────────────────────┘

It wraps the existing DrugDatabaseProcessor from document_processing/,
converting its DrugInfo output into RawDrugExcelRecord models that are
understood by the rest of this pipeline.

Why a Separate Extractor?
    The DrugDatabaseProcessor was built for the original simple pipeline
    and returns core.models.DrugInfo objects. This extractor ADAPTS that
    output into RawDrugExcelRecord objects (with '_from_excel' suffixed
    fields) so the new pipeline has a clean boundary between raw source
    data and enriched/standardized data.

Expected Excel Files in drugs/:
    1. drug_class_sub-class.xlsx  →  drug_name, drug_class, sub_class
    2. preferred_drugs.xlsx       →  drug_name, is_preferred overlay

Usage:
    extractor = DrugExcelExtractor(drugs_directory="drugs/")
    raw_records = extractor.extract_all_drug_records_from_excel_files()
"""

import logging
import os

from core.models import DrugInfo
from data_ingestion.drug_ingestion_pipeline.drug_standardization_models import (
    RawDrugExcelRecord,
)
from document_processing.excel_processor import DrugDatabaseProcessor

logger = logging.getLogger(__name__)


class DrugExcelExtractor:
    """
    Extracts raw drug records from Excel files in the drugs/ directory.

    This class is responsible for the EXTRACT phase of the ETL pipeline.
    It reads Excel files using the existing DrugDatabaseProcessor, then
    converts each DrugInfo object into a RawDrugExcelRecord for downstream
    processing by the Transform phase.

    Args:
        drugs_directory: Absolute or relative path to the directory
                         containing drug Excel files.
    """

    def __init__(self, drugs_directory: str):
        self._drugs_directory = os.path.abspath(drugs_directory)

        if not os.path.isdir(self._drugs_directory):
            raise FileNotFoundError(
                f"Drugs directory not found: {self._drugs_directory}. "
                f"Ensure the 'drugs/' folder exists and contains the Excel files."
            )

        logger.info(
            f"[EXTRACT] DrugExcelExtractor initialized with directory: "
            f"{self._drugs_directory}"
        )

    def extract_all_drug_records_from_excel_files(self) -> list[RawDrugExcelRecord]:
        """
        Read all Excel files in the drugs directory and return raw records.

        This method:
            1. Uses DrugDatabaseProcessor to read and merge drug_class_sub-class.xlsx
               and preferred_drugs.xlsx.
            2. Converts each DrugInfo object to a RawDrugExcelRecord.
            3. Logs extraction statistics.

        Returns:
            List of RawDrugExcelRecord objects with unprocessed Excel data.
            Returns an empty list if no valid records are found.

        Raises:
            FileNotFoundError: If the drugs directory does not exist.
            DocumentProcessingError: If Excel files cannot be parsed.
        """
        logger.info(
            f"[EXTRACT] Starting drug extraction from Excel files in: "
            f"{self._drugs_directory}"
        )

        drug_database_processor = DrugDatabaseProcessor(self._drugs_directory)
        drug_info_objects: list[DrugInfo] = drug_database_processor.load_all()

        logger.info(
            f"[EXTRACT] DrugDatabaseProcessor returned "
            f"{len(drug_info_objects)} DrugInfo objects"
        )

        raw_records: list[RawDrugExcelRecord] = []
        skipped_count = 0

        for drug_info in drug_info_objects:
            try:
                raw_record = self._convert_drug_info_to_raw_excel_record(drug_info)
                raw_records.append(raw_record)
            except Exception as conversion_error:
                skipped_count += 1
                logger.warning(
                    f"[EXTRACT] Skipped drug '{drug_info.drug_name}': "
                    f"{conversion_error}"
                )

        logger.info(
            f"[EXTRACT] Extraction complete: "
            f"{len(raw_records)} valid records, "
            f"{skipped_count} skipped"
        )
        return raw_records

    def count_source_excel_files(self) -> int:
        """Count how many .xlsx files exist in the drugs directory."""
        xlsx_files = [
            f for f in os.listdir(self._drugs_directory)
            if f.endswith(".xlsx") and not f.startswith("~$")
        ]
        return len(xlsx_files)

    @staticmethod
    def _convert_drug_info_to_raw_excel_record(
        drug_info: DrugInfo,
    ) -> RawDrugExcelRecord:
        """
        Map a DrugInfo object (from the document_processing module) to a
        RawDrugExcelRecord (used by this pipeline).

        This is a straightforward field mapping with explicit naming to
        distinguish raw Excel values from enriched/standardized values.
        """
        return RawDrugExcelRecord(
            drug_name=drug_info.drug_name,
            drug_class_from_excel=drug_info.drug_class or "Unknown",
            sub_class_from_excel=drug_info.sub_class,
            is_preferred_on_formulary=drug_info.is_preferred,
        )
