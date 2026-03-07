"""
Drug Ingestion Pipeline Package
==================================
A comprehensive ETL pipeline that extracts drug data from Excel files,
enriches it with clinical attributes via LLM, validates it through
strict Pydantic models, and loads it into MongoDB.

Package Architecture:

    drug_ingestion_pipeline/
    ├── drug_standardization_models.py    ← Pydantic data contracts (4 models)
    ├── drug_excel_extractor.py           ← EXTRACT: Excel → RawDrugExcelRecord
    ├── drug_llm_enrichment_service.py    ← LLM enrichment with retry logic
    ├── drug_record_transformer.py        ← TRANSFORM: Raw + LLM → StandardizedDrugDocument
    ├── drug_mongodb_loader.py            ← LOAD: Upsert to MongoDB + index management
    └── drug_ingestion_orchestrator.py    ← Orchestrates the full E→T→L flow

Data Flow:

    drugs/*.xlsx
        │
        ▼  (DrugExcelExtractor)
    list[RawDrugExcelRecord]
        │
        ▼  (DrugRecordTransformer + DrugLLMEnrichmentService)
    list[StandardizedDrugDocument]
        │
        ▼  (DrugMongoDBLoader)
    MongoDB 'drugs' collection

Quick Start:
    from data_ingestion.mongodb_connection_manager import MongoDBConnectionManager
    from data_ingestion.drug_ingestion_pipeline import DrugIngestionOrchestrator

    with MongoDBConnectionManager() as connection_manager:
        orchestrator = DrugIngestionOrchestrator(
            drugs_directory="drugs/",
            mongodb_connection_manager=connection_manager,
        )
        result = orchestrator.run_full_extract_transform_load_pipeline()
        print(result.formatted_summary)
"""

from data_ingestion.drug_ingestion_pipeline.drug_excel_extractor import (
    DrugExcelExtractor,
)
from data_ingestion.drug_ingestion_pipeline.drug_ingestion_orchestrator import (
    DrugIngestionOrchestrator,
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
    LLMDrugEnrichmentResponse,
    RawDrugExcelRecord,
    StandardizedDrugDocument,
)

__all__ = [
    # Orchestrator (primary entry point)
    "DrugIngestionOrchestrator",
    # Pipeline components
    "DrugExcelExtractor",
    "DrugLLMEnrichmentService",
    "DrugRecordTransformer",
    "DrugMongoDBLoader",
    # Data models
    "RawDrugExcelRecord",
    "LLMDrugEnrichmentResponse",
    "StandardizedDrugDocument",
    "DrugIngestionPipelineResult",
]
