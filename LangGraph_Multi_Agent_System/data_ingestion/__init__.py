"""
Data Ingestion Package
========================
ETL pipelines for loading medical data into MongoDB:

    - **Guidelines Ingestion Pipeline** — PDF clinical guidelines are extracted
      via Docling, chunked with parent-child strategy, deduplicated (3-layer),
      embedded, and persisted to ChromaDB (vectors) and MongoDB (job registry +
      parent chunks). Entry point: ``run_guidelines_ingestion.py``.

    - **Drug Ingestion Pipeline** — Excel drug data is extracted, enriched via LLM
      with clinical attributes (mechanism, indications, contraindications, etc.),
      validated through Pydantic models, and upserted to MongoDB with standardized
      schema. Entry point: ``run_drug_ingestion.py``.

Infrastructure:
    The ``connections/`` subpackage provides connection managers (MongoDB,
    ChromaDB) reused by both pipelines, along with validation and setup scripts.

Quick Start:
    # 1. Set up MongoDB infrastructure (one-time):
    python -m data_ingestion.infrastructure.infrastructure_setup

    # 2. Run guidelines ingestion:
    python -m data_ingestion.run_guidelines_ingestion

    # 3. Run drug ingestion:
    python -m data_ingestion.run_drug_ingestion
"""

from data_ingestion.drug_ingestion_pipeline import (
    DrugIngestionOrchestrator,
    StandardizedDrugDocument,
    DrugIngestionPipelineResult as StandardizedDrugIngestionPipelineResult,
)
from data_ingestion.connections.mongodb_connection_manager import (
    MongoDBConnectionManager,
)
from data_ingestion.connections.chroma_connection_manager import (
    ChromaConnectionManager,
)

__all__ = [
    # Drug Ingestion Pipeline
    "DrugIngestionOrchestrator",
    "StandardizedDrugDocument",
    "StandardizedDrugIngestionPipelineResult",
    # Shared infrastructure
    "MongoDBConnectionManager",
    "ChromaConnectionManager",
]
