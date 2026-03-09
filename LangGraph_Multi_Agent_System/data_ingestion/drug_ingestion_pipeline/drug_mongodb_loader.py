"""
Drug MongoDB Loader — LOAD Phase
====================================
Bulk-upserts StandardizedDrugDocument objects into the MongoDB 'drugs'
collection and manages the indexes that support clinical agent queries.

SERIALIZATION NOTE
------------------
StandardizedDrugDocument uses `meta` as the Python field name with alias `_meta`.
When writing to MongoDB, we call model_dump(by_alias=True) so that the
sub-document is stored under the key "_meta" in MongoDB documents.

When querying the _meta sub-document in MongoDB:
    db.drugs.find({"_meta.llm_enrichment_model": {$ne: null}})

INDEX STRATEGY
--------------
Indexes are designed around the query patterns clinical agents use:

    drug_name (unique)       → exact drug lookup + upsert deduplication key
    drug_class               → "find all ACE inhibitors"
    therapeutic_category     → "find all cardiovascular drugs"
    pharmacological_class    → "find all H1 antihistamines"
    active_ingredient        → therapeutic substitution and duplicate therapy
    black_box_warning        → safety screening ("flag all BBW drugs in this regimen")
    _meta.llm_enrichment_model → "find drugs NOT yet enriched" for incremental runs

REMOVED INDEXES (compared to previous version):
    idx_is_preferred_on_formulary  → field no longer in drug master record
    idx_data_quality_score_desc    → score no longer persisted (computed at read time)
"""

import logging

from pymongo import ASCENDING, UpdateOne
from pymongo.errors import BulkWriteError

from data_ingestion.drug_ingestion_pipeline.drug_standardization_models import (
    StandardizedDrugDocument,
)
from data_ingestion.connections.mongodb_connection_manager import (
    MongoDBConnectionManager,
)

logger = logging.getLogger(__name__)

DRUGS_COLLECTION_NAME = "drugs"


class DrugMongoDBLoader:
    """
    Handles the LOAD phase: index management + bulk upsert to MongoDB.

    Args:
        mongodb_connection_manager: Active connection manager instance.
        collection_name: Target MongoDB collection name. Default: 'drugs'.
    """

    def __init__(
        self,
        mongodb_connection_manager: MongoDBConnectionManager,
        collection_name: str = DRUGS_COLLECTION_NAME,
    ):
        self._connection_manager = mongodb_connection_manager
        self._collection_name = collection_name

    def ensure_collection_indexes_exist(self) -> None:
        """
        Create or verify all indexes on the drugs collection.

        Idempotent — MongoDB create_index is a no-op if the index already exists.
        Call on every pipeline run.

        Index list (7 indexes):
            1. drug_name          (unique) — upsert key + exact lookup
            2. drug_class                  — therapeutic class filter
            3. therapeutic_category        — organ-system category filter
            4. pharmacological_class       — mechanism-based filter
            5. active_ingredient           — therapeutic substitution / duplicate detection
            6. black_box_warning           — safety screening
            7. _meta.llm_enrichment_model  — incremental enrichment tracking
        """
        collection = self._connection_manager.get_collection(self._collection_name)

        index_definitions = [
            {
                "keys": [("drug_name", ASCENDING)],
                "name": "idx_drug_name_unique",
                "unique": True,
            },
            {
                "keys": [("drug_class", ASCENDING)],
                "name": "idx_drug_class",
            },
            {
                "keys": [("therapeutic_category", ASCENDING)],
                "name": "idx_therapeutic_category",
            },
            {
                "keys": [("pharmacological_class", ASCENDING)],
                "name": "idx_pharmacological_class",
            },
            {
                "keys": [("active_ingredient", ASCENDING)],
                "name": "idx_active_ingredient",
            },
            {
                "keys": [("black_box_warning", ASCENDING)],
                "name": "idx_black_box_warning",
            },
            {
                "keys": [("_meta.llm_enrichment_model", ASCENDING)],
                "name": "idx_meta_llm_enrichment_model",
            },
        ]

        for index_def in index_definitions:
            try:
                collection.create_index(
                    index_def["keys"],
                    name=index_def["name"],
                    unique=index_def.get("unique", False),
                )
                logger.debug(f"[LOAD] Index ensured: {index_def['name']}")
            except Exception as error:
                logger.warning(
                    f"[LOAD] Could not create index '{index_def['name']}': {error}"
                )

        logger.info(
            f"[LOAD] {len(index_definitions)} indexes ensured on '{self._collection_name}'"
        )

    def upsert_standardized_drug_documents(
        self,
        standardized_documents: list[StandardizedDrugDocument],
    ) -> tuple[int, int, list[str]]:
        """
        Bulk-upsert standardized drug documents into MongoDB.

        Uses UpdateOne(filter={drug_name: ...}, update={$set: ...}, upsert=True)
        so the pipeline is fully idempotent — re-running always converges to the
        latest state without creating duplicates.

        SERIALIZATION: model_dump(by_alias=True) ensures the _meta sub-document
        uses the MongoDB key "_meta" (not the Python attribute name "meta").

        Returns:
            (new_inserts_count, updated_existing_count, error_messages)
        """
        if not standardized_documents:
            logger.warning("[LOAD] No documents to upsert")
            return 0, 0, []

        logger.info(
            f"[LOAD] Upserting {len(standardized_documents)} documents "
            f"into '{self._collection_name}'"
        )

        collection = self._connection_manager.get_collection(self._collection_name)

        upsert_operations = []
        for document in standardized_documents:
            # by_alias=True → Python field 'meta' serializes as '_meta' in MongoDB
            document_dict = document.model_dump(by_alias=True)
            upsert_operations.append(
                UpdateOne(
                    filter={"drug_name": document.drug_name},
                    update={"$set": document_dict},
                    upsert=True,
                )
            )

        new_inserts = 0
        updated_existing = 0
        error_messages: list[str] = []

        try:
            result = collection.bulk_write(upsert_operations, ordered=False)
            new_inserts = result.upserted_count
            updated_existing = result.modified_count

        except BulkWriteError as bulk_error:
            for write_error in bulk_error.details.get("writeErrors", []):
                drug_name = (
                    write_error.get("op", {}).get("q", {}).get("drug_name", "unknown")
                )
                msg = (
                    f"Write error on '{drug_name}': "
                    f"{write_error.get('errmsg', 'Unknown error')}"
                )
                error_messages.append(msg)
                logger.error(f"[LOAD] {msg}")
            new_inserts = bulk_error.details.get("nUpserted", 0)
            updated_existing = bulk_error.details.get("nModified", 0)

        except Exception as error:
            msg = f"Unexpected error during bulk upsert: {error}"
            error_messages.append(msg)
            logger.error(f"[LOAD] {msg}")

        logger.info(
            f"[LOAD] Results: {new_inserts} new, "
            f"{updated_existing} updated, "
            f"{len(error_messages)} errors"
        )
        return new_inserts, updated_existing, error_messages

    def get_drug_names_already_enriched_in_mongodb(self) -> set[str]:
        """
        Query MongoDB for drugs that already have LLM enrichment.

        A drug is considered enriched if _meta.llm_enrichment_model is not null.
        This allows incremental pipeline runs to skip expensive LLM calls for
        drugs enriched in a previous run.

        Note: Now queries _meta.llm_enrichment_model instead of the old
        top-level llm_enrichment_model field.

        Returns:
            Set of drug names (lowercase) that are already enriched.
        """
        collection = self._connection_manager.get_collection(self._collection_name)

        try:
            cursor = collection.find(
                {"_meta.llm_enrichment_model": {"$ne": None}},
                {"drug_name": 1, "_id": 0},
            )
            enriched_names = {
                doc["drug_name"].lower()
                for doc in cursor
                if "drug_name" in doc
            }
            logger.info(
                f"[LOAD] {len(enriched_names)} drugs already enriched in MongoDB"
            )
            return enriched_names

        except Exception as error:
            logger.warning(
                f"[LOAD] Could not query existing enriched drugs: {error}. "
                "All drugs will be re-enriched."
            )
            return set()
