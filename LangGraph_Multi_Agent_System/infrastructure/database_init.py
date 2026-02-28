"""
Database Initialization Script
==============================
Initializes MongoDB collections and indexes for the Medical Knowledge Base.
This script expects the infrastructure to be running and healthy.

It is idempotent, so running it multiple times is safe.
"""

import logging
from core.config import settings

logger = logging.getLogger(__name__)

MONGODB_DATABASE_NAME = "medical_knowledge_base"


def initialize_mongodb_database_and_collections(mongodb_uri: str) -> bool:
    """
    Create the database, collections, and indexes in MongoDB.
    
    Args:
        mongodb_uri: Connection URI to MongoDB.
    """
    print(f"\n[SETUP] Initializing database: {MONGODB_DATABASE_NAME}")

    try:
        from pymongo import MongoClient, ASCENDING

        client = MongoClient(mongodb_uri)
        database = client[MONGODB_DATABASE_NAME]

        # -----------------------------------------------------------------
        # Create the medical_guidelines collection and its indexes
        # -----------------------------------------------------------------
        guidelines_collection_name = "medical_guidelines"
        if guidelines_collection_name not in database.list_collection_names():
            database.create_collection(guidelines_collection_name)
            logger.info(f"Created collection: {guidelines_collection_name}")
        print(f"[SETUP] Collection ready: {guidelines_collection_name}")

        guidelines_collection = database[guidelines_collection_name]

        # Text index enables MongoDB full-text search across guideline content.
        guidelines_collection.create_index(
            [("chunk_content", "text")],
            name="idx_guideline_chunk_content_text",
        )

        # Compound index for filtered retrieval by document and section.
        guidelines_collection.create_index(
            [("source_file", ASCENDING), ("section_heading", ASCENDING)],
            name="idx_guideline_source_section",
        )

        # Unique index on chunk_hash for idempotent ingestion.
        guidelines_collection.create_index(
            [("chunk_hash", ASCENDING)],
            name="idx_guideline_chunk_hash_unique",
            unique=True,
        )

        print(f"[SETUP] Indexes created for: {guidelines_collection_name}")

        # -----------------------------------------------------------------
        # Create the drugs collection and its indexes
        # -----------------------------------------------------------------
        drugs_collection_name = "drugs"
        if drugs_collection_name not in database.list_collection_names():
            database.create_collection(drugs_collection_name)
            logger.info(f"Created collection: {drugs_collection_name}")
        print(f"[SETUP] Collection ready: {drugs_collection_name}")

        drugs_collection = database[drugs_collection_name]

        # Unique index on drug_name for upsert-based deduplication.
        drugs_collection.create_index(
            [("drug_name", ASCENDING)],
            name="idx_drug_name_unique",
            unique=True,
        )

        # Index on drug_class for therapeutic-class-based lookups.
        drugs_collection.create_index(
            [("drug_class", ASCENDING)],
            name="idx_drug_class",
        )

        print(f"[SETUP] Indexes created for: {drugs_collection_name}")

        client.close()
        logger.info("Database initialization complete")
        return True

    except ImportError:
        print("[SETUP] ERROR: pymongo is not installed. Run: pip install pymongo>=4.6.0")
        return False
    except Exception as error:
        logger.error(f"Database initialization failed: {error}")
        print(f"[SETUP] ERROR: Database initialization failed: {error}")
        return False

if __name__ == "__main__":
    from core.config import settings
    
    logging.basicConfig(level=logging.INFO)
    uri = settings.mongodb_settings.mongodb_uri if hasattr(settings, "mongodb_settings") else "mongodb://admin:adminpassword@localhost:27017"
    success = initialize_mongodb_database_and_collections(uri)
    import sys
    sys.exit(0 if success else 1)
