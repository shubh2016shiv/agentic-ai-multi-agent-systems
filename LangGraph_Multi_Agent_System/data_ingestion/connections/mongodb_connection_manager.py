"""
MongoDB Connection Manager
============================
Centralised connection management for all MongoDB-backed pipelines.

Why a Shared Connection Manager?
    Both the drug-ingestion pipeline and the guidelines-ingestion pipeline
    need MongoDB access.  Without a shared manager each pipeline would
    create its own ``MongoClient``, leading to:
      - Connection-pool exhaustion under load
      - Inconsistent connection configuration
      - Leaked connections when errors occur

    This manager solves all three by:
      1. Exposing a stable constructor that pipelines depend on (DIP)
      2. Reading configuration from the centralized ``core.config.settings``
         when no explicit URI is provided
      3. Supporting the context-manager protocol for guaranteed cleanup

Usage (drug pipeline — reads URI from settings):
    from data_ingestion.connections.mongodb_connection_manager import (
        MongoDBConnectionManager,
    )

    with MongoDBConnectionManager(server_selection_timeout_ms=5_000) as mgr:
        if not mgr.verify_connection_health():
            raise RuntimeError("MongoDB unreachable")
        collection = mgr.get_collection("drugs")

Usage (guidelines pipeline — explicit URI):
    mgr = MongoDBConnectionManager(
        mongodb_uri="mongodb://admin:pass@localhost:27017",
        database_name="guidelines_pipeline",
    )
    collection = mgr.get_collection("ingestion_jobs")
    mgr.close()
"""

from __future__ import annotations

import logging
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_MS = 5_000


class MongoDBConnectionManager:
    """
    Centralised MongoDB connection lifecycle manager.

    Supports two construction styles so that it can serve both the drug
    pipeline (which reads config from ``core.config.settings``) and the
    guidelines pipeline (which supplies an explicit URI from its own
    ``PipelineSettings``):

    Style 1 — reads from core settings:
        MongoDBConnectionManager(server_selection_timeout_ms=5_000)

    Style 2 — explicit URI and database:
        MongoDBConnectionManager(
            mongodb_uri="mongodb://...",
            database_name="my_db",
        )

    Both styles support the context-manager protocol (``with`` statement)
    and expose ``get_collection()``, ``get_database()``,
    ``verify_connection_health()``, and ``close()``.
    """

    def __init__(
        self,
        mongodb_uri: Optional[str] = None,
        database_name: Optional[str] = None,
        server_selection_timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ) -> None:
        """
        Initialise the manager.  The MongoClient is created lazily on first
        access via ``get_database()`` or ``get_collection()``.

        Args:
            mongodb_uri: MongoDB connection URI.  Falls back to
                ``core.config.settings.mongodb_uri`` when not supplied.
            database_name: Target database name.  Falls back to
                ``core.config.settings.mongodb_database_name`` when not
                supplied.
            server_selection_timeout_ms: Milliseconds to wait for a server
                selection before raising ``ServerSelectionTimeoutError``.
        """
        if mongodb_uri is None or database_name is None:
            from core.config import settings as _settings

            mongodb_uri = mongodb_uri or _settings.mongodb_uri
            database_name = database_name or _settings.mongodb_database_name

        self._mongodb_uri = mongodb_uri
        self._database_name = database_name
        self._timeout_ms = server_selection_timeout_ms
        self._client: Optional[MongoClient] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_database(self) -> Database:
        """Return (and lazily create) the configured database handle."""
        return self._get_client()[self._database_name]

    def get_collection(self, collection_name: str) -> Collection:
        """
        Return a collection from the configured database.

        Args:
            collection_name: Name of the MongoDB collection.

        Returns:
            A ``pymongo.Collection`` object.
        """
        return self.get_database()[collection_name]

    def verify_connection_health(self) -> bool:
        """
        Ping the MongoDB server to confirm reachability.

        Returns:
            ``True`` if the server is reachable, ``False`` otherwise.
        """
        try:
            self._get_client().admin.command("ping")
            logger.debug("mongodb_health_check_passed", uri=self._mongodb_uri)
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
            logger.warning(
                "mongodb_health_check_failed",
                uri=self._mongodb_uri,
                error=str(exc),
            )
            return False

    def close(self) -> None:
        """
        Close the underlying MongoClient and release connection-pool resources.

        Safe to call multiple times.
        """
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.debug("mongodb_connection_closed", uri=self._mongodb_uri)

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "MongoDBConnectionManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> MongoClient:
        """Lazily instantiate and cache the MongoClient."""
        if self._client is None:
            self._client = MongoClient(
                self._mongodb_uri,
                serverSelectionTimeoutMS=self._timeout_ms,
            )
            logger.debug(
                "mongodb_client_created",
                uri=self._mongodb_uri,
                database=self._database_name,
            )
        return self._client
