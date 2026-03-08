"""
ChromaDB Connection Manager
=============================
Centralised connection lifecycle management for ChromaDB.

Why a Shared Connection Manager?
    Without a manager, each ``ChromaVectorStore`` instance creates its own
    ``chromadb.PersistentClient`` directly in ``__init__``.  This means:
      - Multiple file-lock conflicts if two components open the same
        persistence directory simultaneously
      - Hardcoded ``"./chroma_db"`` path scattered across call-sites
      - No single place to set HNSW construction parameters consistently

    This manager solves all three by:
      1. Encapsulating the ``PersistentClient`` construction (DIP — the
         vector store depends on this abstraction, not on chromadb directly)
      2. Exposing ``get_or_create_collection()`` with HNSW defaults baked in
      3. Supporting the context-manager protocol for guaranteed cleanup

Usage:
    from data_ingestion.connections.chroma_connection_manager import (
        ChromaConnectionManager,
    )

    mgr = ChromaConnectionManager(
        persist_path="./chroma_db",
        hnsw_construction_ef=200,
        hnsw_m=48,
    )
    collection = mgr.get_or_create_collection("medical_guidelines_v1")
    mgr.close()

    # or as a context manager:
    with ChromaConnectionManager(persist_path="./chroma_db") as mgr:
        collection = mgr.get_or_create_collection("medical_guidelines_v1")
"""

from __future__ import annotations

import logging
from typing import Optional

import chromadb
from chromadb import Collection

logger = logging.getLogger(__name__)

_DEFAULT_HNSW_CONSTRUCTION_EF = 200
_DEFAULT_HNSW_M = 48


class ChromaConnectionManager:
    """
    Centralised ChromaDB connection lifecycle manager.

    Creates a single ``chromadb.PersistentClient`` pointed at
    ``persist_path``.  Infrastructure components (e.g. ``ChromaVectorStore``)
    receive this manager via constructor injection and call
    ``get_or_create_collection()`` rather than instantiating
    ``chromadb.PersistentClient`` themselves.

    Args:
        persist_path: File-system path for ChromaDB's on-disk storage.
        hnsw_construction_ef: HNSW ef_construction parameter.  Higher values
            improve recall at the cost of slower index build time.
        hnsw_m: HNSW M parameter (number of bi-directional links per node).
            Higher values improve recall but increase memory usage.
    """

    def __init__(
        self,
        persist_path: str = "./chroma_db",
        hnsw_construction_ef: int = _DEFAULT_HNSW_CONSTRUCTION_EF,
        hnsw_m: int = _DEFAULT_HNSW_M,
    ) -> None:
        self._persist_path = persist_path
        self._hnsw_construction_ef = hnsw_construction_ef
        self._hnsw_m = hnsw_m
        self._client: Optional[chromadb.PersistentClient] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_create_collection(self, collection_name: str) -> Collection:
        """
        Return an existing ChromaDB collection or create it with the
        configured HNSW parameters.

        Args:
            collection_name: Name of the ChromaDB collection.

        Returns:
            A ``chromadb.Collection`` object ready for use.
        """
        collection = self._get_client().get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": self._hnsw_construction_ef,
                "hnsw:M": self._hnsw_m,
            },
        )
        logger.debug(
            "chroma_collection_ready",
            collection=collection_name,
            persist_path=self._persist_path,
        )
        return collection

    def close(self) -> None:
        """
        Release the underlying ChromaDB client.

        ChromaDB's ``PersistentClient`` does not expose an explicit
        ``close()`` API, so this method dereferences the client object to
        allow garbage collection.  Safe to call multiple times.
        """
        if self._client is not None:
            self._client = None
            logger.debug(
                "chroma_connection_released",
                persist_path=self._persist_path,
            )

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "ChromaConnectionManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> chromadb.PersistentClient:
        """Lazily instantiate and cache the ChromaDB client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self._persist_path)
            logger.debug(
                "chroma_client_created",
                persist_path=self._persist_path,
            )
        return self._client
