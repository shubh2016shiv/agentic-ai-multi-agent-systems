"""
Long-Term Memory (RAG via ChromaDB + Provider Embeddings)
==========================================================
Persistent memory backed by a ChromaDB vector store. Used for:
    - Medical guideline retrieval (RAG)
    - Drug information lookup
    - Past case similarity search

This implements the long-term memory layer from the MAS reference
architecture (Chapter 2). Documents are embedded and stored in ChromaDB,
then retrieved based on semantic similarity when agents need context.

Where This Fits in the MAS Architecture
-----------------------------------------
Position in the multi-tier memory stack:

    Working Memory      — per-execution scratchpad (working_memory.py)
    Conversation Memory — per-session dialogue history (conversation_memory.py)
    Long-Term Memory    — persistent knowledge base (THIS FILE)
         ↕ ChromaDB vector store (survives process restarts)

Retrieval-Augmented Generation (RAG) pattern:
    1. [Index]   : Ingest medical guidelines/drug info → embed → store
    2. [Retrieve]: Agent has a patient case → query for relevant context
    3. [Augment] : Prepend retrieved context to the agent's prompt
    4. [Generate]: LLM produces guideline-grounded response

Why long-term memory matters for multi-agent systems:
    Without it, each agent call starts from zero knowledge. The LLM
    either hallucinates guidelines or can only use what was in its
    training data. With ChromaDB, agents can query an always-current,
    domain-specific knowledge base that is independent of the LLM.

Pattern script:
    scripts/memory_management/semantic_retrieval.py  — Pattern 3

WHY WE DON'T USE CHROMADB'S DEFAULT EMBEDDING FUNCTION:
    ChromaDB's out-of-the-box embedding function downloads and runs the
    all-MiniLM-L6-v2 ONNX model (79MB) locally. This means:
        - A surprise 79MB download on first run
        - CPU inference on Windows (slow)
        - A second ML model dependency to manage

    Instead, we use langchain_chroma.Chroma with an explicit embedding
    function obtained from get_embeddings() — the same abstraction used
    for chat models. This means:
        - Embeddings go through the same provider (Gemini / OpenAI / LM Studio)
          that is already configured in .env
        - No additional model downloads
        - Consistent provider selection across the entire codebase

Usage:
    from memory.long_term_memory import LongTermMemory

    ltm = LongTermMemory(collection_name="medical_guidelines")
    ltm.add_documents(chunks)                    # Index documents
    results = ltm.search("COPD treatment", k=3)  # Retrieve relevant context
"""

import os
import logging
from typing import Any

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    ChromaDB-backed vector store for persistent medical knowledge retrieval.

    Encapsulates the complexity of embedding, indexing, and retrieval
    behind a simple API. Agents call .search() to get relevant context
    without needing to understand the underlying vector store.

    Embeddings are produced by the same provider configured in .env
    (Gemini / OpenAI / LM Studio) instead of ChromaDB's built-in ONNX
    model, keeping all ML inference in a single, configured place.

    Args:
        collection_name: Name of the ChromaDB collection.
        persist_directory: Directory for ChromaDB persistence.
            Defaults to <project_root>/chroma_db/
    """

    def __init__(
        self,
        collection_name: str = "medical_knowledge",
        persist_directory: str | None = None,
    ):
        self.collection_name = collection_name
        self._persist_dir = persist_directory or self._default_persist_dir()
        self._vectorstore = None  # langchain_chroma.Chroma instance

    def _default_persist_dir(self) -> str:
        """Get default ChromaDB persistence directory."""
        from core.config import settings
        return os.path.join(settings.project_root, "chroma_db")

    def _get_vectorstore(self):
        """
        Lazy-initialize the LangChain Chroma vector store.

        Uses get_embeddings() from core.config to obtain the provider's
        embedding model — no ONNX download, no extra configuration.
        """
        if self._vectorstore is not None:
            return self._vectorstore

        try:
            from langchain_chroma import Chroma
            from core.config import get_embeddings

            embedding_fn = get_embeddings()

            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=embedding_fn,
                persist_directory=self._persist_dir,
                collection_metadata={"description": "Medical knowledge base for CDSS agents"},
            )

            count = self._vectorstore._collection.count()
            logger.info(
                f"ChromaDB collection '{self.collection_name}' initialized "
                f"({count} documents) using provider embeddings"
            )
            return self._vectorstore

        except ImportError as e:
            logger.error(f"Missing dependency: {e}. Run: pip install langchain-chroma chromadb")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        """
        Add documents to the vector store.

        Documents are embedded using the provider's embedding model
        (not ChromaDB's ONNX model).

        Args:
            documents: List of text strings to embed and store.
            metadatas: Optional metadata for each document (source, section, etc.).
            ids: Optional unique IDs. Auto-generated from content hash if not provided.

        Returns:
            Number of documents added.
        """
        if ids is None:
            import hashlib
            ids = [
                hashlib.md5(doc.encode()).hexdigest()[:16]
                for doc in documents
            ]

        if metadatas is None:
            metadatas = [{}] * len(documents)

        vectorstore = self._get_vectorstore()
        vectorstore.add_texts(texts=documents, metadatas=metadatas, ids=ids)

        logger.info(f"Added {len(documents)} documents to '{self.collection_name}'")
        return len(documents)

    def search(
        self,
        query: str,
        k: int = 3,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for documents relevant to a query using semantic similarity.

        Args:
            query: Search query text.
            k: Number of results to return.
            where: Optional metadata filter
                   (e.g., {"condition": "COPD"}).

        Returns:
            List of dicts with keys: "content", "metadata", "distance"
        """
        vectorstore = self._get_vectorstore()

        try:
            # similarity_search_with_relevance_scores returns (Document, score)
            # where score is 0–1 (higher = more relevant); we convert to distance.
            results_with_scores = vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=where,
            )
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []

        formatted = []
        for doc, score in results_with_scores:
            formatted.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "distance": round(1.0 - score, 4),  # convert similarity → distance
            })

        logger.debug(f"Search '{query[:50]}...' returned {len(formatted)} results")
        return formatted

    def get_document_count(self) -> int:
        """Get the total number of documents in the collection."""
        vectorstore = self._get_vectorstore()
        return vectorstore._collection.count()

    def clear(self) -> None:
        """Delete all documents from the collection."""
        if self._vectorstore is not None:
            try:
                self._vectorstore.delete_collection()
                self._vectorstore = None
                logger.info(f"Cleared collection '{self.collection_name}'")
            except Exception as e:
                logger.error(f"Failed to clear collection: {e}")
