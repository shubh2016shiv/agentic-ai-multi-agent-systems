"""
Abstract base class for vector store operations.

Defines the contract for vector database implementations (ChromaDB, Pinecone,
Weaviate, etc.) that store and retrieve embedded child chunks.
"""

from abc import ABC, abstractmethod
from typing import List

from ..models.chunk import ChildChunk


class AbstractVectorStore(ABC):
    """Interface for vector store implementations."""

    @abstractmethod
    def upsert_chunks(self, chunks: List[ChildChunk]) -> None:
        """
        Insert or update chunks in the vector store.
        
        Args:
            chunks: List of ChildChunk objects with embeddings populated
        
        Raises:
            VectorStoreWriteError: If write operation fails
            VectorStoreDiskFullError: If storage is exhausted
        """
        raise NotImplementedError("Subclasses must implement upsert_chunks()")

    @abstractmethod
    def chunk_exists(self, chunk_id: str) -> bool:
        """
        Check if a chunk with the given ID already exists in the vector store.
        
        Args:
            chunk_id: SHA-256 hash of the chunk content
        
        Returns:
            True if the chunk exists, False otherwise
        """
        raise NotImplementedError("Subclasses must implement chunk_exists()")

    @abstractmethod
    def get_chunks_by_doc_id(self, doc_id: str) -> List[ChildChunk]:
        """
        Retrieve all chunks belonging to a specific document.
        
        Args:
            doc_id: Document ID (SHA-256 hash of PDF bytes)
        
        Returns:
            List of ChildChunk objects for the document
        """
        raise NotImplementedError("Subclasses must implement get_chunks_by_doc_id()")

    @abstractmethod
    def mark_chunks_as_superseded(
        self,
        doc_id: str,
        superseded_by_doc_id: str
    ) -> None:
        """
        Soft-delete chunks from an old document version by marking them superseded.
        
        Args:
            doc_id: Document ID of the old version
            superseded_by_doc_id: Document ID of the new version
        
        Raises:
            VectorStoreWriteError: If update fails
        """
        raise NotImplementedError("Subclasses must implement mark_chunks_as_superseded()")

    @abstractmethod
    def semantic_similarity_search(
        self,
        embeddings: List[List[float]],
        top_k: int
    ) -> List[ChildChunk]:
        """
        Search for semantically similar chunks using cosine similarity.
        
        Args:
            embeddings: Query embedding vectors
            top_k: Number of top results to return per query
        
        Returns:
            List of the most similar ChildChunk objects
        """
        raise NotImplementedError("Subclasses must implement semantic_similarity_search()")
