"""
Abstract base class for parent chunk storage.

Parent chunks provide extended context for LLM reasoning after child chunk
retrieval. They are stored separately from the vector store (typically in
MongoDB) as they are not embedded.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..models.chunk import ParentChunk


class AbstractParentChunkStore(ABC):
    """Interface for parent chunk storage implementations."""

    @abstractmethod
    def save_parent_chunk(self, chunk: ParentChunk) -> None:
        """
        Save a parent chunk to the store.
        
        Args:
            chunk: ParentChunk to save
        
        Raises:
            RegistryConnectionError: If connection to store fails
        """
        raise NotImplementedError("Subclasses must implement save_parent_chunk()")

    @abstractmethod
    def get_parent_chunk(self, parent_chunk_id: str) -> Optional[ParentChunk]:
        """
        Retrieve a parent chunk by its ID.
        
        Args:
            parent_chunk_id: SHA-256 hash identifying the parent chunk
        
        Returns:
            ParentChunk if found, None otherwise
        """
        raise NotImplementedError("Subclasses must implement get_parent_chunk()")

    @abstractmethod
    def get_parent_chunks_by_doc_id(self, doc_id: str) -> List[ParentChunk]:
        """
        Retrieve all parent chunks belonging to a specific document.
        
        Args:
            doc_id: Document ID (SHA-256 hash of PDF bytes)
        
        Returns:
            List of ParentChunk objects for the document
        """
        raise NotImplementedError("Subclasses must implement get_parent_chunks_by_doc_id()")
