"""
Abstract base class for text embedding.

Defines the contract for embedding model implementations (BAAI/bge-m3,
OpenAI embeddings, etc.) that convert text chunks into vector representations.
"""

from abc import ABC, abstractmethod
from typing import List


class AbstractTextEmbedder(ABC):
    """Interface for text embedding implementations."""

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of text chunks.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors, one per input text. Each embedding is a
            list of floats with dimensionality determined by the model.
        
        Raises:
            EmbeddingModelUnavailableError: If the embedding model is down
            EmbeddingBatchFailureError: If specific chunks fail to embed
            EmbeddingTokenOverflowError: If a chunk exceeds model's token limit
        """
        raise NotImplementedError("Subclasses must implement embed_batch()")

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name/identifier of the embedding model.
        
        Returns:
            Model name (e.g., "BAAI/bge-m3", "text-embedding-3-large")
        """
        raise NotImplementedError("Subclasses must implement get_model_name()")

    @abstractmethod
    def get_max_token_limit(self) -> int:
        """
        Get the maximum token limit for this embedding model.
        
        Returns:
            Maximum number of tokens the model can process per input
        """
        raise NotImplementedError("Subclasses must implement get_max_token_limit()")
