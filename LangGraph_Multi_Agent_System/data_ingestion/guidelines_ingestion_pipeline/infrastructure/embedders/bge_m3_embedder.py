"""
BGE-M3 embedding model implementation.

Uses BAAI/bge-m3 for generating dense embeddings of text chunks.
Self-hosted implementation for production use.
"""

from typing import List

import structlog

from ...domain.ports.text_embedder_port import AbstractTextEmbedder
from ...exceptions.pipeline_exceptions import (
    EmbeddingBatchFailureError,
    EmbeddingModelUnavailableError,
    EmbeddingTokenOverflowError,
)
from ...utils.retry_utils import retry_with_exponential_backoff


logger = structlog.get_logger(__name__)


class BGEM3Embedder(AbstractTextEmbedder):
    """
    BAAI/bge-m3 embedding model implementation.
    
    Features:
    - 8192 token context window
    - Dense + sparse + ColBERT multi-granularity retrieval
    - Self-hosted via HuggingFace transformers
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda"):
        """
        Initialize the BGE-M3 embedder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cuda' or 'cpu')
        
        Raises:
            EmbeddingModelUnavailableError: If model fails to load
        """
        self.model_name = model_name
        self.device = device
        self.max_token_limit = 8192
        
        try:
            from FlagEmbedding import BGEM3FlagModel
            from transformers import AutoTokenizer
            
            logger.info(
                "bge_m3_embedder_initializing",
                model_name=model_name,
                device=device,
            )
            
            self.model = BGEM3FlagModel(
                model_name,
                use_fp16=(device == "cuda"),
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
            )
            
            logger.info(
                "bge_m3_embedder_initialized",
                model_name=model_name,
                device=device,
            )
            
        except ImportError as e:
            logger.error(
                "bge_m3_dependencies_missing",
                error=str(e),
                model_name=model_name,
            )
            raise EmbeddingModelUnavailableError(model_name=model_name, cause=e) from e
        except Exception as e:
            logger.error(
                "bge_m3_model_load_failed",
                error=str(e),
                error_type=type(e).__name__,
                model_name=model_name,
            )
            raise EmbeddingModelUnavailableError(model_name=model_name, cause=e) from e

    @retry_with_exponential_backoff(
        max_attempts=3,
        retryable_exceptions=(ConnectionError, TimeoutError, OSError),
    )
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors
        
        Raises:
            EmbeddingModelUnavailableError: If model fails to load
            EmbeddingBatchFailureError: If embedding fails
            EmbeddingTokenOverflowError: If text exceeds token limit
        """
        logger.info("embedding_batch_started", batch_size=len(texts))
        
        for i, text in enumerate(texts):
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                token_count = len(tokens)
                
                if token_count > self.max_token_limit:
                    logger.error(
                        "embedding_token_overflow",
                        chunk_index=i,
                        token_count=token_count,
                        max_tokens=self.max_token_limit,
                    )
                    raise EmbeddingTokenOverflowError(
                        chunk_id=f"chunk_{i}",
                        token_count=token_count,
                        model_limit=self.max_token_limit,
                    )
            except EmbeddingTokenOverflowError:
                raise
            except Exception as e:
                logger.warning(
                    "tokenization_failed_skipping_overflow_check",
                    chunk_index=i,
                    error=str(e),
                )
        
        try:
            embeddings_dict = self.model.encode(
                texts,
                batch_size=32,
                max_length=self.max_token_limit,
            )
            
            embeddings = embeddings_dict['dense_vecs'].tolist()
            
            logger.info(
                "embedding_batch_complete",
                embeddings_generated=len(embeddings),
                embedding_dim=len(embeddings[0]) if embeddings else 0,
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(
                "embedding_batch_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise EmbeddingBatchFailureError(list(range(len(texts))), e) from e

    def get_model_name(self) -> str:
        """
        Get the model name.
        
        Returns:
            Model identifier
        """
        return self.model_name

    def get_max_token_limit(self) -> int:
        """
        Get the maximum token limit.
        
        Returns:
            Maximum tokens
        """
        return self.max_token_limit
