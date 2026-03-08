"""
MongoDB parent chunk store implementation.

Stores parent chunks (large context chunks) separately from vector store.
Parent chunks are retrieved after child chunk matching to provide extended
context to the LLM.
"""

from typing import List, Optional

import structlog
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from ...domain.models.chunk import ParentChunk
from ...domain.ports.chunk_store_port import AbstractParentChunkStore
from ...exceptions.pipeline_exceptions import RegistryConnectionError


logger = structlog.get_logger(__name__)


class MongoParentChunkStore(AbstractParentChunkStore):
    """
    MongoDB-based storage for parent chunks.
    
    Features:
    - Persistent parent chunk storage
    - Efficient retrieval by parent_chunk_id or doc_id
    """

    def __init__(self, mongodb_uri: str, database_name: str, collection_name: str):
        """
        Initialize the MongoDB parent chunk store.
        
        Args:
            mongodb_uri: MongoDB connection URI
            database_name: Database name
            collection_name: Collection name for parent chunks
        """
        try:
            self.client = MongoClient(mongodb_uri)
            self.db = self.client[database_name]
            self.collection = self.db[collection_name]
            
            self.collection.create_index("parent_chunk_id", unique=True)
            self.collection.create_index("doc_id")
            
            logger.info(
                "mongo_parent_chunk_store_initialized",
                database=database_name,
                collection=collection_name,
            )
            
        except ConnectionFailure as e:
            raise RegistryConnectionError(registry_uri=mongodb_uri, cause=e)

    def save_parent_chunk(self, chunk: ParentChunk) -> None:
        """
        Save a parent chunk to MongoDB.
        
        Args:
            chunk: ParentChunk to save
        
        Raises:
            RegistryConnectionError: If connection fails
        """
        try:
            doc = {
                "parent_chunk_id": chunk.parent_chunk_id,
                "doc_id": chunk.doc_id,
                "content": chunk.content,
                "section_heading": chunk.section_heading,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "child_chunk_ids": chunk.child_chunk_ids,
                "token_count": chunk.token_count,
            }
            
            self.collection.replace_one(
                {"parent_chunk_id": chunk.parent_chunk_id},
                doc,
                upsert=True,
            )
            
            logger.debug(
                "parent_chunk_saved",
                parent_chunk_id=chunk.parent_chunk_id,
                doc_id=chunk.doc_id,
            )
            
        except Exception as e:
            logger.error(
                "save_parent_chunk_failed",
                parent_chunk_id=chunk.parent_chunk_id,
                error=str(e),
            )
            raise RegistryConnectionError(registry_uri="mongodb", cause=e)

    def get_parent_chunk(self, parent_chunk_id: str) -> Optional[ParentChunk]:
        """
        Retrieve a parent chunk by ID.
        
        Args:
            parent_chunk_id: Parent chunk ID
        
        Returns:
            ParentChunk if found, None otherwise
        """
        try:
            doc = self.collection.find_one({"parent_chunk_id": parent_chunk_id})
            
            if not doc:
                return None
            
            return ParentChunk(
                parent_chunk_id=doc["parent_chunk_id"],
                doc_id=doc["doc_id"],
                content=doc["content"],
                section_heading=doc["section_heading"],
                page_start=doc["page_start"],
                page_end=doc["page_end"],
                child_chunk_ids=doc.get("child_chunk_ids", []),
                token_count=doc.get("token_count", 0),
            )
            
        except Exception as e:
            logger.error(
                "get_parent_chunk_failed",
                parent_chunk_id=parent_chunk_id,
                error=str(e),
            )
            return None

    def get_parent_chunks_by_doc_id(self, doc_id: str) -> List[ParentChunk]:
        """
        Retrieve all parent chunks for a document.
        
        Args:
            doc_id: Document ID
        
        Returns:
            List of ParentChunk objects
        """
        try:
            docs = self.collection.find({"doc_id": doc_id})
            
            chunks = []
            for doc in docs:
                chunk = ParentChunk(
                    parent_chunk_id=doc["parent_chunk_id"],
                    doc_id=doc["doc_id"],
                    content=doc["content"],
                    section_heading=doc["section_heading"],
                    page_start=doc["page_start"],
                    page_end=doc["page_end"],
                    child_chunk_ids=doc.get("child_chunk_ids", []),
                    token_count=doc.get("token_count", 0),
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(
                "get_parent_chunks_by_doc_id_failed",
                doc_id=doc_id,
                error=str(e),
            )
            return []
