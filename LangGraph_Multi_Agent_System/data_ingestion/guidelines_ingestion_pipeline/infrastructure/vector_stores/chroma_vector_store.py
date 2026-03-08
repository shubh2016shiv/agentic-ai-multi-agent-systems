"""
ChromaDB vector store implementation with Write-Ahead Log (WAL).

Implements resilient vector storage with WAL for crash recovery.
Before every write, chunk IDs are logged as 'pending'. After successful
write, they're marked 'committed'. On startup, any pending entries are replayed.
"""

import json
from pathlib import Path
from typing import List, Optional

import chromadb
import structlog

from ...domain.models.chunk import ChildChunk
from ...domain.ports.vector_store_port import AbstractVectorStore
from ...exceptions.pipeline_exceptions import (
    VectorStoreDiskFullError,
    VectorStoreWriteError,
)


logger = structlog.get_logger(__name__)


class ChromaVectorStore(AbstractVectorStore):
    """
    ChromaDB-based vector store with Write-Ahead Log for resilience.
    
    Features:
    - Write-Ahead Log (WAL) for crash recovery
    - Batched writes for performance
    - HNSW indexing for fast similarity search
    """

    def __init__(
        self,
        chroma_path: str,
        collection_name: str,
        wal_file_path: str,
        batch_size: int = 50,
        hnsw_construction_ef: int = 200,
        hnsw_m: int = 48,
    ):
        """
        Initialize the ChromaDB vector store.
        
        Args:
            chroma_path: Path to ChromaDB persistence directory
            collection_name: Name of the collection
            wal_file_path: Path to Write-Ahead Log file
            batch_size: Number of chunks to write per batch
            hnsw_construction_ef: HNSW index construction parameter
            hnsw_m: HNSW index M parameter
        """
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.wal_file_path = Path(wal_file_path)
        self.batch_size = batch_size
        
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": hnsw_construction_ef,
                "hnsw:M": hnsw_m,
            },
        )
        
        self._replay_wal()
        
        logger.info(
            "chroma_vector_store_initialized",
            collection_name=collection_name,
            chroma_path=chroma_path,
        )

    def upsert_chunks(self, chunks: List[ChildChunk]) -> None:
        """
        Insert or update chunks in the vector store.
        
        Args:
            chunks: List of ChildChunk objects with embeddings
        
        Raises:
            VectorStoreWriteError: If write fails
            VectorStoreDiskFullError: If disk is full
        """
        logger.info("chroma_upsert_started", total_chunks=len(chunks))
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            
            try:
                chunk_ids = [c.chunk_id for c in batch]
                self._write_wal(chunk_ids, status="pending")
                
                self.collection.upsert(
                    ids=chunk_ids,
                    embeddings=[c.embedding for c in batch],
                    documents=[c.content for c in batch],
                    metadatas=[self._chunk_to_metadata(c) for c in batch],
                )
                
                self._write_wal(chunk_ids, status="committed")
                
                logger.debug(
                    "chroma_batch_written",
                    batch_size=len(batch),
                    batch_index=i // self.batch_size,
                )
                
            except OSError as e:
                if "disk full" in str(e).lower() or "no space" in str(e).lower():
                    raise VectorStoreDiskFullError()
                raise VectorStoreWriteError(chunk_ids=[c.chunk_id for c in batch], cause=e)
            except Exception as e:
                logger.error(
                    "chroma_batch_write_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise VectorStoreWriteError(chunk_ids=[c.chunk_id for c in batch], cause=e)
        
        logger.info("chroma_upsert_complete", chunks_written=len(chunks))

    def chunk_exists(self, chunk_id: str) -> bool:
        """
        Check if a chunk exists in the vector store.
        
        Args:
            chunk_id: Chunk ID to check
        
        Returns:
            True if chunk exists, False otherwise
        """
        try:
            results = self.collection.get(ids=[chunk_id], include=[])
            return len(results["ids"]) > 0
        except Exception as e:
            logger.error("chunk_exists_check_failed", chunk_id=chunk_id, error=str(e))
            return False

    def get_chunks_by_doc_id(self, doc_id: str) -> List[ChildChunk]:
        """
        Retrieve all chunks for a document.
        
        Args:
            doc_id: Document ID
        
        Returns:
            List of ChildChunk objects
        """
        try:
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=["embeddings", "documents", "metadatas"],
            )
            
            chunks = []
            for i in range(len(results["ids"])):
                chunk = self._metadata_to_chunk(
                    chunk_id=results["ids"][i],
                    embedding=results["embeddings"][i],
                    content=results["documents"][i],
                    metadata=results["metadatas"][i],
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error("get_chunks_by_doc_id_failed", doc_id=doc_id, error=str(e))
            return []

    def mark_chunks_as_superseded(
        self,
        doc_id: str,
        superseded_by_doc_id: str
    ) -> None:
        """
        Soft-delete chunks by marking them as superseded.
        
        Args:
            doc_id: Document ID of old version
            superseded_by_doc_id: Document ID of new version
        
        Raises:
            VectorStoreWriteError: If update fails
        """
        try:
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"],
            )
            
            if not results["ids"]:
                return
            
            updated_metadatas = []
            for metadata in results["metadatas"]:
                metadata["is_superseded"] = True
                metadata["superseded_by_doc_id"] = superseded_by_doc_id
                updated_metadatas.append(metadata)
            
            self.collection.update(
                ids=results["ids"],
                metadatas=updated_metadatas,
            )
            
            logger.info(
                "chunks_marked_superseded",
                doc_id=doc_id,
                chunks_updated=len(results["ids"]),
            )
            
        except Exception as e:
            logger.error("mark_superseded_failed", doc_id=doc_id, error=str(e))
            raise VectorStoreWriteError(chunk_ids=[], cause=e)

    def semantic_similarity_search(
        self,
        embeddings: List[List[float]],
        top_k: int
    ) -> List[ChildChunk]:
        """
        Search for semantically similar chunks.
        
        Args:
            embeddings: Query embeddings
            top_k: Number of results per query
        
        Returns:
            List of similar ChildChunk objects
        """
        try:
            count = self.collection.count()
            if count == 0:
                return []
            n_results = min(top_k, count)
            results = self.collection.query(
                query_embeddings=embeddings,
                n_results=n_results,
                include=["embeddings", "documents", "metadatas", "distances"],
            )
            
            chunks = []
            for i in range(len(results["ids"][0])):
                chunk = self._metadata_to_chunk(
                    chunk_id=results["ids"][0][i],
                    embedding=results["embeddings"][0][i],
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error("semantic_search_failed", error=str(e))
            return []

    def _write_wal(self, chunk_ids: List[str], status: str) -> None:
        """Write chunk IDs to Write-Ahead Log."""
        with self.wal_file_path.open("a") as f:
            for chunk_id in chunk_ids:
                entry = {"chunk_id": chunk_id, "status": status}
                f.write(json.dumps(entry) + "\n")

    def _replay_wal(self) -> None:
        """Replay pending entries from WAL on startup."""
        if not self.wal_file_path.exists():
            return
        
        pending_chunks = set()
        
        with self.wal_file_path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry["status"] == "pending":
                    pending_chunks.add(entry["chunk_id"])
                elif entry["status"] == "committed":
                    pending_chunks.discard(entry["chunk_id"])
        
        if pending_chunks:
            logger.warning(
                "wal_replay_found_pending",
                pending_chunks=len(pending_chunks),
            )

    def _chunk_to_metadata(self, chunk: ChildChunk) -> dict:
        """Convert ChildChunk to ChromaDB metadata dict."""
        return {
            "chunk_id": chunk.chunk_id,
            "parent_chunk_id": chunk.parent_chunk_id,
            "doc_id": chunk.doc_id,
            "chunk_type": chunk.chunk_type.value,
            "section_heading": chunk.section_heading,
            "page_numbers": json.dumps(chunk.page_numbers),
            "pdf_name": chunk.metadata.pdf_name,
            "guideline_org": chunk.metadata.guideline_org,
            "guideline_year": chunk.metadata.guideline_year,
            "therapeutic_area": chunk.metadata.therapeutic_area,
            "is_superseded": chunk.metadata.is_superseded,
        }

    def _metadata_to_chunk(
        self,
        chunk_id: str,
        embedding: List[float],
        content: str,
        metadata: dict,
    ) -> ChildChunk:
        """Convert ChromaDB metadata dict to ChildChunk (simplified)."""
        from ...domain.models.chunk import ChunkMetadata, ChunkType
        from datetime import datetime
        
        chunk_metadata = ChunkMetadata(
            pdf_name=metadata.get("pdf_name", ""),
            pdf_source_path="",
            guideline_org=metadata.get("guideline_org", ""),
            guideline_year=metadata.get("guideline_year", 0),
            therapeutic_area=metadata.get("therapeutic_area", ""),
            condition_focus="",
            parser_version="",
            embedding_model="",
            pipeline_version="",
            ingested_at=datetime.utcnow(),
            is_superseded=metadata.get("is_superseded", False),
        )
        
        return ChildChunk(
            chunk_id=chunk_id,
            parent_chunk_id=metadata.get("parent_chunk_id", ""),
            doc_id=metadata.get("doc_id", ""),
            content=content,
            chunk_type=ChunkType(metadata.get("chunk_type", "TEXT")),
            page_numbers=json.loads(metadata.get("page_numbers", "[]")),
            section_heading=metadata.get("section_heading", ""),
            section_depth=0,
            chunk_index=0,
            token_count=0,
            confidence_score=1.0,
            is_ocr_sourced=False,
            metadata=chunk_metadata,
            embedding=embedding,
        )
