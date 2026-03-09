"""
Document hashing utility for generating deterministic IDs.

Provides SHA-256 hashing for:
- Document IDs (based on raw PDF bytes)
- Chunk IDs (based on section heading + content + page number)

Pure utility with no external dependencies. All methods are stateless.
"""

import hashlib
from pathlib import Path


class DocumentHasher:
    """
    Generates deterministic SHA-256 hashes for documents and chunks.
    
    No dependencies. All methods are pure functions.
    """

    @staticmethod
    def compute_document_id(pdf_path: Path) -> str:
        """
        Compute a deterministic document ID from PDF file contents.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            SHA-256 hash of the PDF bytes as a hex string
        
        Raises:
            FileNotFoundError: If PDF file does not exist
            IOError: If file cannot be read
        """
        pdf_bytes = pdf_path.read_bytes()
        return hashlib.sha256(pdf_bytes).hexdigest()

    @staticmethod
    def compute_chunk_id(section_heading: str, content: str, page: int) -> str:
        """
        Compute a deterministic chunk ID from its semantic components.
        
        The chunk ID is stable across re-ingestion of the same content,
        enabling idempotent deduplication at the chunk level.
        
        Args:
            section_heading: Section or heading text
            content: Chunk content text
            page: Page number where chunk appears
        
        Returns:
            SHA-256 hash of the combined fingerprint as a hex string
        """
        fingerprint = f"{section_heading}::{content}::{page}"
        return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
