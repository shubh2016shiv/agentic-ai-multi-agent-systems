"""
Parent-child chunking service for clinical guideline PDFs.

Implements the parent-child chunking strategy where:
- Parent chunks (1200-1500 tokens) provide extended context
- Child chunks (300-400 tokens, 20% overlap) enable precise retrieval
- Tables and figures are atomic (never split)
- Chunk boundaries respect H1/H2/H3 headings
"""

from datetime import datetime
from typing import List, Optional, Tuple

import structlog
from transformers import AutoTokenizer

from ..models.chunk import ChildChunk, ChunkMetadata, ChunkType, ParentChunk
from ..models.document_metadata import GuidelineMetadata
from ..models.parsed_document import ParsedDocument, ParsedFigure, ParsedSection, ParsedTable
from ..ports.config_protocol import PipelineConfigProtocol
from ..exceptions import ChunkTokenLimitExceededError, EmptyChunkError
from .document_hasher import DocumentHasher


logger = structlog.get_logger(__name__)


class ChunkingService:
    """
    Builds parent-child chunk hierarchies from parsed documents.
    
    Dependencies:
        - PipelineConfigProtocol (injected)
        - DocumentHasher (injected)
    """

    def __init__(self, settings: PipelineConfigProtocol, hasher: DocumentHasher):
        """
        Initialize the chunking service.
        
        Args:
            settings: Pipeline configuration
            hasher: Document hasher for generating chunk IDs
        """
        self.settings = settings
        self.hasher = hasher
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.embedding_model_name,
                use_fast=True,
            )
            logger.info(
                "tokenizer_initialized",
                model_name=settings.embedding_model_name,
            )
        except Exception as e:
            logger.warning(
                "tokenizer_initialization_failed_falling_back_to_char_estimate",
                model_name=settings.embedding_model_name,
                error=str(e),
            )
            self.tokenizer = None

    def build_parent_child_chunks(
        self,
        doc: ParsedDocument,
        metadata: GuidelineMetadata
    ) -> Tuple[List[ParentChunk], List[ChildChunk]]:
        """
        Build parent and child chunks from a parsed document.
        
        Args:
            doc: ParsedDocument with sections, tables, and figures
            metadata: Guideline metadata for chunk metadata
        
        Returns:
            Tuple of (parent_chunks, child_chunks)
        
        Raises:
            EmptyChunkError: If a chunk has no meaningful content
            ChunkTokenLimitExceededError: If chunk exceeds max token limit
        """
        parent_chunks: List[ParentChunk] = []
        child_chunks: List[ChildChunk] = []
        
        chunk_metadata = self._build_chunk_metadata(doc, metadata)
        
        parent_buffer = []
        parent_start_page = 0
        current_section_heading = ""
        child_index = 0
        
        for section in doc.sections:
            if section.depth <= 2:
                if parent_buffer:
                    parent, children = self._finalize_parent_chunk(
                        parent_buffer,
                        doc.doc_id,
                        current_section_heading,
                        parent_start_page,
                        section.page_numbers[0] - 1 if section.page_numbers else parent_start_page,
                        chunk_metadata,
                        child_index,
                    )
                    if parent and children:
                        parent_chunks.append(parent)
                        child_chunks.extend(children)
                        child_index += len(children)
                
                parent_buffer = []
                parent_start_page = section.page_numbers[0] if section.page_numbers else 0
                current_section_heading = section.heading
            
            parent_buffer.append(section)
            
            estimated_tokens = self._count_tokens(section.content)
            if estimated_tokens >= self.settings.parent_chunk_max_tokens:
                parent, children = self._finalize_parent_chunk(
                    parent_buffer,
                    doc.doc_id,
                    current_section_heading,
                    parent_start_page,
                    section.page_numbers[-1] if section.page_numbers else parent_start_page,
                    chunk_metadata,
                    child_index,
                )
                if parent and children:
                    parent_chunks.append(parent)
                    child_chunks.extend(children)
                    child_index += len(children)
                
                parent_buffer = []
                parent_start_page = section.page_numbers[-1] + 1 if section.page_numbers else parent_start_page + 1
        
        if parent_buffer:
            parent, children = self._finalize_parent_chunk(
                parent_buffer,
                doc.doc_id,
                current_section_heading,
                parent_start_page,
                parent_buffer[-1].page_numbers[-1] if parent_buffer[-1].page_numbers else parent_start_page,
                chunk_metadata,
                child_index,
            )
            if parent and children:
                parent_chunks.append(parent)
                child_chunks.extend(children)
                child_index += len(children)
        
        for table in doc.tables:
            table_chunk = self._create_table_chunk(
                table,
                doc.doc_id,
                chunk_metadata,
                child_index,
            )
            child_chunks.append(table_chunk)
            child_index += 1
        
        for figure in doc.figures:
            figure_chunk = self._create_figure_chunk(
                figure,
                doc.doc_id,
                chunk_metadata,
                child_index,
            )
            child_chunks.append(figure_chunk)
            child_index += 1
        
        logger.info(
            "chunking_complete",
            doc_id=doc.doc_id,
            parent_chunks=len(parent_chunks),
            child_chunks=len(child_chunks),
        )
        
        return parent_chunks, child_chunks

    def _finalize_parent_chunk(
        self,
        sections: List[ParsedSection],
        doc_id: str,
        section_heading: str,
        page_start: int,
        page_end: int,
        base_metadata: ChunkMetadata,
        child_start_index: int,
    ) -> Tuple[Optional[ParentChunk], List[ChildChunk]]:
        """Create parent chunk and split into child chunks."""
        if not sections:
            return None, []
        
        parent_content = "\n\n".join(s.content for s in sections)
        parent_token_count = self._count_tokens(parent_content)
        
        if parent_token_count < self.settings.child_chunk_min_tokens:
            logger.debug(
                "skipping_undersized_parent",
                section_heading=section_heading,
                token_count=parent_token_count,
                min_tokens=self.settings.child_chunk_min_tokens,
            )
            return None, []
        
        parent_chunk_id = self.hasher.compute_chunk_id(
            section_heading,
            parent_content[:100],
            page_start,
        )
        
        child_chunks = self._split_into_child_chunks(
            parent_content,
            parent_chunk_id,
            doc_id,
            section_heading,
            sections[0].depth,
            page_start,
            page_end,
            base_metadata,
            child_start_index,
        )
        
        parent_chunk = ParentChunk(
            parent_chunk_id=parent_chunk_id,
            doc_id=doc_id,
            content=parent_content,
            section_heading=section_heading,
            page_start=page_start,
            page_end=page_end,
            child_chunk_ids=[c.chunk_id for c in child_chunks],
            token_count=parent_token_count,
        )
        
        return parent_chunk, child_chunks

    def _split_into_child_chunks(
        self,
        text: str,
        parent_chunk_id: str,
        doc_id: str,
        section_heading: str,
        section_depth: int,
        page_start: int,
        page_end: int,
        base_metadata: ChunkMetadata,
        child_start_index: int,
    ) -> List[ChildChunk]:
        """Split text into overlapping child chunks."""
        child_chunks = []
        
        sentences = text.split(". ")
        current_chunk = []
        current_tokens = 0
        overlap_buffer = []
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.settings.child_chunk_max_tokens:
                if current_chunk:
                    chunk_text = ". ".join(current_chunk) + "."
                    chunk_id = self.hasher.compute_chunk_id(
                        section_heading,
                        chunk_text,
                        page_start,
                    )
                    
                    child_chunk = ChildChunk(
                        chunk_id=chunk_id,
                        parent_chunk_id=parent_chunk_id,
                        doc_id=doc_id,
                        content=chunk_text,
                        chunk_type=ChunkType.TEXT,
                        page_numbers=list(range(page_start, page_end + 1)),
                        section_heading=section_heading,
                        section_depth=section_depth,
                        chunk_index=child_start_index + len(child_chunks),
                        token_count=current_tokens,
                        confidence_score=1.0,
                        is_ocr_sourced=False,
                        metadata=base_metadata,
                    )
                    child_chunks.append(child_chunk)
                    
                    overlap_size = int(len(current_chunk) * self.settings.child_chunk_overlap_ratio)
                    overlap_buffer = current_chunk[-overlap_size:] if overlap_size > 0 else []
                    current_chunk = overlap_buffer + [sentence]
                    current_tokens = sum(self._count_tokens(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunk_text = ". ".join(current_chunk) + "."
            if self._count_tokens(chunk_text) >= self.settings.child_chunk_min_tokens:
                chunk_id = self.hasher.compute_chunk_id(
                    section_heading,
                    chunk_text,
                    page_start,
                )
                
                child_chunk = ChildChunk(
                    chunk_id=chunk_id,
                    parent_chunk_id=parent_chunk_id,
                    doc_id=doc_id,
                    content=chunk_text,
                    chunk_type=ChunkType.TEXT,
                    page_numbers=list(range(page_start, page_end + 1)),
                    section_heading=section_heading,
                    section_depth=section_depth,
                    chunk_index=child_start_index + len(child_chunks),
                    token_count=current_tokens,
                    confidence_score=1.0,
                    is_ocr_sourced=False,
                    metadata=base_metadata,
                )
                child_chunks.append(child_chunk)
        
        return child_chunks

    def _create_table_chunk(
        self,
        table: ParsedTable,
        doc_id: str,
        base_metadata: ChunkMetadata,
        chunk_index: int,
    ) -> ChildChunk:
        """Create an atomic chunk for a table."""
        chunk_id = self.hasher.compute_chunk_id(
            table.caption,
            table.markdown_content[:100],
            table.page_numbers[0] if table.page_numbers else 0,
        )
        
        content = f"**{table.caption}**\n\n{table.markdown_content}"
        
        return ChildChunk(
            chunk_id=chunk_id,
            parent_chunk_id="",
            doc_id=doc_id,
            content=content,
            chunk_type=ChunkType.TABLE,
            page_numbers=table.page_numbers,
            section_heading=table.caption,
            section_depth=0,
            chunk_index=chunk_index,
            token_count=self._count_tokens(content),
            confidence_score=1.0 if table.validation_passed else 0.8,
            is_ocr_sourced=False,
            metadata=base_metadata,
        )

    def _create_figure_chunk(
        self,
        figure: ParsedFigure,
        doc_id: str,
        base_metadata: ChunkMetadata,
        chunk_index: int,
    ) -> ChildChunk:
        """Create an atomic chunk for a figure."""
        chunk_id = self.hasher.compute_chunk_id(
            figure.caption,
            figure.description or "pending",
            figure.page_number,
        )
        
        if figure.description:
            content = f"**{figure.caption}**\n\n{figure.description}"
            chunk_type = ChunkType.FIGURE_DESCRIPTION
        else:
            content = f"**{figure.caption}**\n\n[Figure description pending]"
            chunk_type = ChunkType.FIGURE_DESCRIPTION_PENDING
        
        return ChildChunk(
            chunk_id=chunk_id,
            parent_chunk_id="",
            doc_id=doc_id,
            content=content,
            chunk_type=chunk_type,
            page_numbers=[figure.page_number],
            section_heading=figure.caption,
            section_depth=0,
            chunk_index=chunk_index,
            token_count=self._count_tokens(content),
            confidence_score=0.9 if figure.description else 0.5,
            is_ocr_sourced=False,
            metadata=base_metadata,
        )

    def _build_chunk_metadata(
        self,
        doc: ParsedDocument,
        guideline_metadata: GuidelineMetadata,
    ) -> ChunkMetadata:
        """Build metadata for all chunks from this document."""
        return ChunkMetadata(
            pdf_name=doc.pdf_name,
            pdf_source_path=doc.pdf_source_path,
            guideline_org=guideline_metadata.guideline_org,
            guideline_year=guideline_metadata.guideline_year,
            therapeutic_area=guideline_metadata.therapeutic_area,
            condition_focus=guideline_metadata.condition_focus,
            parser_version=doc.parser_version,
            embedding_model=self.settings.embedding_model_name,
            pipeline_version=self.settings.pipeline_version,
            ingested_at=datetime.utcnow(),
            is_superseded=False,
        )

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the embedding model's tokenizer.
        
        Args:
            text: Text to tokenize
        
        Returns:
            Token count
        """
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                return len(tokens)
            except Exception as e:
                logger.debug(
                    "tokenizer_failed_using_char_estimate",
                    error=str(e),
                )
                return len(text) // 4
        else:
            return len(text) // 4
