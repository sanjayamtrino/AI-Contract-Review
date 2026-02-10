import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from docx.document import Document
from numpy import dot
from numpy.linalg import norm

from src.config.logging import Logger
from src.config.settings import get_settings
from src.exceptions.parser_exceptions import (
    DocxCleaningException,
    DocxMetadataExtractionException,
    DocxParagraphExtractionException,
    DocxTableExtractionException,
)
from src.schemas.registry import Chunk, ParseResult
from src.services.registry.base_parser import BaseParser
from src.services.session_manager import SessionData
from src.services.vector_store.embeddings.embedding_service import BGEEmbeddingService
from src.services.vector_store.manager import get_faiss_vector_store, index_chunks

_SECTION_LABEL_RE = re.compile(
    r"^("
    r"\d+[\.\)]?\s+\S.*|"
    r"\d+[\.\)]?\s*$|"  # "1."  /  "2)"
    r"[A-Z][A-Z\s\.\,\&\'\-]{1,60}$"
    r")"
)


class DocxParser(BaseParser, Logger):
    """Parser for DOCX with semantic chunking."""

    # Paragraphs shorter than this word count are candidates for the
    # structural-heading heuristic (regardless of .docx style).
    _HEADING_MAX_WORDS: int = 8

    # Chunks with fewer words than this are considered orphans and will be
    # merged into a neighbouring chunk in the post-pass.
    _ORPHAN_MIN_WORDS: int = 5

    def __init__(self) -> None:
        """Initialize the parser."""
        super().__init__()
        self.settings = get_settings()
        self.embedding_service = BGEEmbeddingService()
        self.vector_store = get_faiss_vector_store(self.embedding_service.get_embedding_dimensions())

    @staticmethod
    def _is_structural_heading(text: str, max_words: int = 8) -> bool:
        """Return True when *text* looks like a title or section header even
        though the .docx author did not apply a Heading style.

        ALL must pass:
            1. Word count <= max_words   — real headings are short.
            2. Matches _SECTION_LABEL_RE — all-caps title".

        """
        words = text.split()
        if len(words) > max_words or len(words) == 0:
            return False
        return bool(_SECTION_LABEL_RE.match(text.strip()))

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Returns the similarity score between two vectors."""

        return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    @staticmethod
    def _merge_orphan_chunks(chunks: List[str], min_words: int = 5) -> List[str]:
        """Merge any chunk that is too short to stand alone into its neighbour."""

        if not chunks:
            return chunks

        merged: List[str] = []
        carry: str = ""

        for chunk in chunks:
            # Prepend any carried-over fragment
            if carry:
                chunk = carry + " " + chunk
                carry = ""

            if len(chunk.split()) < min_words:
                # Too short to emit — carry forward
                carry = chunk
            else:
                merged.append(chunk)

        # If something is still carried it means the very last chunk was short; append it to the previous chunk.
        if carry:
            if merged:
                merged[-1] = merged[-1] + " " + carry
            else:
                merged.append(carry)

        return merged

    def _clean_text(self, text: str) -> str:
        """Clean the text to remove unwanted characters."""

        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace("\u00a0", " ")
        text = text.replace("\u200b", "")
        text = text.replace("\ufeff", "")
        text = text.replace("\r", "")
        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
        return text.lstrip(" .\n\t")

    async def clean_document(self, document: Document) -> None:
        """Clean the document by removing trailing spaces and extra chars."""

        try:
            for paragraph in document.paragraphs:
                if paragraph.text:
                    paragraph.text = " ".join(paragraph.text.split())
        except Exception as e:
            raise DocxCleaningException(str(e)) from e

    async def _extract_metadata(self, document: Document) -> Dict[str, Any]:
        """Extract document metadata."""

        try:
            properties = document.core_properties
            paragraph_words = sum(len(p.text.split()) for p in document.paragraphs)
            table_words = sum(len(cell.text.split()) for table in document.tables for row in table.rows for cell in row.cells)

            return {
                "source": "docx",
                "author": properties.author or "Unknown",
                "title": properties.title or "Untitled",
                "created_at": properties.created.isoformat() if properties.created else None,
                "modified_at": properties.modified.isoformat() if properties.modified else None,
                "paragraph_count": len(document.paragraphs),
                "table_count": len(document.tables),
                "word_count": paragraph_words + table_words,
            }

        except Exception as e:
            raise DocxMetadataExtractionException(str(e)) from e

    async def _extract_tables(self, document: Document) -> List[Dict[str, Any]]:
        try:
            tables = []
            for t_idx, table in enumerate(document.tables):
                rows = [[self._clean_text(cell.text) for cell in row.cells] for row in table.rows]
                tables.append(
                    {
                        "table_index": t_idx,
                        "content": rows,
                    }
                )
            return tables
        except Exception as e:
            raise DocxTableExtractionException(str(e)) from e

    async def _extract_paragraphs(self, document: Document) -> List[Dict[str, Any]]:
        try:
            data = []
            for idx, p in enumerate(document.paragraphs):
                if p.text.strip():
                    cleaned = self._clean_text(p.text)
                    if cleaned:
                        # A paragraph is a heading if the .docx style says so
                        # OR if it matches our structural-heading heuristic.
                        is_heading = bool((p.style and p.style.name.startswith("Heading")) or self._is_structural_heading(cleaned, self._HEADING_MAX_WORDS))
                        data.append(
                            {
                                "index": idx,
                                "content": cleaned,
                                "is_heading": is_heading,
                            }
                        )
            return data
        except Exception as e:
            raise DocxParagraphExtractionException(str(e)) from e

    async def _semantic_chunk_paragraphs(self, paragraphs: List[Dict[str, Any]]) -> List[str]:
        """Do the semantic chunking for the paragraphs to be context aware."""

        texts = [para["content"] for para in paragraphs]

        # Embedd the paragraphs
        embeddings = [await self.embedding_service.generate_embeddings(text=t, task="text-matching") for t in texts]

        # Compute pairwise cosine similarities between consecutice paragraphs
        similarities = [self.cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]

        # Determine the similarity threshold
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)

        threshold = mean_sim - 0.75 * std_sim

        split_points = {i for i, sim in enumerate(similarities) if sim < threshold}

        chunks: List[str] = []
        current: List[str] = []
        current_len: int = 0
        max_len: int = self.settings.chunk_size

        def _flush() -> None:
            """Flush the current buffer into chunks (no-op when buffer is empty)."""
            nonlocal current, current_len
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0

        for i, para in enumerate(paragraphs):
            text = para["content"]
            text_len = len(text)

            # --- PRE-APPEND flushes ---
            if para["is_heading"]:
                _flush()
            elif current_len + text_len > max_len:
                _flush()

            # Append paragraph to the current buffer
            current.append(text)
            current_len += text_len

            # --- POST-APPEND flush ---
            if i in split_points:
                _flush()

        # Flush any remaining paragraphs
        _flush()

        chunks = self._merge_orphan_chunks(chunks, min_words=self._ORPHAN_MIN_WORDS)

        return chunks

    async def parse(self, document: Document, session_data: Optional["SessionData"] = None) -> ParseResult:
        start = time.time()

        try:
            await self.clean_document(document)
            metadata = await self._extract_metadata(document)

            # assign a unique id for this document
            document_id = str(uuid.uuid4())
            metadata["document_id"] = document_id
            paragraphs = await self._extract_paragraphs(document)
            tables = await self._extract_tables(document)

            # Determine which vector store to use
            vector_store = session_data.vector_store if session_data else self.vector_store

            semantic_chunks = await self._semantic_chunk_paragraphs(paragraphs)

            chunks: List[Chunk] = []
            chunk_index = 0

            # Text chunks
            for text in semantic_chunks:
                cleaned = self._clean_text(text)
                if not cleaned:
                    continue

                vector = await self.embedding_service.generate_embeddings(text=cleaned, task="text-matching")
                await vector_store.index_embedding(vector)

                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        chunk_index=chunk_index,
                        content=cleaned,
                        embedding_model=self.embedding_service.model_name,
                        embedding_vector=None,  # vector,
                        metadata={"chunk_type": "semantic_paragraph"},
                        created_at=datetime.utcnow().isoformat(),
                    )
                )
                chunk_index += 1

            # Table chunks
            for table in tables:
                rows = [" | ".join(r) for r in table["content"]]
                table_text = self._clean_text(" ".join(rows))
                if not table_text:
                    continue

                vector = await self.embedding_service.generate_embeddings(text=table_text)
                await vector_store.index_embedding(vector)

                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        chunk_index=chunk_index,
                        content=table_text,
                        embedding_model=self.embedding_service.model_name,
                        embedding_vector=None,  # vector,
                        metadata={
                            "chunk_type": "table",
                            "table_index": table["table_index"],
                            "row_count": len(table["content"]),
                        },
                        created_at=datetime.utcnow().isoformat(),
                    )
                )
                chunk_index += 1

            if session_data:
                from src.services.vector_store.manager import index_chunks_in_session

                index_chunks_in_session(session_data, chunks, metadata)
            else:
                index_chunks(chunks)

            return ParseResult(
                success=True,
                chunks=chunks,
                metadata=metadata,
                error_message=None,
                processing_time=time.time() - start,
            )

        except Exception as e:
            self.logger.error(str(e))
            return ParseResult(
                success=False,
                chunks=[],
                metadata={},
                error_message=str(e),
                processing_time=0.0,
            )

    def is_healthy(self) -> Any:
        """Get the health status of the parser."""
        pass
