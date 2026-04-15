"""
Semantic DOCX parser — chunks documents by semantic similarity between paragraphs.

Unlike fixed-size chunking, this parser:
  - Detects heading boundaries (both styled and structural)
  - Splits paragraphs containing multiple inline clause headings
  - Groups consecutive paragraphs by embedding similarity
  - Merges orphan (too-short) chunks into neighbours
  - Preserves clause numbering prefixes for cross-referencing
"""

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
    EmptyTextException,
)
from src.schemas.registry import Chunk, ParseResult
from src.schemas.rule_check import TextInfo
from src.services.registry.base_parser import BaseParser
from src.services.session_manager import SessionData
from src.services.vector_store.embeddings.base_embedding_service import BaseEmbeddingService
from src.services.vector_store.manager import get_faiss_vector_store

# Regex: numbered section labels or ALL-CAPS titles
_SECTION_LABEL_RE = re.compile(
    r"^("
    r"\d+[\.\)]?\s+\S.*|"
    r"\d+[\.\)]?\s*$|"
    r"[A-Z][A-Z\s\.\,\&\'\-]{1,60}$"
    r")"
)

# Regex: multi-word inline clause headings like "Audit Rights. " or "No Warranty. "
_INLINE_CLAUSE_HEADING_RE = re.compile(
    r"(?:^|(?<=\. ))"
    r"((?:[A-Z][a-z]+(?:'[a-z]+)?)"
    r"(?:"
    r"(?:\s+(?:and|&|of|the|to|for|on|in|or|with|at|by|an|a|your|its|from))*"
    r"\s+[A-Z][a-z]+(?:'[a-z]+)?"
    r")+"
    r"\.)\s"
)

# Regex: single-word clause titles like "Term. ", "Assignment. ", "Definitions. ".
# Constrained to first word with 4+ total characters to exclude "Mr", "Dr", "Ms",
# etc., and must be followed by a capital letter so we only match clause-body
# starts, not mid-sentence capitalization.
_SINGLE_WORD_HEADING_RE = re.compile(
    r"(?:^|(?<=\. ))"
    r"([A-Z][a-z]{3,}(?:'[a-z]+)?\.)"
    r"\s(?=[A-Z])"
)

# Words that look like single-word clause titles but are actually sentence
# adverbs or conjunctions. Anything in this set is ignored when matched by
# _SINGLE_WORD_HEADING_RE.
_SINGLE_WORD_HEADING_BLACKLIST = frozenset({
    "however", "further", "furthermore", "additionally", "moreover",
    "otherwise", "nevertheless", "notwithstanding", "accordingly",
    "therefore", "thus", "hence", "thereby", "thereafter", "whereas",
    "specifically", "particularly", "collectively", "individually",
    "alternatively", "consequently", "subsequently", "respectively",
    "conversely", "similarly", "meanwhile", "indeed", "finally",
    "firstly", "secondly", "thirdly", "lastly", "instead",
})


def _find_clause_heading_matches(text: str) -> List[Dict[str, Any]]:
    """Return clause-heading matches in paragraph order, deduped by start position.

    Each entry is {"start": int, "end": int, "heading": str} where end points
    to the first character AFTER the heading (i.e. start of body text).
    """
    found: Dict[int, Dict[str, Any]] = {}

    for m in _INLINE_CLAUSE_HEADING_RE.finditer(text):
        found[m.start()] = {"start": m.start(), "end": m.end(1) + 1, "heading": m.group(1)}

    for m in _SINGLE_WORD_HEADING_RE.finditer(text):
        word = m.group(1).rstrip(".").lower()
        if word in _SINGLE_WORD_HEADING_BLACKLIST:
            continue
        # Multi-word match at the same position wins — it's more specific.
        found.setdefault(m.start(), {"start": m.start(), "end": m.end(1) + 1, "heading": m.group(1)})

    return [found[k] for k in sorted(found)]


class DocxParser(BaseParser, Logger):
    """Semantic DOCX parser with heading detection and similarity-based chunking."""

    _HEADING_MAX_WORDS: int = 8     # Max words for structural heading heuristic
    _ORPHAN_MIN_WORDS: int = 5      # Min words for a standalone chunk

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        from src.dependencies import get_service_container

        self.service_container = get_service_container()
        self.embedding_service: BaseEmbeddingService = self.service_container.embedding_service
        self.vector_store = get_faiss_vector_store(self.embedding_service.get_embedding_dimensions())

    @staticmethod
    def _is_structural_heading(text: str, max_words: int = 8) -> bool:
        """True if text looks like a section heading (short + matches label pattern)."""
        words = text.split()
        if len(words) > max_words or len(words) == 0:
            return False
        return bool(_SECTION_LABEL_RE.match(text.strip()))

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Cosine similarity between two vectors."""
        return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    @staticmethod
    def _merge_orphan_chunks(chunks: List[str], min_words: int = 5) -> List[str]:
        """Merge chunks too short to stand alone into their neighbours."""
        if not chunks:
            return chunks

        merged: List[str] = []
        carry: str = ""

        for chunk in chunks:
            if carry:
                chunk = carry + " " + chunk
                carry = ""

            if len(chunk.split()) < min_words:
                carry = chunk
            else:
                merged.append(chunk)

        # Append any remaining short fragment to the last chunk
        if carry:
            if merged:
                merged[-1] = merged[-1] + " " + carry
            else:
                merged.append(carry)

        return merged

    # Clause numbering patterns (e.g. "1.1 ", "2.3.4 ", "(a) ", "b) ")
    _CLAUSE_PREFIX_RE = re.compile(
        r"^(\d+[\.\)]\d*[\.\d]*\s|"
        r"\([a-z]+\)\s|"
        r"[a-z]\)\s)"
    )

    def _clean_text(self, text: str) -> str:
        """Clean text while preserving clause numbering prefixes."""
        if not text or not text.strip():
            raise EmptyTextException("Text cannot be empty.")

        # Replace special whitespace chars
        text = text.replace("\u00a0", " ")
        text = text.replace("\u200b", "")
        text = text.replace("\ufeff", "")
        text = text.replace("\r", "")
        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Strip leading dots but preserve clause prefixes (e.g. "1.", "(a)")
        stripped = text.lstrip(" \n\t")
        if self._CLAUSE_PREFIX_RE.match(stripped):
            text = stripped
        else:
            text = stripped.lstrip(".")

        return text

    async def clean_document(self, document: Document) -> None:
        """Normalize whitespace in all paragraphs."""
        try:
            for paragraph in document.paragraphs:
                if paragraph.text:
                    paragraph.text = " ".join(paragraph.text.split())
        except Exception as e:
            raise DocxCleaningException(str(e)) from e

    async def _extract_metadata(self, document: Document) -> Dict[str, Any]:
        """Extract document metadata (author, title, dates, word count)."""
        try:
            self.logger.info("Extracting document metadata")
            props = document.core_properties
            para_words = sum(len(p.text.split()) for p in document.paragraphs)
            table_words = sum(
                len(cell.text.split())
                for table in document.tables
                for row in table.rows
                for cell in row.cells
            )

            return {
                "source": "docx",
                "author": props.author or "Unknown",
                "title": props.title or "Untitled",
                "created_at": props.created.isoformat() if props.created else None,
                "modified_at": props.modified.isoformat() if props.modified else None,
                "paragraph_count": len(document.paragraphs),
                "table_count": len(document.tables),
                "word_count": para_words + table_words,
            }
        except Exception as e:
            raise DocxMetadataExtractionException(str(e)) from e

    async def _extract_tables(self, document: Document) -> List[Dict[str, Any]]:
        """Extract all tables as lists of row data."""
        try:
            tables = []
            for t_idx, table in enumerate(document.tables):
                rows = [[self._clean_text(cell.text) for cell in row.cells] for row in table.rows]
                tables.append({"table_index": t_idx, "content": rows})
            return tables
        except Exception as e:
            raise DocxTableExtractionException(str(e)) from e

    async def _extract_paragraphs(self, document: Document) -> List[Dict[str, Any]]:
        """Extract paragraphs with heading detection (styled + structural heuristic)."""
        try:
            data = []
            for idx, p in enumerate(document.paragraphs):
                if p.text.strip():
                    cleaned = self._clean_text(p.text)
                    if cleaned:
                        is_heading = bool(
                            (p.style and p.style.name.startswith("Heading"))
                            or self._is_structural_heading(cleaned, self._HEADING_MAX_WORDS)
                        )
                        data.append({"index": idx, "content": cleaned, "is_heading": is_heading})
            return data
        except Exception as e:
            raise DocxParagraphExtractionException(str(e)) from e

    @staticmethod
    def _split_at_clause_boundaries(paragraphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split paragraphs at every inline clause heading.

        Detects both multi-word titles (e.g. "Audit Rights.",
        "Integration Testing.") and single-word titles (e.g. "Term.",
        "Assignment.", "Definitions.") and splits at every boundary so that
        each clause later becomes its own chunk.
        """
        result: List[Dict[str, Any]] = []

        for para in paragraphs:
            if para["is_heading"]:
                result.append(para)
                continue

            text = para["content"]
            boundaries = _find_clause_heading_matches(text)

            if not boundaries:
                result.append(para)
                continue

            segments: List[Dict[str, Any]] = []

            # Text before the first clause heading belongs to the previous
            # clause; emit as a plain (non-heading) paragraph.
            if boundaries[0]["start"] > 0:
                prefix = text[:boundaries[0]["start"]].strip()
                if prefix:
                    segments.append({"heading": None, "body": prefix})

            # Each clause heading starts a segment that runs to the next
            # heading (or end of paragraph).
            for i, b in enumerate(boundaries):
                body_start = b["end"]
                body_end = boundaries[i + 1]["start"] if i + 1 < len(boundaries) else len(text)
                body = text[body_start:body_end].strip()
                segments.append({"heading": b["heading"], "body": body})

            for seg in segments:
                if seg["heading"] is None:
                    # Prefix of a mid-paragraph heading — belongs to previous clause.
                    result.append({
                        "index": para["index"],
                        "content": seg["body"],
                        "is_heading": False,
                    })
                    continue

                result.append({
                    "index": para["index"],
                    "content": seg["heading"],
                    "is_heading": True,
                })
                if seg["body"]:
                    result.append({
                        "index": para["index"],
                        "content": seg["body"],
                        "is_heading": False,
                        "clause_boundary": True,
                    })

        return result

    async def _semantic_chunk_paragraphs(self, paragraphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group paragraphs into chunks based on embedding similarity.

        Returns list of {text, section_heading} dicts.
        """
        # Pre-process: split inline clause headings
        paragraphs = self._split_at_clause_boundaries(paragraphs)

        texts = [para["content"] for para in paragraphs]
        embeddings = [
            await self.embedding_service.generate_embeddings(text=t, task="text-matching")
            for t in texts
        ]

        # Compute pairwise similarities between consecutive paragraphs
        similarities = [
            self.cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        # Split at points below threshold (mean - 0.75 * std)
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        threshold = mean_sim - 0.75 * std_sim
        split_points = {i for i, sim in enumerate(similarities) if sim < threshold}

        # Build chunks respecting headings, size limits, and split points
        chunks: List[str] = []
        chunk_headings: List[Optional[str]] = []
        current: List[str] = []
        current_len: int = 0
        current_heading: Optional[str] = None
        max_len: int = self.settings.chunk_size
        pending_heading: Optional[str] = None

        def _flush() -> None:
            nonlocal current, current_len, pending_heading
            if current:
                chunks.append(" ".join(current))
                chunk_headings.append(pending_heading)
                current = []
                current_len = 0

        for i, para in enumerate(paragraphs):
            text = para["content"]
            text_len = len(text)

            if para["is_heading"]:
                current_heading = text

            # Flush before headings or when size limit exceeded
            if para["is_heading"]:
                _flush()
                pending_heading = current_heading
            elif current_len + text_len > max_len:
                _flush()
                pending_heading = current_heading

            if not current:
                pending_heading = current_heading

            current.append(text)
            current_len += text_len

            # Flush at semantic split points or clause boundaries
            if i in split_points or para.get("clause_boundary"):
                _flush()

        _flush()

        # Merge orphan chunks
        raw_texts = self._merge_orphan_chunks(chunks, min_words=self._ORPHAN_MIN_WORDS)

        # Re-align headings after merging
        merged_headings: List[Optional[str]] = []
        src_idx = 0
        for merged_text in raw_texts:
            if src_idx < len(chunk_headings):
                merged_headings.append(chunk_headings[src_idx])
            else:
                merged_headings.append(None)

            consumed = 0
            for j in range(src_idx, len(chunks)):
                if chunks[j] in merged_text:
                    consumed += 1
                else:
                    break
            src_idx += max(consumed, 1)

        return [
            {"text": text, "section_heading": heading}
            for text, heading in zip(raw_texts, merged_headings)
        ]

    async def parse_data(self, data: List[TextInfo], session_data: Optional[Any] = None) -> ParseResult:
        """Parse structured text data (list of paragraphs)."""
        start = time.time()

        paragraphs = [d.text for d in data]
        if not paragraphs:
            raise ValueError("No paragraphs found in the data.")

        # Embed and index each paragraph
        for para in paragraphs:
            if not para:
                continue
            vector = await self.embedding_service.generate_embeddings(text=para, task="text-matching")
            await self.vector_store.index_embedding(vector)

        chunks: List[Chunk] = []
        for i, text in enumerate(paragraphs):
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=None,
                    chunk_index=i,
                    content=text,
                    embedding_model=self.embedding_service.model_name,
                    embedding_vector=None,
                    metadata={"chunk_type": "semantic_paragraph"},
                    created_at=datetime.utcnow().isoformat(),
                )
            )

        return ParseResult(
            success=True,
            chunks=chunks,
            metadata={"paragraph_count": len(paragraphs), "source": "parsed_data"},
            processing_time=time.time() - start,
        )

    async def parse_document(self, document: Document, session_data: Optional["SessionData"] = None) -> ParseResult:
        """Parse a DOCX document using semantic chunking."""
        start = time.time()

        try:
            await self.clean_document(document)
            metadata = await self._extract_metadata(document)

            document_id = str(uuid.uuid4())
            metadata["document_id"] = document_id

            paragraphs = await self._extract_paragraphs(document)
            tables = await self._extract_tables(document)

            vector_store = session_data.vector_store if session_data else self.vector_store
            semantic_chunks = await self._semantic_chunk_paragraphs(paragraphs)

            chunks: List[Chunk] = []
            chunk_index = 0

            # Text chunks
            for chunk_info in semantic_chunks:
                cleaned = self._clean_text(chunk_info["text"])
                if not cleaned:
                    continue

                vector = await self.embedding_service.generate_embeddings(text=cleaned, task="text-matching")
                await vector_store.index_embedding(vector)

                chunk_metadata: Dict[str, Any] = {"chunk_type": "semantic_paragraph"}
                if chunk_info.get("section_heading"):
                    chunk_metadata["section_heading"] = chunk_info["section_heading"]

                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        chunk_index=chunk_index,
                        content=cleaned,
                        embedding_model=self.embedding_service.model_name,
                        embedding_vector=None,
                        metadata=chunk_metadata,
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
                        embedding_vector=None,
                        metadata={
                            "chunk_type": "table",
                            "table_index": table["table_index"],
                            "row_count": len(table["content"]),
                        },
                        created_at=datetime.utcnow().isoformat(),
                    )
                )
                chunk_index += 1

            # NOTE: chunk_store indexing is handled by IngestionService._parse_data()

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
                success=False, chunks=[], metadata={},
                error_message=str(e), processing_time=0.0,
            )

    def is_healthy(self) -> Any:
        """Check parser health status."""
        return True
