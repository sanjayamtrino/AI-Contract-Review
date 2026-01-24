import time
from datetime import datetime
from typing import Any, Dict, List

from docx.document import Document

from src.config.logging import Logger
from src.config.settings import get_settings
from src.schemas.registry import Chunk, ParseResult
from src.services.ingestion.chunker import TextChunker
from src.services.ingestion.cleaner import DocumentCleaner
from src.services.ingestion.loader import DocxLoader
from src.services.registry.base_parser import BaseParser


class DocxParser(BaseParser, Logger):
    """
    DOCX Parser implementing load → clean → chunk pipeline.
    """

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self.loader = DocxLoader()
        self.cleaner = DocumentCleaner()
        self.chunker = TextChunker()

    async def _extract_metadata(self, document: Document) -> Dict[str, Any]:
        props = document.core_properties

        return {
            "source": "docx",
            "author": props.author or "Unknown",
            "title": props.title or "Untitled",
            "created_at": props.created.isoformat() if props.created else None,
            "modified_at": props.modified.isoformat() if props.modified else None,
            "paragraph_count": len(document.paragraphs),
            "table_count": len(document.tables),
        }

    async def parse(self, data: bytes) -> ParseResult:
        """
        Parse DOCX bytes into structured chunks.
        """
        start_time = time.time()

        try:
            self.logger.info("Starting DOCX parsing pipeline.")

            document = await self.loader.load_from_bytes(data)
            await self.cleaner.clean(document)

            metadata = await self._extract_metadata(document)

            full_text = "\n\n".join(p.text for p in document.paragraphs if p.text.strip())

            raw_chunks = self.chunker.chunk_text(full_text)

            chunks: List[Chunk] = []
            for idx, content in enumerate(raw_chunks):
                chunks.append(
                    Chunk(
                        chunk_id=None,
                        document_id=None,
                        chunk_index=idx,
                        content=content,
                        embedding_vector=None,
                        embedding_model=None,
                        metadata={},
                        created_at=datetime.utcnow().isoformat(),
                    )
                )

            return ParseResult(
                success=True,
                chunks=chunks,
                metadata=metadata,
                error_message=None,
                processing_time=time.time() - start_time,
            )

        except Exception as exc:
            self.logger.error("DOCX parsing failed.", extra={"reason": str(exc)})

            return ParseResult(
                success=False,
                chunks=[],
                metadata={},
                error_message=str(exc),
                processing_time=time.time() - start_time,
            )
