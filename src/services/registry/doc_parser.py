import time
from datetime import datetime
from typing import Any, Dict, List

from docx.document import Document

from src.config.logging import Logger
from src.config.settings import get_settings
from src.schemas.registry import Chunk, ParseResult
from src.services.registry.base_parser import BaseParser


class DocxParser(BaseParser, Logger):
    """Parser for DOCX documents."""

    def __init__(self) -> None:
        """Initialize the DOCX parser."""

        super().__init__()
        self.settings = get_settings()

    async def clean_document(self, document: Document) -> None:
        """Clean the document before parsing to remove unwanted elements."""

        try:
            for paragraph in document.paragraphs:
                if paragraph.text:
                    paragraph.text = " ".join(paragraph.text.split())

            self.logger.debug("Document cleaned successfully.")
        except Exception as e:
            self.logger.error(f"Error cleaning document: {e}")

    async def _extract_metadata(self, document: Document) -> Dict[str, Any]:
        """Extract metadata about the document."""

        try:
            properties = document.core_properties

            metadata: Dict[str, Any] = {
                "source": "docx",
                "author": properties.author or "Unknown",
                "title": properties.title or "Untitled",
                "subject": properties.subject or "",
                "created_at": properties.created.isoformat() if properties.created else None,
                "modified_at": properties.modified.isoformat() if properties.modified else None,
                "last_modified_by": properties.last_modified_by or "Unknown",
                "revision": properties.revision or 0,
                "paragraph_count": len(document.paragraphs),
                "table_count": len(document.tables),
                "image_count": len(document.inline_shapes),
                "word_count": sum(len(p.text.split()) for p in document.paragraphs),
            }

            return metadata
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {e}")
            return {"source": "docx", "error": "Metadata extraction failed"}

    async def _extract_paragraphs(self, document: Document) -> List[Dict[str, Any]]:
        """Extract paragraphs from the document."""

        paragraphs_data: List[Dict[str, Any]] = []

        try:
            for index, paragraph in enumerate(document.paragraphs):
                if paragraph.text.strip():
                    paragraphs_data.append(
                        {
                            "index": index,
                            "content": paragraph.text.strip(),
                            "is_heading": paragraph.style.name.startswith("Heading") if paragraph.style else False,
                        }
                    )

        except Exception as e:
            self.logger.error(f"Error extracting paragraphs: {e}")

        return paragraphs_data

    async def parse(self, document: Document) -> ParseResult:
        """Parse the DOCX data."""

        start_time = time.time()

        try:
            self.logger.info("Starting DOCX parsing.")

            # Clean the document
            await self.clean_document(document=document)

            # Extract metadata
            metadata = await self._extract_metadata(document=document)

            # Extract paragraphs
            paragraphs = await self._extract_paragraphs(document=document)

            full_text = "\n\n".join([p["content"] for p in paragraphs])

            processing_time = time.time() - start_time

            # We need to perform the chunking here
            # ------------------------------------

            chunk = Chunk(
                chunk_id=None,
                document_id=None,
                chunk_index=0,
                content=full_text,
                embedding_vector=None,
                embedding_model=None,
                metadata={},
                created_at=datetime.utcnow().isoformat(),
            )

            return ParseResult(
                success=True,
                chunks=[chunk],
                metadata=metadata,
                error_message=None,
                processing_time=processing_time,
            )
        except Exception as e:
            self.logger.error(f"Error parsing DOCX document: {e}")
            return ParseResult(
                success=False,
                chunks=[],
                metadata={},
                error_message=str(e),
                processing_time=0.0,
            )
