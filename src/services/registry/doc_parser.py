import time
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List

from docx import Document as DocxDocument
from docx.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
from src.services.vector_store.embedding_service import (
    BGEEmbeddingService,
    HuggingFaceEmbeddingService,
)
from src.services.vector_store.gemini_embeddings import GeminiEmbeddingService


class DocxParser(BaseParser, Logger):
    """Parser for DOCX documents."""

    def __init__(self) -> None:
        """Initialize the DOCX parser."""

        super().__init__()
        self.settings = get_settings()
        # self.embedding_service = HuggingFaceEmbeddingService()
        # self.embedding_service = BGEEmbeddingService()
        self.embedding_service = GeminiEmbeddingService()

    async def clean_document(self, document: Document) -> None:
        """Clean the document before parsing to remove unwanted elements."""

        try:
            for paragraph in document.paragraphs:
                if paragraph.text:
                    paragraph.text = " ".join(paragraph.text.split())

            self.logger.debug("Document cleaned successfully.")
        except Exception as e:
            self.logger.error(f"Error cleaning document: {e}")
            raise DocxCleaningException(f"Error cleaning document: {e}") from e

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
            raise DocxMetadataExtractionException(f"Error extracting metadata: {e}") from e

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
            raise DocxParagraphExtractionException(f"Error extracting paragraphs: {e}") from e

        return paragraphs_data

    async def _extract_images(self, document: Document) -> List[Dict[str, Any]]:
        """Extract images from the document."""

        return []

    async def _extract_tables(self, document: Document) -> List[Dict[str, Any]]:
        """Extract tables from the document."""

        tables_data: List[Dict[str, Any]] = []

        try:
            for table_index, table in enumerate(document.tables):
                table_content = []
                for row in table.rows:
                    row_data = [cell.text.strip() for cell in row.cells]
                    table_content.append(row_data)

                tables_data.append(
                    {
                        "table_index": table_index,
                        "content": table_content,
                    }
                )

        except Exception as e:
            self.logger.error(f"Error extracting tables: {e}")
            raise DocxTableExtractionException(f"Error extracting tables: {e}") from e

        return tables_data

    async def _get_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Get the text splitter for chunking the document content."""

        return RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=[
                "\n\n\n",  # Multiple newlines (section breaks)
                "\n\n",  # Paragraph breaks
                "\n",  # Line breaks
                ". ",  # Sentence ends
                "! ",  # Exclamation
                "? ",  # Question
                "; ",  # Semicolon
                ": ",  # Colon
                ", ",  # Comma
                " ",  # Space
                "",  # Character level (last resort)
            ],
            length_function=len,
            keep_separator=True,
            is_separator_regex=False,
        )

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

            # Extract tables
            # tables = await self._extract_tables(document=document)

            full_text = "\n\n".join([p["content"] for p in paragraphs])

            # full_tables_text = "\n\n".join(["\n".join(["\t".join(row) for row in table["content"]]) for table in tables])
            # print(full_tables_text)

            # We need to perform the chunking here and create list of chunks
            # ------------------------------------

            text_splitter = await self._get_text_splitter()

            texts: List[str] = text_splitter.split_text(full_text)
            chunks: List[Chunk] = []

            for index, text in enumerate(texts):
                self.logger.debug(f"Chunk created with length {len(text)}.")

                # Embedd the text
                vector_data = await self.embedding_service.generate_embeddings(text=text)

                chunk = Chunk(
                    chunk_id=None,
                    document_id=None,
                    chunk_index=index,
                    content=text,
                    embedding_model=self.embedding_service.model_name,
                    embedding_vector=vector_data,
                    metadata={},
                    created_at=datetime.utcnow().isoformat(),
                )
                chunks.append(chunk)

            processing_time = time.time() - start_time

            return ParseResult(
                success=True,
                chunks=chunks,
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

    async def _get_health_status(self) -> Dict[str, Any]:
        """Get the health status of the DOCX parser."""

        status: str = "healthy"
        info: Dict[str, Any] = {}

        # Settings accessibility check
        try:
            _ = self.settings.chunk_size
            _ = self.settings.chunk_overlap
            info["settings_accessible"] = True
        except Exception as e:
            status = "unhealthy"
            info["settings_accessible"] = False
            info["error"] = str(e)

        # Text splitter check
        try:
            _ = await self._get_text_splitter()
            info["text_splitter_accessible"] = True
        except Exception as e:
            status = "unhealthy"
            info["text_splitter_accessible"] = False
            info["error"] = str(e)

        # DOCX test parsing check
        try:

            sample_docx = BytesIO()
            doc = DocxDocument()
            doc.add_paragraph("This is a test paragraph.")
            doc.save(sample_docx)
            sample_docx.seek(0)

            document = DocxDocument(sample_docx)
            parse_result = await self.parse(document=document)

            if parse_result.success:
                info["docx_parsing"] = "successful"
            else:
                status = "unhealthy"
                info["docx_parsing"] = "failed"
                info["error"] = parse_result.error_message
        except Exception as e:
            status = "unhealthy"
            info["docx_parsing"] = "failed"
            info["error"] = str(e)

        return {
            "status": status,
            "info": info,
        }

    async def is_healthy(self) -> bool:
        """Checks if the Parser is in healthy condition."""

        status: Dict[str, Any] = await self._get_health_status()
        return True if status.get("status", "") == "healthy" else False
