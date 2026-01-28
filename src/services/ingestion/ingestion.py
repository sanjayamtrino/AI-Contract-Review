import time
from io import BytesIO
from typing import Any, Dict, Optional

from docx import Document

from src.config.logging import Logger
from src.config.settings import get_settings
from src.exceptions.base_exception import AppException
from src.exceptions.ingestion_exceptions import DocumentLoadException, ParserNotFoundException, ParsingFailedException
from src.schemas.registry import ParseResult
from src.services.registry.base_parser import BaseParser
from src.services.registry.registry import ParserRegistry


class IngestionService(Logger):
    """Ingestion service for processing documents."""

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self.registry = ParserRegistry()
        self.vector_store = None

    async def _parse_data(self, data: BytesIO) -> ParseResult:
        """
        Ingest and parse document bytes.

        Flow:
        load → parse → return structured chunks
        """
        start_time = time.time()

        parser: Optional[BaseParser] = self.registry.get_parser()

        if not parser:
            self.logger.error(
                "Parser resolution failed",
                extra={"event": "PARSER_NOT_FOUND"},
            )
            raise ParserNotFoundException("No parser registered for this document type.")

        try:
            self.logger.info(
                "Loading document",
                extra={"event": "DOCUMENT_LOAD_START"},
            )
            document = Document(data)

        except Exception as exc:
            self.logger.error(
                "Document loading failed",
                extra={
                    "event": "DOCUMENT_LOAD_FAILED",
                    "reason": str(exc),
                },
            )
            raise DocumentLoadException("Failed to load document.") from exc

        try:
            self.logger.info(
                "Starting document parsing",
                extra={"event": "DOCUMENT_PARSE_START"},
            )

            result = await parser.parse(document=document)
            result.processing_time = time.time() - start_time

            self.logger.info(
                "Document parsed successfully",
                extra={
                    "event": "DOCUMENT_PARSE_COMPLETED",
                    "duration_ms": int(result.processing_time * 1000),
                    "chunks": len(result.chunks),
                },
            )

            return result

        except AppException:
            # Already meaningful, just bubble up
            raise

        except Exception as exc:
            self.logger.error(
                "Parsing failed",
                extra={
                    "event": "DOCUMENT_PARSE_FAILED",
                    "reason": str(exc),
                },
            )
            raise ParsingFailedException("Parser failed to process document.") from exc

    async def _get_health_status(self) -> Dict[str, Any]:
        """Health status of ingestion service."""
        parser = self.registry.get_parser()

        health: Dict[str, Any] = {
            "parser_available": parser is not None,
            "vector_store_connected": self.vector_store is not None,
            "status": "healthy",
        }

        if not parser:
            health["status"] = "unhealthy"

        return health
