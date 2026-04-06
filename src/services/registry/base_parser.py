"""Abstract base class for document parsers."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from docx.document import Document

from src.config.logging import Logger
from src.schemas.registry import ParseResult
from src.services.session_manager import SessionData


class BaseParser(ABC, Logger):
    """All parsers in the registry must implement this interface."""

    @abstractmethod
    async def parse_document(self, document: Document, session_data: Optional["SessionData"] = None) -> ParseResult:
        """Parse a DOCX document into chunks."""
        pass

    @abstractmethod
    async def parse_data(self, data: Any, session_data: Optional["SessionData"] = None) -> ParseResult:
        """Parse structured data into chunks."""
        pass

    @abstractmethod
    async def clean_document(self, document: Document) -> None:
        """Pre-process the document before parsing."""
        pass

    @abstractmethod
    def is_healthy(self) -> Any:
        """Check parser health status."""
        pass
