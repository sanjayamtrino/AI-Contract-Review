from abc import ABC, abstractmethod

from docx.document import Document

from src.config.logging import Logger
from src.schemas.registry import ParseResult


class BaseParser(ABC, Logger):
    """Abstract base class for parsers in the registry service."""

    @abstractmethod
    async def parse(self, document: Document) -> ParseResult:
        """Parse the given document."""
        pass

    @abstractmethod
    async def clean_document(self, document: Document) -> None:
        """Clean the document before parsing."""
        pass
