"""Parser registry — manages available document parsers."""

from typing import Dict, Optional

from src.config.logging import Logger
from src.exceptions.parser_exceptions import ParserAlreadyRegistered
from src.services.registry.base_parser import BaseParser
from src.services.registry.semantic_parser import DocxParser


class ParserRegistry(Logger):
    """Registry for document parsers, keyed by file type."""

    def __init__(self) -> None:
        self.parsers: Dict[str, BaseParser] = {}
        self._register_default_parsers()

    def _register_default_parsers(self) -> None:
        """Register built-in parsers."""
        self.parsers["DOCX"] = DocxParser()
        self.logger.info("Registered default parser: DOCX")

    def register_parser(self, name: str, parser_class: BaseParser) -> None:
        """Register a new parser by name."""
        if name in self.parsers:
            raise ParserAlreadyRegistered(f"Parser '{name}' is already registered.")
        self.parsers[name] = parser_class
        self.logger.info(f"Registered parser: {name}")

    def get_parser(self) -> Optional[BaseParser]:
        """Get the appropriate parser (currently defaults to DOCX)."""
        return self.parsers.get("DOCX")
