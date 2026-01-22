from abc import ABC, abstractmethod
from src.config.logging import Logger
from src.schemas.registry import ParseResult

class BaseParser(ABC, Logger):
    """Abstract base class for parsers in the registry service."""

    @abstractmethod
    def parse(self, data: str) -> ParseResult:
        """Parse the given data """
        pass 
