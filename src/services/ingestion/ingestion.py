import time

from src.config.settings import get_settings 
from src.config.logging import Logger
from src.services.registry.base_parser import BaseParser
from src.services.registry.registry import ParserRegistry
from src.schemas.registry import ParseResult

class IngestionService(Logger):
    """Ingestion service for processing data.""" 

    def __init__(self) -> None:
        """Initialize the ingestion service."""

        super().__init__()
        self.settings = get_settings() 
        self.registry = ParserRegistry()
        self.vector_store = None 

    def _parse_data(self, data: str) -> ParseResult:
        """Parse data using the registry services."""
        parser: BaseParser = self.registry.get_parser() 

        if not parser:
            self.logger.error("No parser found for the given extension.") 
            raise ValueError("No parser found for the given extension.") 
        
        start_time = time.time()
        parsed_data = parser.parse(data=data) 
        parsed_data.processing_time = time.time() - start_time
        self.logger.info(f"Data parsed in {parsed_data.processing_time:.2f} seconds.")
        return parsed_data 