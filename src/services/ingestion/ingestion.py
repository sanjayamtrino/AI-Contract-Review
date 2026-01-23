import time
from io import BytesIO
from typing import Union

from docx import Document

from src.config.logging import Logger
from src.config.settings import get_settings
from src.schemas.registry import ParseResult
from src.services.registry.base_parser import BaseParser
from src.services.registry.registry import ParserRegistry


class IngestionService(Logger):
    """Ingestion service for processing data."""

    def __init__(self) -> None:
        """Initialize the ingestion service."""

        super().__init__()
        self.settings = get_settings()
        self.registry = ParserRegistry()
        self.vector_store = None

    async def _parse_data(self, data: BytesIO) -> ParseResult:
        """Parse data using the registry services."""

        parser: Union[BaseParser, None] = self.registry.get_parser()

        if not parser:
            self.logger.error("No parser found for the given extension.")
            raise ValueError("No parser found for the given extension.")

        start_time = time.time()
        document = Document(data)
        parsed_data = await parser.parse(document=document)
        parsed_data.processing_time = time.time() - start_time
        self.logger.info(f"Data parsed in {parsed_data.processing_time:.2f} seconds.")
        return parsed_data
