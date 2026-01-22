from src.config.settings import get_settings
from src.config.logging import Logger
from src.services.registry.base_parser import BaseParser
from src.schemas.registry import ParseResult

class DocxParser(BaseParser, Logger):
    """Parser for DOCX documents."""

    def __init__(self) -> None:
        """Initialize the DOCX parser."""

        super().__init__()
        self.settings = get_settings()

    def parse(self, data: str) -> ParseResult:
        """Parse the DOCX data."""

        self.logger.info("Starting DOCX parsing.") 

        return ParseResult(
            success=True,
            chunks=[{"text": "This is a sample parsed chunk."}], 
            metadata={"source": "docx"}, 
            error_message=None,
            processing_time=0.12345 
        )