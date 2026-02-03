import time
from io import BytesIO
from typing import Any, Dict, Union

from docx import Document

from src.config.logging import Logger
from src.config.settings import get_settings
from src.schemas.registry import ParseResult
from src.services.registry.base_parser import BaseParser
from src.services.registry.registry import ParserRegistry
from src.services.vector_store.gemini_embeddings import GeminiEmbeddingService
from src.services.vector_store.faiss_db import InMemoryFIASS
from langchain_text_splitters import RecursiveCharacterTextSplitter


class IngestionService(Logger):
    """Ingestion service for processing data."""

    def __init__(self) -> None:
        """Initialize the ingestion service."""

        super().__init__()
        self.settings = get_settings()
        self.registry = ParserRegistry()
        
        # Initialize Embedding Service and Vector Store
        try:
           self.embedding_service = GeminiEmbeddingService()
           self.vector_store = InMemoryFIASS()
           self.text_splitter = RecursiveCharacterTextSplitter(
               chunk_size=1000,
               chunk_overlap=200,
               length_function=len
           )
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store components: {e}")
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
        
        # Store in Vector DB if available
        if self.vector_store and self.embedding_service:
            try:
                # 1. Chunk the text
                full_text = parsed_data.content # Assuming ParseResult has content field
                if not full_text:
                    # If content is empty, try to combine paragraphs/sections
                    # This depends on how ParseResult is structured. 
                    # Use a safe fallback if content is missing.
                     # Re-read doc to get text if needed, but Parser should provide it.
                     pass

                chunks = self.text_splitter.split_text(full_text)
                self.logger.info(f"Generated {len(chunks)} chunks.")

                # 2. Generate Embeddings & Metadata
                vectors = []
                metadata = []
                
                for i, chunk in enumerate(chunks):
                    # We process chunks sequentially here. For production, batching is better.
                    vec = await self.embedding_service.generate_embeddings(chunk)
                    vectors.append(vec)
                    metadata.append({
                        "text": chunk,
                        "source": parsed_data.filename,
                        "chunk_index": i
                    })
                
                # 3. Add to Store
                if vectors:
                    self.vector_store.add_vectors(vectors, metadata)
                    self.logger.info(f"Stored {len(vectors)} vectors in FAISS.")
                    
            except Exception as e:
                self.logger.error(f"Failed to store vectors: {e}")
                
        return parsed_data

    async def _get_health_status(self) -> Dict[str, Any]:
        """Get the health status of the ingestion service."""

        parser = self.registry.get_parser()
        health_info: Dict[str, Any] = {
            "parser_accessible": await parser.is_healthy() if parser else False,
            "vector_store_accessible": self.vector_store is not None,
        }

        health_info["status"] = health_info["parser_accessible"] and health_info["vector_store_accessible"]

        return health_info
