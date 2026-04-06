"""
Ingestion service — parses uploaded documents, generates embeddings,
and indexes chunks into the session or global vector store.
"""

import time
from io import BytesIO
from typing import Any, Dict, Optional, Union

from docx import Document

from src.config.logging import Logger
from src.config.settings import get_settings
from src.exceptions.ingestion_exceptions import ParserNotFound
from src.schemas.registry import ParseResult
from src.services.registry.base_parser import BaseParser
from src.services.registry.registry import ParserRegistry
from src.services.vector_store.embeddings.base_embedding_service import BaseEmbeddingService
from src.services.vector_store.manager import index_chunks, index_chunks_in_session


class IngestionService(Logger):
    """Orchestrates document parsing, embedding generation, and vector indexing."""

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self.registry = ParserRegistry()
        self.vector_store = None

        from src.dependencies import get_service_container
        service_container = get_service_container()
        self.embedding_service: BaseEmbeddingService = service_container.embedding_service

    async def _parse_data(self, data: Union[BytesIO, Dict[str, Any]], session_data: Optional[Any] = None) -> ParseResult:
        """Parse input data, generate embeddings, and index into the vector store."""
        parser: Union[BaseParser, None] = self.registry.get_parser()

        if not parser:
            self.logger.error("No parser found for the given extension.")
            raise ParserNotFound("No parser found. Check available parsers via the /parsers API.")

        # Parse document or structured data
        if isinstance(data, BytesIO):
            start_time = time.time()
            document = Document(data)
            parsed_data: ParseResult = await parser.parse_document(document=document, session_data=session_data)
            parsed_data.processing_time = time.time() - start_time
            self.logger.info(f"Document parsed in {parsed_data.processing_time:.2f}s")
        else:
            start_time = time.time()
            parsed_data: ParseResult = await parser.parse_data(data=data, session_data=session_data)
            parsed_data.processing_time = time.time() - start_time
            self.logger.info(f"Data parsed in {parsed_data.processing_time:.2f}s")

        # Generate embeddings and index chunks
        if parsed_data.chunks:
            for chunk in parsed_data.chunks:
                embedding = await self.embedding_service.generate_embeddings(text=chunk.content, task="text-matching")

                if session_data:
                    await session_data.vector_store.index_embedding(embedding)
                else:
                    from src.services.vector_store.manager import get_faiss_vector_store
                    global_store = get_faiss_vector_store(self.embedding_service.get_embedding_dimensions())
                    await global_store.index_embedding(embedding)

            # Index chunks into chunk store
            if session_data:
                index_chunks_in_session(session_data, parsed_data.chunks, parsed_data.metadata)
                self.logger.info(f"Indexed {len(parsed_data.chunks)} chunks into session {session_data.session_id}")
            else:
                index_chunks(parsed_data.chunks)
                self.logger.info(f"Indexed {len(parsed_data.chunks)} chunks into global store")

        return parsed_data

    async def _get_health_status(self) -> Dict[str, Any]:
        """Check parser and vector store availability."""
        parser = self.registry.get_parser()
        health_info: Dict[str, Any] = {
            "parser_accessible": await parser.is_healthy() if parser else False,
            "vector_store_accessible": self.vector_store is not None,
        }
        health_info["status"] = health_info["parser_accessible"] and health_info["vector_store_accessible"]
        return health_info
