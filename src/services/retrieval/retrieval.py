from typing import Optional

from src.config.logging import Logger
from src.config.settings import get_settings
from src.services.vector_store.embeddings.embedding_service import (
    HuggingFaceEmbeddingService,
)
from src.services.vector_store.manager import get_faiss_vector_store


class RetrievalService(Logger):
    """Retrieval Service for retrieving the data."""

    def __init__(self) -> None:
        super().__init__()

        self.settings = get_settings()
        self.embedding_service = HuggingFaceEmbeddingService()
        self.vector_store = get_faiss_vector_store(self.embedding_service.get_embedding_dimensions())

    async def retrieve_data(self, query: str, top_k: int = 5, threshold: Optional[float] = 0.5) -> None:
        """Retrieve and return relavent embeddings."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        try:
            query_embedding = await self.embedding_service.generate_embeddings(text=query)
            search_result = await self.vector_store.search_index(query_embedding, top_k)

            print(search_result)

        except Exception as e:
            raise ValueError("Unable to retrive the data.") from e
