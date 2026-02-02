from typing import Any, Dict, List, Optional

from src.config.logging import Logger
from src.config.settings import get_settings
from src.services.vector_store.embeddings.embedding_service import (
    BGEEmbeddingService,
    HuggingFaceEmbeddingService,
)
from src.services.vector_store.embeddings.jina_embeddings import JinaEmbeddings
from src.services.vector_store.embeddings.openai_embeddings import OpenAIEmbeddings
from src.services.vector_store.manager import get_chunks, get_faiss_vector_store


class RetrievalService(Logger):
    """Retrieval Service for retrieving the data."""

    def __init__(self) -> None:
        super().__init__()

        self.settings = get_settings()
        # self.embedding_service = HuggingFaceEmbeddingService()
        # self.embedding_service = OpenAIEmbeddings()
        self.embedding_service = BGEEmbeddingService()
        # self.embedding_service = JinaEmbeddings()
        self.vector_store = get_faiss_vector_store(self.embedding_service.get_embedding_dimensions())

    async def retrieve_data(self, query: str, top_k: int = 5, threshold: Optional[float] = 0.0) -> Dict[str, Any]:
        """Retrieve and return relevant document chunks based on query."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embeddings(text=query, task="retrieval.query")

            # Search vector store for top-k similar embeddings
            search_result = await self.vector_store.search_index(query_embedding, top_k)

            indices = search_result.get("indices", [])
            scores = search_result.get("scores", [])

            # Fetch chunks from the manager by their indices
            retrieved_chunks = []
            for idx, score in zip(indices, scores):
                if threshold is not None and score < threshold:
                    self.logger.debug(f"Skipping result with score {score} (below threshold {threshold})")
                    continue

                chunk = get_chunks([idx])
                if chunk:
                    retrieved_chunks.append(
                        {
                            "index": idx,
                            "content": chunk[0].content,
                            "similarity_score": float(score),
                            "metadata": chunk[0].metadata,
                            "created_at": chunk[0].created_at,
                        }
                    )

            result = {
                "query": query,
                "chunks": retrieved_chunks,
                "num_results": len(retrieved_chunks),
                "search_metadata": {
                    "search_time": search_result.get("search_time", 0),
                    "requested_top_k": top_k,
                    "returned_results": len(retrieved_chunks),
                },
            }

            self.logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query")
            return result

        except Exception as e:
            self.logger.error(f"Error retrieving data: {str(e)}")
            raise ValueError("Unable to retrieve the data.") from e
