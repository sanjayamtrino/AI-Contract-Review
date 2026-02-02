from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.logging import Logger
from src.config.settings import get_settings
from src.schemas.query_rewriter import QueryRewriterResponse
from src.services.llm.gemini_model import GeminiModel
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
        self.llm = GeminiModel()
        self.rewrite_query_prompt = Path(r"src\services\prompts\v1\query_rewriter.mustache").read_text()
        self.vector_store = get_faiss_vector_store(self.embedding_service.get_embedding_dimensions())

    async def rewrite_query(self, query: str) -> List[str]:
        """Rewrite the given query."""

        context: Dict[str, Any] = {
            "query": query,
        }
        response: QueryRewriterResponse = await self.llm.generate(prompt=self.rewrite_query_prompt, context=context, response_model=QueryRewriterResponse)
        return response.queries

    async def retrieve_data(self, query: str, top_k: int = 5, threshold: Optional[float] = 0.0) -> Dict[str, Any]:
        """Retrieve and return relevant document chunks based on query."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        try:
            queries = await self.rewrite_query(query=query)

            all_hits: Dict[int, Dict[str, Any]] = {}

            for query_rewriten in queries:
                new_query = query + " | " + query_rewriten
                print(new_query)
                # Generate query embedding
                query_embedding = await self.embedding_service.generate_embeddings(text=new_query, task="retrieval.query")

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

                    if idx not in all_hits or score > all_hits[idx]["similarity_score"]:
                        chunk = get_chunks([idx])
                        if not chunk:
                            continue
                        all_hits[idx] = {
                            "index": idx,
                            "content": chunk[0].content,
                            "similarity_score": float(score),
                            "metadata": chunk[0].metadata,
                            "created_at": chunk[0].created_at,
                            "matched_query": new_query,
                        }

            ranked_chunks = sorted(
                all_hits.values(),
                key=lambda x: x["similarity_score"],
                reverse=True,
            )[:top_k]

            self.logger.info(f"Retrieved {len(ranked_chunks)} chunks for query")

            return {
                "query": new_query,
                "rewritten_queries": queries,
                "chunks": ranked_chunks,
                "num_results": len(ranked_chunks),
                "search_metadata": {
                    "search_time": search_result.get("search_time", 0),
                    "requested_top_k": top_k,
                    "returned_results": len(retrieved_chunks),
                },
            }

        except Exception as e:
            self.logger.error(f"Error retrieving data: {str(e)}")
            raise ValueError("Unable to retrieve the data.") from e
