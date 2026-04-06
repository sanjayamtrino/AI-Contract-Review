"""
Retrieval service — semantic search with query rewriting and dynamic top-k.

Rewrites the user query into multiple variants, embeds each, searches the
FAISS index, deduplicates, and optionally applies dynamic top-k expansion.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.logging import Logger
from src.config.settings import get_settings
from src.schemas.query_rewriter import QueryRewriterResponse
from src.services.llm.base_model import BaseLLMModel
from src.services.session_manager import SessionData
from src.services.vector_store.embeddings.base_embedding_service import BaseEmbeddingService
from src.services.vector_store.manager import get_chunks, get_chunks_from_session, get_faiss_vector_store


class RetrievalService(Logger):
    """Retrieves relevant document chunks via embedding similarity search."""

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()

        from src.dependencies import get_service_container
        service_container = get_service_container()
        self.embedding_service: BaseEmbeddingService = service_container.embedding_service
        self.llm: BaseLLMModel = service_container.azure_openai_model
        self.rewrite_query_prompt = Path(r"src\services\prompts\v1\query_rewriter.mustache").read_text()
        self.vector_store = get_faiss_vector_store(self.embedding_service.get_embedding_dimensions())

    async def rewrite_query(self, query: str) -> List[str]:
        """Generate multiple query variants for broader retrieval coverage."""
        context: Dict[str, Any] = {"query": query}
        self.logger.info(f"Rewriting query: {query}")

        response: QueryRewriterResponse = await self.llm.generate(
            prompt=self.rewrite_query_prompt, context=context, response_model=QueryRewriterResponse
        )
        return [q.query for q in response.queries]

    async def retrieve_document(self) -> Dict[str, Any]:
        """Retrieve all document chunks (placeholder)."""
        return {}

    async def retrieve_data(
        self,
        query: str,
        top_k: int = 5,
        dynamic_k: bool = False,
        threshold: Optional[float] = 0.0,
        session_data: Optional[SessionData] = None,
    ) -> Dict[str, Any]:
        """Retrieve relevant chunks for a query with optional dynamic top-k expansion.

        Dynamic top-k keeps adding chunks as long as the score drop between
        consecutive results stays within 2%, up to 2x the requested top_k.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        try:
            queries = await self.rewrite_query(query=query)
            self.logger.info(f"Generated {len(queries)} rewritten queries for: '{query}'")

            all_hits: Dict[int, Dict[str, Any]] = {}
            initial_k = max(20, top_k * 3) if dynamic_k else top_k

            for query_rewritten in queries:
                new_query = query + " | " + query_rewritten

                self.logger.info(f"Generating embedding for: '{new_query}'")
                query_embedding = await self.embedding_service.generate_embeddings(text=new_query, task="retrieval.query")

                # Search session-specific or global index
                if session_data:
                    search_result = await session_data.vector_store.search_index(query_embedding, initial_k)
                    chunk_getter = lambda idx: get_chunks_from_session(session_data, [idx])  # noqa: E731
                else:
                    search_result = await self.vector_store.search_index(query_embedding, initial_k)
                    chunk_getter = lambda idx: get_chunks([idx])  # noqa: E731

                indices = search_result.get("indices", [])
                scores = search_result.get("scores", [])

                # Deduplicate hits, keeping the highest score per chunk
                for idx, score in zip(indices, scores):
                    if threshold is not None and score < threshold:
                        continue
                    if idx not in all_hits or score > all_hits[idx]["similarity_score"]:
                        chunk = chunk_getter(idx)
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

            ranked_chunks = sorted(all_hits.values(), key=lambda x: x["similarity_score"], reverse=True)

            # Apply dynamic top-k or standard top-k
            final_chunks = []
            if dynamic_k and ranked_chunks:
                base_chunks = ranked_chunks[:top_k]
                final_chunks.extend(base_chunks)

                if base_chunks and ranked_chunks[top_k:]:
                    last_score = base_chunks[-1]["similarity_score"]

                    for chunk in ranked_chunks[top_k:]:
                        current_score = chunk["similarity_score"]
                        if current_score >= last_score * 0.98:  # 2% drop tolerance
                            final_chunks.append(chunk)
                            last_score = current_score
                        else:
                            break

                        if len(final_chunks) >= top_k * 2:
                            break
            else:
                final_chunks = ranked_chunks[:top_k]

            self.logger.info(f"Retrieved {len(final_chunks)} chunks (top_k={top_k}, dynamic={dynamic_k})")

            return {
                "query": new_query,
                "rewritten_queries": queries,
                "chunks": final_chunks,
                "num_results": len(final_chunks),
                "search_metadata": {
                    "search_time": search_result.get("search_time", 0),
                    "requested_top_k": top_k,
                    "dynamic_k_enabled": dynamic_k,
                    "returned_results": len(final_chunks),
                },
            }

        except Exception as e:
            self.logger.error(f"Retrieval error: {str(e)}")
            raise ValueError("Unable to retrieve the data.") from e
