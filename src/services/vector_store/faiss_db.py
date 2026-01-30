import time
from typing import Any, Dict, List

import faiss
import numpy as np

from src.config.logging import Logger


class FAISSVectorStore(Logger):
    def __init__(self, embedding_dimention: int) -> None:
        super().__init__()
        self.dimension = embedding_dimention

        self.index = faiss.IndexFlatIP(self.dimension)

        self.stats: Dict[str, Any] = {
            "vectors_added": 0,
            "search_requests": 0,
            "total_add_time": 0.0,
            "total_search_time": 0.0,
        }

    def _validate_vectors(self, vectors: np.ndarray) -> np.ndarray:
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {vectors.shape[1]}")
        faiss.normalize_L2(vectors)
        return vectors

    def add(self, embeddings: List[List[float]]) -> None:
        if not embeddings:
            raise ValueError("Embeddings list cannot be empty.")

        vectors = np.array(embeddings, dtype=np.float32)

        try:
            start_time = time.time()
            vectors = self._validate_vectors(vectors)
            self.index.add(vectors)
            elapsed_time = time.time() - start_time

            self.stats["vectors_added"] += vectors.shape[0]
            self.stats["total_add_time"] += elapsed_time

            self.logger.info(f"Added {vectors.shape[0]} vector(s) in {elapsed_time:.2f}s")
        except Exception as e:
            raise ValueError("Unable to index embeddings into database.") from e
