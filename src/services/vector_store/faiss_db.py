from typing import Any, Dict, List

import faiss

from src.config.logging import Logger
from src.config.settings import get_settings


class InMemoryFIASS(Logger):
    """FIASS In-Memory vector store Service."""

    def __init__(self) -> None:
        """Initialize the FIASS DB."""

        super().__init__()
        self.settings = get_settings()
        self.dimention = self.settings.db_dimention

        # Index with Inner Product (for cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimention)

        self.stats: Dict[str, Any] = {
            "vectors_added": 0,
            "search_requets": 0,
            "total_add_time": 0.0,
            "total_search_time": 0.0,
        }
        
        # Store metadata in memory (index -> metadata)
        self.metadata_store: Dict[int, Dict[str, Any]] = {}

    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        """Add vectors and metadata to the index."""
        import numpy as np
        
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors and metadata items must match.")
        
        current_count = self.index.ntotal
        
        # Convert to float32 numpy array
        vectors_np = np.array(vectors, dtype=np.float32)
        
        # Add to FAISS index
        self.index.add(vectors_np)
        
        # Store metadata
        for i, meta in enumerate(metadata):
            self.metadata_store[current_count + i] = meta
            
        self.stats["vectors_added"] += len(vectors)

    def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        import numpy as np
        
        # Convert query to numpy array
        query_np = np.array([query_vector], dtype=np.float32)
        
        # Search index
        distances, indices = self.index.search(query_np, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 indicates no found neighbor
                meta = self.metadata_store.get(idx, {})
                results.append({
                    "score": float(distances[0][i]),
                    "metadata": meta,
                    "id": int(idx)
                })
                
        self.stats["search_requets"] += 1
        return results
