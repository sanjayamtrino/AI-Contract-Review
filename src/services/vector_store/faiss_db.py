from typing import Any, Dict

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
