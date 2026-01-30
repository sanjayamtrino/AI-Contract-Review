from threading import Lock
from typing import Optional

from src.config.logging import get_logger
from src.services.vector_store.faiss_db import FAISSVectorStore

_lock = Lock()
_instance: Optional[FAISSVectorStore] = None
_instance_dimension: Optional[int] = None


def get_faiss_vector_store(embedding_dimension: int) -> FAISSVectorStore:
    """Singleton FAISSVectorStore instance. If an instance already exists,
    the existing instance is returned. If the requested `embedding_dimension`
    differs from the already-created store, the existing instance will be
    used and a warning logged."""

    global _instance, _instance_dimension

    logger = get_logger("FAISSVectorStoreManager")

    with _lock:
        if _instance is None:
            logger.info(f"Creating FAISSVectorStore with dimension={embedding_dimension}")
            _instance = FAISSVectorStore(embedding_dimension=embedding_dimension)
            _instance_dimension = embedding_dimension
            return _instance

        if _instance_dimension != embedding_dimension:
            logger.warning(f"Requested FAISS dimension {embedding_dimension} does not match existing " f"dimension {_instance_dimension}. Using existing instance.")

        return _instance
