from threading import Lock
from typing import Dict, List, Optional

from src.config.logging import get_logger
from src.schemas.registry import Chunk
from src.services.vector_store.faiss_db import FAISSVectorStore

_lock = Lock()
_instance: Optional[FAISSVectorStore] = None
_instance_dimension: Optional[int] = None
_chunk_store: Dict[int, Chunk] = {}
_chunk_counter: int = 0


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


def index_chunks(chunks: List[Chunk]) -> None:
    """Index a list of chunks into the shared chunk store."""

    global _chunk_counter

    logger = get_logger("FAISSVectorStoreManager")

    with _lock:
        for chunk in chunks:
            _chunk_store[_chunk_counter] = chunk
            _chunk_counter += 1

        logger.info(f"Indexed {len(chunks)} chunks. Total chunks in store: {len(_chunk_store)}")


def get_chunk(chunk_index: int) -> Optional[Chunk]:
    """Retrieve a chunk by its index from the shared store."""
    with _lock:
        return _chunk_store.get(chunk_index)


def get_chunks(chunk_indices: List[int]) -> List[Chunk]:
    """Retrieve multiple chunks by their indices."""
    with _lock:
        return [_chunk_store[idx] for idx in chunk_indices if idx in _chunk_store]


def get_all_chunks() -> Dict[int, Chunk]:
    """Retrieve all indexed chunks."""
    with _lock:
        return _chunk_store.copy()


def reset_chunks() -> None:
    """Clear all indexed chunks and reset counter."""
    global _chunk_counter

    logger = get_logger("FAISSVectorStoreManager")

    with _lock:
        _chunk_store.clear()
        _chunk_counter = 0
        logger.info("Chunk store reset.")
