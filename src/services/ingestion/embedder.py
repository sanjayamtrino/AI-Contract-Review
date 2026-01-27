from typing import List

from src.config.logging import Logger
from src.schemas.registry import Chunk


class EmbeddingService(Logger):
    """
    Generates embeddings for document chunks.
    """

    def __init__(self, model_name: str = "text-embedding-3-large") -> None:
        self.model_name = model_name

    async def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Attach embeddings to chunks.
        """
        self.logger.info("Generating embeddings for chunks.")

        for chunk in chunks:
            #  PLACEHOLDER – replace with Azure/OpenAI later
            chunk.embedding_vector = self._fake_embedding(chunk.content)
            chunk.embedding_model = self.model_name

        return chunks

    def _fake_embedding(self, text: str) -> List[float]:
        """
        Temporary deterministic embedding for development.
        """
        return [float(ord(c)) / 1000 for c in text[:128]]
