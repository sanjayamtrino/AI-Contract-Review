from typing import List

from src.config.logging import Logger


class TextChunker(Logger):
    """
    Splits text into ordered chunks.
    """

    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) List[str]:
        """
        Chunk text with overlap.

        Args:
            text (str): Cleaned document text.
            chunk_size (int): Size of each chunk.
            overlap (int): Overlap between chunks.

        Returns:
            List[str]: Ordered list of text chunks.
        """
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap.")

        chunks: List[str] = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap

        self.logger.debug("Text chunking completed.", extra={"total_chunks": len(chunks)})

        return chunks
