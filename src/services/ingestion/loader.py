from io import BytesIO

from docx import Document

from src.config.logging import Logger


class DocxLoader(Logger):
    """
    Loads a DOCX document into memory.
    """

    async def load_from_bytes(self, file_bytes: bytes) -> Document:
        """
        Load a DOCX document from in-memory bytes.

        Args:
            file_bytes (bytes): Raw DOCX file bytes.

        Returns:
            Document: python-docx Document object.
        """
        try:
            self.logger.info("Loading DOCX document into memory.")
            return Document(BytesIO(file_bytes))
        except Exception as exc:
            self.logger.error("Failed to load DOCX document.", extra={"reason": str(exc)})
            raise
