from docx.document import Document

from src.config.logging import Logger


class DocumentCleaner(Logger):
    """
    Cleans a DOCX document in memory.
    """

    async def clean(self, document: Document) -> None:
        """
        Normalize document content by removing extra whitespace.

        Args:
            document (Document): Loaded DOCX document.
        """
        try:
            for paragraph in document.paragraphs:
                if paragraph.text:
                    paragraph.text = " ".join(paragraph.text.split())

            self.logger.debug("Document cleaned successfully.")
        except Exception as exc:
            self.logger.error("Document cleaning failed.", extra={"reason": str(exc)})
            raise
