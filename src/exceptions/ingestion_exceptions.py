from src.exceptions.base_exception import AppException


class IngestionException(AppException):
    """Base exception for ingestion-related failures."""


class FileReadException(IngestionException):
    """Raised when uploaded file cannot be read."""


class UnsupportedFileTypeException(IngestionException):
    """Raised when file type is not supported."""


class IngestionProcessingException(IngestionException):
    """Raised when ingestion pipeline fails."""


class ParserNotFoundException(AppException):
    def __init__(self, message: str = "No parser registered for given document type"):
        super().__init__(message, code="PARSER_NOT_FOUND")


class DocumentLoadException(AppException):
    def __init__(self, message: str):
        super().__init__(message, code="DOCUMENT_LOAD_FAILED")
