from src.exceptions.base_exception import AppException


class EmbeddingGenerationFailed(AppException):
    """Raised when embedding generation fails."""


class EmptyEmbeddingInput(AppException):
    """Raised when empty text is passed for embedding."""
