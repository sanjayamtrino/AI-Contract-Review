from src.exceptions.base_exception import AppException


class ParserAlreadyRegistered(AppException):
    """Raised when trying to register an existing parser."""


class ParserNotRegistered(AppException):
    """Raised when requested parser is missing."""
