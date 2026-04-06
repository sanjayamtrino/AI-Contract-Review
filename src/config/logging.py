"""
Structured logging configuration with session context propagation.

Provides console, file, and error-file handlers with rotating log files.
Session IDs are injected into every log record via ContextualFilter.
"""

import logging
import logging.config
import os
from datetime import datetime
from typing import Any, Dict

from src.config.settings import get_settings


class ContextualFilter(logging.Filter):
    """Injects session_id into log records from request context."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from src.api.context import get_session_id
            record.session_id = get_session_id() or "-"
        except (ImportError, RuntimeError):
            record.session_id = "-"
        return True


def setup_logging() -> None:
    """Configure application logging with file rotation and structured output."""
    settings = get_settings()
    os.makedirs(settings.logs_directory, exist_ok=True)

    log_filename = os.path.join(
        settings.logs_directory,
        f"AI_Contract_Review_{datetime.now().strftime('%Y%m%d')}.log",
    )

    logging_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "contextual": {"()": ContextualFilter},
        },
        "formatters": {
            "detailed": {
                "format": (
                    "%(asctime)s - %(name)s - %(levelname)s - "
                    "[session:%(session_id)s] - "
                    "%(filename)s:%(lineno)d - %(funcName)s - %(message)s"
                )
            },
            "simple": {
                "format": "%(asctime)s - %(levelname)s - [%(session_id)s] - %(message)s"
            },
            "json": {
                "format": (
                    '{"timestamp": "%(asctime)s", "session_id": "%(session_id)s", '
                    '"document_id": "%(document_id)s", "request_id": "%(request_id)s", '
                    '"logger": "%(name)s", "level": "%(levelname)s", "file": "%(filename)s", '
                    '"line": %(lineno)d, "function": "%(funcName)s", '
                    '"message": "%(message)s"}'
                )
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
                "filters": ["contextual"],
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": log_filename,
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "filters": ["contextual"],
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "json",
                "filename": os.path.join(settings.logs_directory, "errors.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "filters": ["contextual"],
            },
        },
        "loggers": {
            "contract_review": {
                "level": "DEBUG" if settings.debug else "INFO",
                "handlers": ["console", "file", "error_file"],
                "propagate": False,
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False,
            },
        },
        "root": {"level": "INFO", "handlers": ["console", "file"]},
    }

    logging.config.dictConfig(logging_config)


def get_logger(name: str) -> logging.Logger:
    """Get a namespaced logger under the AI_Contract hierarchy."""
    return logging.getLogger(f"AI_Contract.{name}")


class Logger:
    """Mixin that provides a class-level logger property."""

    @property
    def logger(self) -> logging.Logger:
        return get_logger(self.__class__.__name__)
