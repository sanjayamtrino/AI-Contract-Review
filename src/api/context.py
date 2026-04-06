"""
Request-scoped context using Python's contextvars.

Stores session_id, document_id, and request_id that automatically
propagate through async calls without explicit parameter passing.
"""

from contextvars import ContextVar
from typing import Optional

session_id_var: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
document_id_var: ContextVar[Optional[str]] = ContextVar("document_id", default=None)
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def set_session_id(session_id: Optional[str]) -> None:
    session_id_var.set(session_id)


def get_session_id() -> Optional[str]:
    return session_id_var.get()


def set_document_id(document_id: Optional[str]) -> None:
    document_id_var.set(document_id)


def get_document_id() -> Optional[str]:
    return document_id_var.get()


def set_request_id(request_id: Optional[str]) -> None:
    request_id_var.set(request_id)


def get_request_id() -> Optional[str]:
    return request_id_var.get()


def clear_context() -> None:
    """Reset all context variables after request completion."""
    session_id_var.set(None)
    document_id_var.set(None)
    request_id_var.set(None)
