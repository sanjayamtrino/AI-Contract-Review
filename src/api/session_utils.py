"""FastAPI dependencies for extracting session ID from request headers."""

from typing import Optional

from fastapi import Header, HTTPException

from src.api.context import set_session_id


async def get_session_id(x_session_id: Optional[str] = Header(None)) -> str:
    """Extract and validate the required X-Session-ID header."""
    if not x_session_id or not x_session_id.strip():
        raise HTTPException(status_code=400, detail="Missing required header: X-Session-ID")

    session_id = x_session_id.strip()
    set_session_id(session_id)
    return session_id


async def get_optional_session_id(x_session_id: Optional[str] = Header(None)) -> Optional[str]:
    """Extract the optional X-Session-ID header."""
    if x_session_id:
        session_id = x_session_id.strip()
        set_session_id(session_id)
        return session_id
    return None
