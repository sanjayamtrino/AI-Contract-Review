from typing import Optional

from fastapi import Header, HTTPException


async def get_session_id(x_session_id: Optional[str] = Header(None)) -> str:
    """Extract session_id from request header."""

    if not x_session_id or not x_session_id.strip():
        raise HTTPException(status_code=400, detail="Missing required header: X-Session-ID")

    return x_session_id.strip()


async def get_optional_session_id(x_session_id: Optional[str] = Header(None)) -> Optional[str]:
    """Extract session_id from request header (optional)."""

    if x_session_id:
        return x_session_id.strip()
    return None
