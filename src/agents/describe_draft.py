"""Describe & Draft Agent -- generates clause/agreement drafts in 3 styles."""

from typing import Optional

from src.schemas.draft import DraftResponse
from src.tools.drafter import generate_drafts


async def run(
    session_id: Optional[str],
    user_prompt: str,
) -> DraftResponse:
    """Generate 3 draft alternatives from a user description."""
    return await generate_drafts(
        session_id=session_id,
        user_prompt=user_prompt,
    )
