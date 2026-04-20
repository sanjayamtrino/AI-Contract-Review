"""
Describe & Draft Agent — thin dispatcher.

Delegates all logic to src/tools/drafter.py. No business logic here.
"""
from src.schemas.describe_draft import DescribeDraftResponse
from src.tools.drafter import generate_describe_draft


async def run(
    prompt: str,
    session_id: str,
    regenerate: bool = False,
) -> DescribeDraftResponse:
    """Run the describe-and-draft agent for a session."""
    return await generate_describe_draft(
        prompt=prompt,
        session_id=session_id,
        regenerate=regenerate,
    )
