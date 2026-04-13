"""Draft endpoint -- generate clauses/agreements from user descriptions."""

from fastapi import APIRouter

from src.agents.describe_draft import run
from src.schemas.draft import DraftRequest, DraftResponse

router = APIRouter()


@router.post("/generate")
async def generate_draft(request: DraftRequest) -> DraftResponse:
    """Generate 3 draft alternatives (Formal, Plain English, Concise) from a user description."""
    return await run(
        session_id=request.session_id,
        user_prompt=request.user_prompt,
    )
