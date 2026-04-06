"""Compare endpoint — clause-by-clause diff between two contract documents."""

from fastapi import APIRouter

from src.agents.compare import run
from src.schemas.compare import CompareRequest, CompareResponse

router = APIRouter()


@router.post("/diff")
async def compare_documents(request: CompareRequest) -> CompareResponse:
    """Compare two contract documents within the same session.

    Returns a clause-level diff with change classification,
    risk assessment, and legal implications.
    """
    return await run(
        session_id=request.session_id,
        document_id_a=request.document_id_a,
        document_id_b=request.document_id_b,
    )
