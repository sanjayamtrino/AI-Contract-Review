from fastapi import APIRouter

from src.agents.compare import run
from src.schemas.compare import CompareRequest, CompareResponse

router = APIRouter()


@router.post("/diff")
async def compare_documents(request: CompareRequest) -> CompareResponse:
    """Compare two contract versions and return structured clause-level diff."""
    try:
        result = await run(
            session_id=request.session_id,
            document_id_a=request.document_id_a,
            document_id_b=request.document_id_b,
        )
        return result
    except Exception as err:
        return CompareResponse(
            error=f"Comparison failed: {str(err)}",
            sections=[],
            metadata=None,
        )
