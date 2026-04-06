"""Missing clauses endpoint — detect absent, incomplete, or ambiguous contract clauses."""

from typing import Any, Dict

from fastapi import APIRouter

from src.schemas.missing_clauses import MissingClausesRequest
from src.tools.missing_clauses import get_missing_clauses

router = APIRouter()


@router.post("/check")
async def check_missing_clauses(request: MissingClausesRequest) -> Dict[str, Any]:
    """Analyze contract text and identify missing, incomplete, or ambiguous clauses."""
    try:
        result = await get_missing_clauses(contract_text=request.contract_text)
    except ValueError as err:
        return {"error": str(err)}

    return {
        "contract_type": result.contract_type,
        "missing_clauses": [clause.model_dump() for clause in result.missing_clauses],
        "total_missing": result.total_missing,
        "summary": result.summary,
    }
