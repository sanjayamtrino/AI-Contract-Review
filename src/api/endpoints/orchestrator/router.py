from typing import Any, Dict

from fastapi import APIRouter, Depends

from src.api.session_utils import get_session_id
from src.orchestrator.main import run_orchestrator

router = APIRouter()


@router.post("/orchestrator/query/")
async def orchestrator_query(
    query: str,
    session_id: str = Depends(get_session_id),
) -> Dict[str, Any]:
    """
    The orchestrator decides which tool to call
    (get_summary or get_key_details)
    based on the user's query.
    """
    result = await run_orchestrator(query=query, session_id=session_id)
    return {
        "session_id": session_id,
        "response": result["response"],
    }
