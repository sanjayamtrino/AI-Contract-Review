from typing import Any, Dict

from fastapi import APIRouter

from src.orchestrator.main import run_orchestrator

router = APIRouter()


@router.post("/orchestrator/query/")
async def orchestrator_query(query: str) -> Dict[str, Any]:
    """
    The orchestrator decides which tool to call
    (get_summary, get_key_information, or get_location)
    based on the user's query.
    """
    result = await run_orchestrator(query=query)
    return result