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
    The orchestrator decides which agent to route to
    and returns a structured OrchestratorResponse.

    Domain errors (agent failure, unknown agent, session not found) are
    returned as 200 with an error field in the body. HTTP 500 means the
    orchestrator itself is down.
    """
    result = await run_orchestrator(query=query, session_id=session_id)
    return {"session_id": session_id, **result.model_dump()}
