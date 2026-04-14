"""Playbook Review endpoint — standalone entry point for the playbook review agent."""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.session_utils import get_session_id
from src.agents.playbook_review import run as run_playbook_agent
from src.schemas.agents import AgentResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Agents"])


@router.post("/review")
async def review_contract(
    query: str = Query(
        default="Review this contract",
        description="What to review — e.g. 'Review this contract' or 'Check the confidentiality clause'",
    ),
    session_id: str = Depends(get_session_id),
) -> Dict[str, Any]:
    """Run the playbook review agent against an ingested document.

    Requires a valid session with an already-ingested document (via POST /api/v1/ingest/).
    Pass the session ID in the X-Session-ID header.
    """
    try:
        result: AgentResponse = await run_playbook_agent(query=query, session_id=session_id)
        return result.model_dump()
    except ValueError as err:
        logger.warning("Playbook review failed (bad input): %s", err)
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:
        logger.exception("Playbook review failed unexpectedly")
        raise HTTPException(status_code=500, detail=f"Playbook review error: {str(err)}")
