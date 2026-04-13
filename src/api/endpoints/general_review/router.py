"""General Review endpoint — standalone entry point for the general review agent."""

import logging

from fastapi import APIRouter, Body, Depends, HTTPException

from src.agents.general_review import run as run_general_review_agent
from src.api.session_utils import get_session_id
from src.schemas.general_review import GeneralReviewRequest, GeneralReviewResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["General Review"])


@router.post("/review", response_model=GeneralReviewResponse)
async def review_contract(
    request: GeneralReviewRequest = Body(...),
    session_id: str = Depends(get_session_id),
) -> GeneralReviewResponse:
    """Run the general review agent against an ingested document.

    Two modes:
    - **Clause review**: provide ``selected_clause`` (and optionally
      ``clause_title``) to review a specific clause.
    - **Full document review**: omit ``selected_clause`` to review the
      entire contract.

    Requires a valid session with an already-ingested document (via
    ``POST /api/v1/ingest/``). Pass the session ID in the ``X-Session-ID``
    header.
    """
    try:
        return await run_general_review_agent(
            prompt=request.prompt,
            session_id=session_id,
            selected_clause=request.selected_clause,
            clause_title=request.clause_title,
        )
    except ValueError as err:
        logger.warning("General review failed (bad input): %s", err)
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:
        logger.exception("General review failed unexpectedly")
        raise HTTPException(status_code=500, detail=f"General review error: {str(err)}")
