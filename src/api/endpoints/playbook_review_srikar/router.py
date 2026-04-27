from fastapi import APIRouter, Header

from src.schemas.playbook_review_srikar import (
    PlayBookReviewFinalResponse,
    RuleCheckRequest,
)
from src.tools.playbook_review_srikar import review_document

router = APIRouter(tags=["agents"])


@router.post("/playbook-review-srikar", response_model=PlayBookReviewFinalResponse)
async def playbook_review_srikar_endpoint(
    request: RuleCheckRequest,
    session_id: str = Header(..., alias="X-Session-Id"),
) -> PlayBookReviewFinalResponse:
    return await review_document(session_id=session_id, request=request)
