import io
from typing import Any

from docx import Document
from fastapi import APIRouter, Depends, UploadFile

from src.api.session_utils import get_session_id
from src.schemas.doc_chat import DocChatResponse
from src.schemas.general_review import GeneralReviewRequest, GeneralReviewResponse
from src.schemas.playbook_review import PlayBookReviewFinalResponse, RuleCheckRequest
from src.tools.comparision import run as compare_documents_service
from src.tools.doc_chat import query_document as query_document_service
from src.tools.general_review import general_review as general_review_service
from src.tools.playbook_review import review_document as playbook_review_service

router = APIRouter(tags=["agents"])


@router.post("/compare-documents")
async def compare_documents_endpoint(file_a: UploadFile, file_b: UploadFile) -> Any:
    """Compare two documents and return their differences."""

    document_a = Document(io.BytesIO(await file_a.read()))
    document_b = Document(io.BytesIO(await file_b.read()))

    comparison_result = await compare_documents_service(document_a, document_b)
    return comparison_result


@router.post("/playbook-review", response_model=PlayBookReviewFinalResponse)
async def playbook_review_endpoint(request: RuleCheckRequest) -> PlayBookReviewFinalResponse:
    """Run playbook validation checks."""

    review_result = await playbook_review_service(request)
    return review_result


@router.post("/general-review", response_model=GeneralReviewResponse)
async def general_review_endpoint(request: GeneralReviewRequest) -> GeneralReviewResponse:
    """Run general-purpose rule-based review."""

    review_result = await general_review_service(request)
    return review_result


@router.post("/query-document", response_model=DocChatResponse)
async def query_document_endpoint(query: str, session_id: str = Depends(get_session_id)) -> DocChatResponse:
    """Query the document chunks based on the given query and session ID."""

    llm_result = await query_document_service(query=query, session_id=session_id)
    return llm_result


# @router.post("/generate-nda-headings")
# async def generate_nda_endpoint(request: NDAGenerationRequest, session_id: str = Depends(get_session_id)):
#     """Stepwise NDA generation endpoint."""

#     response = await generate_nda_headings(request, session_id)
#     return HTMLResponse(response, status_code=200)


# @router.post("/generate-nda-content")
# async def generate_nda_content(request: NDAGenerationRequest, session_id: str = Depends(get_session_id)):
#     """Generate content for a specific NDA heading based on type of document."""

#     response = await generate_heading_description(request)
#     return HTMLResponse(response, status_code=200)
