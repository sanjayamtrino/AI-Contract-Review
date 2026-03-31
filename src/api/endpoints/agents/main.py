import io

from docx import Document
from fastapi import APIRouter, Depends, UploadFile
from fastapi.responses import HTMLResponse

from src.api.session_utils import get_session_id
from src.schemas.tool_schema import NDAGenerationRequest
from src.tools.comparision import compare_doc_versions
from src.tools.nda_generation import generate_heading_description, generate_nda_headings

router = APIRouter(tags=["agents"])


@router.post("/version-comparison")
async def compare_documents(doc1: UploadFile, doc2: UploadFile):
    """Endpoint to compare two document versions."""

    doc1_obj = Document(io.BytesIO(await doc1.read()))
    doc2_obj = Document(io.BytesIO(await doc2.read()))

    result = await compare_doc_versions(doc1_obj, doc2_obj)
    return result


@router.post("/generate-nda-headings")
async def generate_nda_endpoint(request: NDAGenerationRequest, session_id: str = Depends(get_session_id)):
    """Stepwise NDA generation endpoint."""

    response = await generate_nda_headings(request, session_id)
    return HTMLResponse(response, status_code=200)


@router.post("/generate-nda-content")
async def generate_nda_content(request: NDAGenerationRequest, session_id: str = Depends(get_session_id)):
    """Generate content for a specific NDA heading based on type of document."""

    response = await generate_heading_description(request)
    return HTMLResponse(response, status_code=200)
