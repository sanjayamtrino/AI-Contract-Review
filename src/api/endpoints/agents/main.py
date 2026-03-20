import io

from docx import Document
from fastapi import APIRouter, UploadFile

from src.tools.comparision import compare_doc_versions

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/compare-documents")
async def compare_documents(doc1: UploadFile, doc2: UploadFile):
    """Endpoint to compare two document versions."""

    doc1_obj = Document(io.BytesIO(await doc1.read()))
    doc2_obj = Document(io.BytesIO(await doc2.read()))

    result = await compare_doc_versions(doc1_obj, doc2_obj)
    return result
