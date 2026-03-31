import io
from typing import Any

from docx import Document
from fastapi import APIRouter, UploadFile

from src.tools.comparision import diff_paragraphs

router = APIRouter(tags=["test-agents"])


@router.post("/compare-documents")
async def statistics_compare_documents(doc1: UploadFile, doc2: UploadFile) -> Any:
    """Test endpoint to compare two document versions."""

    doc1_obj = Document(io.BytesIO(await doc1.read()))
    doc2_obj = Document(io.BytesIO(await doc2.read()))

    doc1_paras = [para.text.strip() for para in doc1_obj.paragraphs if para.text.strip()]
    doc2_paras = [para.text.strip() for para in doc2_obj.paragraphs if para.text.strip()]

    changes = await diff_paragraphs(doc1_paras, doc2_paras)

    return changes
