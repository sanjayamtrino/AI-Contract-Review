import io
from typing import Any

from docx import Document
from fastapi import APIRouter, UploadFile

# from src.tools.comparision import compare_docs

router = APIRouter(tags=["test-agents"])


# @router.post("/compare-documents")
# async def statistics_compare_documents(doc1: UploadFile, doc2: UploadFile) -> Any:
#     """Compare two document versions."""

#     doc1_obj = Document(io.BytesIO(await doc1.read()))
#     doc2_obj = Document(io.BytesIO(await doc2.read()))

#     # comparator = DocumentComparator()

#     # result = await comparator.compare(doc1_obj, doc2_obj)
#     result = await compare_docs(doc1_obj, doc2_obj)

#     return result
