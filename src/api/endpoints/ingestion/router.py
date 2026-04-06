"""Ingestion endpoints — upload files or JSON data for parsing and indexing."""

from io import BytesIO
from typing import List

from fastapi import APIRouter, Depends, UploadFile

from src.api.session_utils import get_session_id
from src.dependencies import get_service_container
from src.schemas.registry import ParseResult
from src.schemas.rule_check import TextInfo

router = APIRouter()


@router.post("/ingest/")
async def ingest_data(file: UploadFile, session_id: str = Depends(get_session_id)) -> ParseResult:
    """Upload and ingest a DOCX file into a session's vector store."""
    contents = await file.read()
    file_like = BytesIO(contents)

    service_container = get_service_container()
    session_data = service_container.session_manager.get_or_create_session(session_id)

    return await service_container.ingestion_service._parse_data(
        data=file_like,
        session_data=session_data,
    )


@router.post("/ingest-json/")
async def ingest_json(json_data: List[TextInfo], session_id: str = Depends(get_session_id)) -> ParseResult:
    """Ingest structured JSON paragraph data into a session's vector store."""
    service_container = get_service_container()
    session_data = service_container.session_manager.get_or_create_session(session_id)

    return await service_container.ingestion_service._parse_data(
        data=json_data,
        session_data=session_data,
    )
