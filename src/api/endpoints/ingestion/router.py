from io import BytesIO

from fastapi import APIRouter, UploadFile

from src.schemas.registry import ParseResult
from src.services.ingestion.ingestion import IngestionService

ingestion_service = IngestionService()

router = APIRouter()


@router.post("/ingest/")
async def ingest_data(file: UploadFile) -> ParseResult:
    """Ingest the provided file data."""

    contents = await file.read()
    file_like = BytesIO(contents)

    return await ingestion_service._parse_data(data=file_like)
