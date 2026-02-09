from io import BytesIO

from fastapi import APIRouter, UploadFile

from src.dependencies import get_service_container
from src.schemas.registry import ParseResult

router = APIRouter()


@router.post("/ingest/")
async def ingest_data(file: UploadFile) -> ParseResult:
    """Ingest the provided file data."""

    contents = await file.read()
    file_like = BytesIO(contents)

    # Get service from the dependency container
    service_container = get_service_container()
    return await service_container.ingestion_service._parse_data(data=file_like)
