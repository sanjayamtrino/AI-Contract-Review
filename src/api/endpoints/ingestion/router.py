from src.services.ingestion.ingestion import IngestionService
from fastapi import APIRouter
from src.schemas.registry import ParseResult

ingestion_service = IngestionService()

router = APIRouter() 

@router.post("/ingest/")
async def ingest_data(file_data: str) -> ParseResult:
    """Ingest the provided file data."""

    return ingestion_service._parse_data(data=file_data)