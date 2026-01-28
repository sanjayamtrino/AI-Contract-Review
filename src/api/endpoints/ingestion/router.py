from io import BytesIO

from fastapi import APIRouter, HTTPException, UploadFile, status

from src.exceptions.base_exception import AppException
from src.schemas.registry import ParseResult
from src.services.ingestion.ingestion import IngestionService

router = APIRouter()
ingestion_service = IngestionService()


@router.post("/ingest/", response_model=ParseResult)
async def ingest_data(file: UploadFile) -> ParseResult:
    try:
        contents = await file.read()
        file_like = BytesIO(contents)
        return await ingestion_service._parse_data(data=file_like)

    except AppException as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": exc.message,
                "code": exc.code,
            },
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected internal error",
        ) from exc
