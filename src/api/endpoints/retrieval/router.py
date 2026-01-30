from fastapi import APIRouter

from src.services.retrieval.retrieval import RetrievalService

retrieval_service = RetrievalService()

router = APIRouter()


@router.post("/query/")
async def query_document(query: str) -> None:
    await retrieval_service.retrieve_data(query=query)
