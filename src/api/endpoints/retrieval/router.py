from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.dependencies import get_service_container
from src.tools.summarizer import get_summary

router = APIRouter()


class Dummy(BaseModel):
    """Nothing"""

    response: str = Field(..., description="Detailed reponse for the given user query.")


prompt_template = Path(r"src\services\prompts\v1\llm_response.mustache").read_text(encoding="utf-8")


@router.post("/query/")
async def query_document(query: str) -> None:
    # Get services from the dependency container
    service_container = get_service_container()
    retrieval_service = service_container.retrieval_service
    azure_model = service_container.azure_openai_model

    result = await retrieval_service.retrieve_data(query=query)

    data: Dict[str, Any] = {
        "context": result["chunks"],
        "question": query,
    }

    llm_result = await azure_model.generate(prompt=prompt_template, context=data, response_model=Dummy)
    print(llm_result, result["chunks"])
    return {
        "llm_result": llm_result,
        "retrieved chunks": result,
    }


@router.get("/summarizer")
async def get_chunks():
    result = await get_summary()
    return result
