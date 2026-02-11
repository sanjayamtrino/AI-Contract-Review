from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.api.session_utils import get_session_id
from src.dependencies import get_service_container

# from src.tools.summarizer import get_summary

router = APIRouter()


class Dummy(BaseModel):
    """Nothing"""

    response: str = Field(..., description="Detailed reponse for the given user query.")


prompt_template = Path(r"src\services\prompts\v1\llm_response.mustache").read_text(encoding="utf-8")


@router.post("/query/")
async def query_document(
    query: str,
    session_id: str = Depends(get_session_id),
) -> Dict[str, Any]:
    """Query documents for a specific session."""

    # Get service container and session manager
    service_container = get_service_container()
    session_manager = service_container.session_manager
    retrieval_service = service_container.retrieval_service
    azure_model = service_container.azure_openai_model

    # Get session (don't create if doesn't exist - get_session instead of get_or_create)
    session_data = session_manager.get_session(session_id)
    if not session_data:
        return {"error": "Session not found. Please ingest documents first.", "session_id": session_id}

    result = await retrieval_service.retrieve_data(
        query=query,
        session_data=session_data,
    )

    data: Dict[str, Any] = {
        "context": result["chunks"],
        "question": query,
    }

    llm_result = await azure_model.generate(prompt=prompt_template, context=data, response_model=Dummy)
    # print(llm_result, result["chunks"])
    return {
        "llm_result": llm_result,
        "retrieved_chunks": result,
        "session_id": session_id,
    }


# @router.get("/summarizer")
# async def get_chunks(
#     session_id: str = Depends(get_session_id),
# ) -> Dict[str, Any]:
#     """Get summary for a specific session."""

#     try:
#         result = await get_summary(session_id=session_id)
#     except ValueError as err:
#         return {"error": str(err), "session_id": session_id}

#     return {
#         "summary": result,
#         "session_id": session_id,
#     }
