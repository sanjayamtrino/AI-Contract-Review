from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from src.api.session_utils import get_session_id
from src.dependencies import get_service_container
from src.schemas.llm_response import QueryLLmResponse
from src.schemas.rule_check import (
    PlayBookReviewRequest,
    PlayBookReviewResponse,
    RuleCheckRequest,
    RuleResult,
)
from src.services.retrieval.rules_batching import get_matching_paras

router = APIRouter()

llm_prompt_template = Path(r"src\services\prompts\v1\llm_response.mustache").read_text(encoding="utf-8")
similarity_prompt_template = Path(r"src\services\prompts\v1\rule_check.mustache").read_text(encoding="utf-8")


@router.post("/query/")
async def query_document(query: str, top_k: int = 5, dynamic_k: bool = False, session_id: str = Depends(get_session_id)) -> Dict[str, Any]:
    """Query documents for a specific session."""

    # Get service container and session manager
    service_container = get_service_container()
    azure_model = service_container.azure_openai_model
    session_manager = service_container.session_manager
    retrieval_service = service_container.retrieval_service

    # Get session (don't create if doesn't exist - get_session instead of get_or_create)
    session_data = session_manager.get_session(session_id)
    if not session_data:
        return {"error": "Session not found. Please ingest documents first.", "session_id": session_id}

    result = await retrieval_service.retrieve_data(
        query=query,
        top_k=top_k,
        dynamic_k=dynamic_k,
        session_data=session_data,
    )

    data: Dict[str, Any] = {
        "context": result["chunks"],
        "question": query,
    }

    llm_result = await azure_model.generate(prompt=llm_prompt_template, context=data, response_model=QueryLLmResponse)
    # print(llm_result, result["chunks"])
    return {
        "llm_result": llm_result,
        "retrieved_chunks": result,
        "session_id": session_id,
    }


@router.post("/playbook/statistical-review", response_model=List[RuleResult])
async def statistical_reivew(request: RuleCheckRequest) -> List[RuleResult]:
    """Playbook review endpoint to find similarity between paras and rules only."""

    response: List[RuleResult] = await get_matching_paras(request=request)

    return response


@router.post("/playbook/ai-review", response_model=List[PlayBookReviewResponse])
async def llm_review(request: RuleCheckRequest) -> List[PlayBookReviewResponse]:
    """Playbook review endpoint to find similarity between paras and rules and return the LLM response."""

    service_container = get_service_container()
    llm_service = service_container.azure_openai_model

    response: List[RuleResult] = await get_matching_paras(request=request)

    result: List[PlayBookReviewResponse] = []
    for res in response:
        context: Dict[str, Any] = {
            "rule_title": res.title,
            "rule_description": res.description,
            "paragraphs": res.paragraphcontext,
        }

        llm_reponse = await llm_service.generate(prompt=similarity_prompt_template, context=context, response_model=PlayBookReviewResponse)
        result.append(llm_reponse)

    return result


@router.post("/playbook/ai-rule-Review", response_model=PlayBookReviewResponse)
async def review_rules(request: PlayBookReviewRequest) -> PlayBookReviewResponse:
    """Review the rule against the given paras."""

    service_container = get_service_container()
    llm_service = service_container.azure_openai_model

    context: Dict[str, Any] = {
        "rule_title": request.rule_title,
        # "rule_instruction": request.instruction,
        "rule_description": request.description,
        "paragraphs": " ".join([para for para in request.paragraphs]),
    }

    llm_result = await llm_service.generate(prompt=similarity_prompt_template, context=context, response_model=PlayBookReviewResponse)

    return llm_result
