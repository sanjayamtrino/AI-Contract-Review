from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from src.api.session_utils import get_session_id
from src.dependencies import get_service_container
from src.schemas.llm_response import QueryLLmResponse
from src.schemas.rule_check import (
    PlaybookAnalysisResponse,
    PlayBookReviewRequest,
    PlayBookReviewResponse,
    RuleCheckRequest,
    RuleResult,
)
from src.services.retrieval.rules_batching import (
    get_matching_pairs_faiss,
    get_matching_paras,
)

router = APIRouter()

llm_prompt_template = Path(r"src\services\prompts\v1\llm_response.mustache").read_text(encoding="utf-8")
similarity_prompt_template = Path(r"src\services\prompts\v1\ai_review_prompt.mustache").read_text(encoding="utf-8")


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

    response: List[RuleResult] = await get_matching_pairs_faiss(request=request)

    return response


@router.post("/playbook/ai-review", response_model=List[PlayBookReviewResponse])
async def llm_review(request: RuleCheckRequest) -> List[PlayBookReviewResponse]:
    """Playbook review endpoint to find similarity between paras and rules and return the LLM response."""

    service_container = get_service_container()
    llm_service = service_container.azure_openai_model

    response: List[RuleResult] = await get_matching_pairs_faiss(request=request)

    result: List[PlayBookReviewResponse] = []
    for res in response:
        # print(res.instruction)
        context: Dict[str, Any] = {
            "rule_title": res.title,
            "rule_instruction": res.instruction,
            "rule_description": res.description,
            "paragraphs": res.paragraphcontext,
        }

        llm_reponse = await llm_service.generate(prompt=similarity_prompt_template, context=context, response_model=PlayBookReviewResponse)
        # print("#" * 20)
        # print(llm_reponse)
        # print("#" * 20)
        result.append(llm_reponse)

    return result


@router.post("/playbook/test-ai")
async def review(request: RuleCheckRequest) -> Any:
    """Review API"""

    service_container = get_service_container()
    llm_service = service_container.azure_openai_model

    # Load master prompt template
    prompt = Path(r"src\services\prompts\v1\master_playbook_review_prompt.mustache").read_text()

    # Get matching paragraphs (List[RuleResult])
    rule_results: List[RuleResult] = await get_matching_paras(request=request)

    # Transform RuleResult list into prompt-friendly structure
    formatted_rules = []
    for rule in rule_results:
        formatted_rules.append(
            {"title": rule.title, "instruction": rule.instruction, "description": rule.description, "paragraph_id": rule.paragraphidentifier, "paragraph_text": rule.paragraphcontext}
        )

    context: Dict[str, Any] = {"total_rules": len(formatted_rules), "rules": formatted_rules}

    # Send to LLM
    response = await llm_service.generate(prompt=prompt, context=context, response_model=PlaybookAnalysisResponse)

    return response


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
