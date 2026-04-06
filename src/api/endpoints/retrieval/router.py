"""Retrieval endpoints — query documents, run playbook reviews, and general reviews."""

from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.session_utils import get_session_id
from src.dependencies import get_service_container
from src.schemas.llm_response import QueryLLmResponse
from src.schemas.rule_check import (
    PlaybookAnalysisResponse,
    PlayBookReviewLLMResponse,
    PlayBookReviewRequest,
    PlayBookReviewResponse,
    RuleCheckRequest,
    RuleResult,
)
from src.services.retrieval.rules_batching import get_matching_paras

router = APIRouter()

llm_prompt_template = Path(r"src\services\prompts\v1\llm_response.mustache").read_text(encoding="utf-8")
similarity_prompt_template = Path(r"src\services\prompts\v1\ai_review_prompt_v2.mustache").read_text(encoding="utf-8")


@router.post("/query/")
async def query_document(query: str, top_k: int = 5, dynamic_k: bool = False, session_id: str = Depends(get_session_id)) -> Dict[str, Any]:
    """Retrieve relevant chunks and generate an LLM response for a user query."""
    service_container = get_service_container()
    azure_model = service_container.azure_openai_model
    session_manager = service_container.session_manager
    retrieval_service = service_container.retrieval_service

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

    return {
        "llm_result": llm_result,
        "retrieved_chunks": result,
        "session_id": session_id,
    }


@router.post("/playbook/statistical-review", response_model=List[RuleResult])
async def statistical_review(request: RuleCheckRequest) -> List[RuleResult]:
    """Retrieve similar paragraphs for each rule without LLM evaluation."""
    return await get_matching_paras(request=request)


@router.post("/playbook/ai-review", response_model=List[PlayBookReviewResponse])
async def llm_review(request: RuleCheckRequest) -> List[PlayBookReviewResponse]:
    """Retrieve paragraphs and run LLM evaluation for each rule."""
    service_container = get_service_container()
    llm_service = service_container.azure_openai_model

    response: List[RuleResult] = await get_matching_paras(request=request)

    result: List[PlayBookReviewResponse] = []
    for res in response:
        context: Dict[str, Any] = {
            "rule_title": res.title,
            "rule_instruction": res.instruction,
            "rule_description": res.description,
            "paragraphs": res.paragraphcontext,
        }

        llm_response: PlayBookReviewLLMResponse = await llm_service.generate(
            prompt=similarity_prompt_template, context=context, response_model=PlayBookReviewLLMResponse
        )
        result.append(
            PlayBookReviewResponse(
                rule_title=res.title,
                rule_instruction=res.instruction,
                rule_description=res.description,
                content=llm_response,
            )
        )

    return result


@router.post("/playbook/test-ai")
async def review(request: RuleCheckRequest) -> Any:
    """Master playbook review — all rules evaluated in a single LLM call."""
    service_container = get_service_container()
    llm_service = service_container.azure_openai_model

    prompt = Path(r"src\services\prompts\v1\master_playbook_review_prompt.mustache").read_text()
    rule_results: List[RuleResult] = await get_matching_paras(request=request)

    formatted_rules = [
        {
            "title": rule.title,
            "instruction": rule.instruction,
            "description": rule.description,
            "paragraph_id": rule.paragraphidentifier,
            "paragraph_text": rule.paragraphcontext,
        }
        for rule in rule_results
    ]

    context: Dict[str, Any] = {"total_rules": len(formatted_rules), "rules": formatted_rules}
    return await llm_service.generate(prompt=prompt, context=context, response_model=PlaybookAnalysisResponse)


@router.post("/playbook/ai-rule-Review", response_model=PlayBookReviewResponse)
async def review_rules(request: PlayBookReviewRequest) -> PlayBookReviewResponse:
    """Evaluate a single rule against provided paragraphs."""
    service_container = get_service_container()
    llm_service = service_container.azure_openai_model

    context: Dict[str, Any] = {
        "rule_title": request.rule_title,
        "rule_description": request.description,
        "paragraphs": " ".join(request.paragraphs),
    }

    return await llm_service.generate(
        prompt=similarity_prompt_template, context=context, response_model=PlayBookReviewResponse
    )


# --- General Review (user-provided rule) ---

class GeneralReviewFix(BaseModel):
    original_text: str
    fixed_text: str
    fix_summary: str


class GeneralReviewLLMResponse(BaseModel):
    reason: str
    suggested_fix: List[GeneralReviewFix]


class GeneralReviewRequest(BaseModel):
    paragraph: str
    rule: str


class GeneralReviewResponse(BaseModel):
    paragraph: str
    rule: str
    reason: str
    suggested_fix: List[GeneralReviewFix]


@router.post("/general-review/", response_model=GeneralReviewResponse)
async def general_review(request: GeneralReviewRequest) -> GeneralReviewResponse:
    """Check a paragraph against a user-provided rule or guideline."""
    service_container = get_service_container()
    llm_service = service_container.azure_openai_model

    prompt = Path(r"src\services\prompts\v1\general_review.mustache").read_text(encoding="utf-8")
    context = {"paragraph": request.paragraph, "rule": request.rule}

    llm_result = await llm_service.generate(prompt=prompt, context=context, response_model=GeneralReviewLLMResponse)

    return GeneralReviewResponse(
        paragraph=request.paragraph,
        rule=request.rule,
        reason=llm_result.reason,
        suggested_fix=llm_result.suggested_fix,
    )
