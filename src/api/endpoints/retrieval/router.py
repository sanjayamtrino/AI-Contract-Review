from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.session_utils import get_session_id
from src.dependencies import get_service_container
from src.schemas.llm_response import QueryLLmResponse
from src.schemas.rule_check import PlaybookAnalysisResponse, PlayBookReviewLLMResponse, PlayBookReviewRequest, PlayBookReviewResponse, ResponseStatus, RuleCheckRequest, RuleResult
from src.services.retrieval.rules_batching import get_matching_paras  # get_matching_pairs_faiss,

router = APIRouter()

llm_prompt_template = Path(r"src\services\prompts\v1\llm_response.mustache").read_text(encoding="utf-8")
similarity_prompt_template = Path(r"src\services\prompts\v1\ai_review_prompt_v3.mustache").read_text(encoding="utf-8")


@router.post("/query/")
async def query_document(
    query: str,
    top_k: int = 5,
    dynamic_k: bool = False,
    session_id: str = Depends(get_session_id),
) -> Dict[str, Any]:
    """Query documents for a specific session."""

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
async def statistical_reivew(request: RuleCheckRequest) -> List[RuleResult]:
    """Playbook review endpoint to find similarity between paras and rules only."""

    response: List[RuleResult] = await get_matching_paras(request=request)
    return response


@router.post("/playbook/ai-review", response_model=List[PlayBookReviewResponse])
async def llm_review(request: RuleCheckRequest) -> List[PlayBookReviewResponse]:

    service_container = get_service_container()
    llm_service = service_container.azure_openai_model

    response: List[RuleResult] = await get_matching_paras(request=request)

    result: List[PlayBookReviewResponse] = []

    for res in response:

        # Extract paragraph identifiers from retrieval
        para_ids = []
        if res.paragraphidentifier:
            para_ids = [p.strip() for p in res.paragraphidentifier.split(",")]

        # If no paragraphs retrieved → NOT FOUND without LLM
        if not res.paragraphcontext or not res.paragraphcontext.strip():
            result.append(
                PlayBookReviewResponse(
                    rule_title=res.title,
                    rule_instruction=res.instruction,
                    rule_description=res.description,
                    content=PlayBookReviewLLMResponse(
                        para_identifiers=para_ids,
                        status=ResponseStatus.NOT_FOUND,
                        reason=f"No clause addressing '{res.title}' was found in the document.",
                        suggestion=f"Add a '{res.title}' clause to the agreement.",
                        suggested_fix="",
                    ),
                )
            )
            continue

        context: Dict[str, Any] = {
            "RULE_TITLE": res.title,
            "RULE_INSTRUCTION": res.instruction,
            "RULE_DESCRIPTION": res.description,
            "PARAGRAPHS": res.paragraphcontext,
        }

        llm_reponse: PlayBookReviewLLMResponse = await llm_service.generate(
            prompt=similarity_prompt_template,
            context=context,
            response_model=PlayBookReviewLLMResponse,
        )

        # Ensure para_identifiers shows the paragraphs LLM checked
        llm_reponse.para_identifiers = para_ids

        result.append(
            PlayBookReviewResponse(
                rule_title=res.title,
                rule_instruction=res.instruction,
                rule_description=res.description,
                content=llm_reponse,
            )
        )

    return result


@router.post("/playbook/test-ai")
async def review(request: RuleCheckRequest) -> Any:
    """Review API"""

    service_container = get_service_container()
    llm_service = service_container.azure_openai_model

    prompt = Path(r"src\services\prompts\v1\master_playbook_review_prompt.mustache").read_text()

    rule_results: List[RuleResult] = await get_matching_paras(request=request)

    formatted_rules = []
    for rule in rule_results:
        formatted_rules.append(
            {
                "title": rule.title,
                "instruction": rule.instruction,
                "description": rule.description,
                "paragraph_id": rule.paragraphidentifier,
                "paragraph_text": rule.paragraphcontext,
            }
        )

    context: Dict[str, Any] = {"total_rules": len(formatted_rules), "rules": formatted_rules}

    response = await llm_service.generate(prompt=prompt, context=context, response_model=PlaybookAnalysisResponse)

    return response


@router.post("/playbook/ai-rule-Review", response_model=PlayBookReviewResponse)
async def review_rules(request: PlayBookReviewRequest) -> PlayBookReviewResponse:
    """Review the rule against the given paras."""

    service_container = get_service_container()
    llm_service = service_container.azure_openai_model

    # NOT FOUND immediately if no paragraphs provided
    if not request.paragraphs or not any(p.strip() for p in request.paragraphs):
        return PlayBookReviewResponse(
            rule_title=request.rule_title,
            rule_instruction=request.instruction,
            rule_description=request.description,
            content=PlayBookReviewLLMResponse(
                para_identifiers=[],
                status=ResponseStatus.NOT_FOUND,
                reason=f"No paragraphs were provided to evaluate '{request.rule_title}'.",
                suggestion="Provide the relevant contract paragraphs for evaluation.",
                suggested_fix="",
            ),
        )

    # Label each paragraph so LLM can cite by ID
    labeled_paragraphs = "\n\n".join(f"[P{str(i).zfill(4)}] {para.strip()}" for i, para in enumerate(request.paragraphs) if para.strip())

    context: Dict[str, Any] = {
        "RULE_TITLE": request.rule_title,
        "RULE_INSTRUCTION": request.instruction,
        "RULE_DESCRIPTION": request.description,
        "PARAGRAPHS": labeled_paragraphs,
    }

    llm_result = await llm_service.generate(
        prompt=similarity_prompt_template,
        context=context,
        response_model=PlayBookReviewLLMResponse,
    )

    return PlayBookReviewResponse(
        rule_title=request.rule_title,
        rule_instruction=request.instruction,
        rule_description=request.description,
        content=llm_result,
    )


class GeneralReviewResponse(BaseModel):
    reason: str
    suggested_fix: str


class GeneralReviewRequest(BaseModel):
    paragraph: str
    rule: str


@router.post("/general-review/", response_model=GeneralReviewResponse)
async def general_review(request: GeneralReviewRequest) -> GeneralReviewResponse:
    """General review endpoint for any custom review between rules and paras."""

    service_container = get_service_container()
    llm_service = service_container.azure_openai_model

    prompt = Path(r"src\services\prompts\v1\general_review.mustache").read_text()

    context = {
        "paragraph": request.paragraph,
        "rule": request.rule,
    }

    response = await llm_service.generate(prompt=prompt, context=context, response_model=GeneralReviewResponse)

    return response
