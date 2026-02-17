from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from src.dependencies import get_service_container
from src.schemas.rule_check import RuleCheckRequest, RuleCheckResponse

router = APIRouter()

# Load prompt template (consider caching this or loading it once)
PROMPT_TEMPLATE_PATH = Path(r"src\services\prompts\v1\rule_check.mustache")


@router.post("/AIContractReview", response_model=RuleCheckResponse)
async def check_rules(request: RuleCheckRequest) -> RuleCheckResponse:
    """
    Check text paragraphs against a list of rules using an LLM.
    """
    service_container = get_service_container()
    azure_model = service_container.azure_openai_model

    if not PROMPT_TEMPLATE_PATH.exists():
        raise HTTPException(status_code=500, detail="Prompt template not found")

    prompt_template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")

    # Prepare context for the template
    context = {"rules": [rule.model_dump() for rule in request.rulesinformation], "paragraphs": [text.model_dump() for text in request.textinformation]}

    try:
        # Generate response using LLM
        # We pass the schema of RuleCheckResponse to the LLM to enforce structure
        response = await azure_model.generate(prompt=prompt_template, context=context, response_model=RuleCheckResponse)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing rule check: {str(e)}")
