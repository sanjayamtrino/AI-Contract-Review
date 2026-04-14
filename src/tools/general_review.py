from pathlib import Path

from src.dependencies import get_service_container
from src.schemas.general_review import GeneralReviewRequest, GeneralReviewResponse


async def general_review(request: GeneralReviewRequest) -> GeneralReviewResponse:
    """General review function for any custom review between rules and paras."""

    service_container = get_service_container()
    llm_service = service_container.azure_openai_model
    # faiss_store = service_container.faiss_store

    prompt = Path(r"src\services\prompts\v1\general_review.mustache").read_text()

    context = {
        "paragraph": request.paragraph,
        "rule": request.rule,
    }

    response: GeneralReviewResponse = await llm_service.generate(prompt=prompt, context=context, response_model=GeneralReviewResponse)

    return response
