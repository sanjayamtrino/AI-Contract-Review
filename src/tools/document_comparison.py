from pathlib import Path
from typing import Any, Dict, List

from src.config.logging import get_logger
from src.dependencies import get_service_container
from src.schemas.rule_check import DocumentComparisonResponse, TextInfo

logger = get_logger(__name__)


def _build_contract_text(paragraphs: List[TextInfo]) -> str:
    """Convert textinformation list into plain contract text for LLM."""
    return "\n\n".join(item.text.strip() for item in paragraphs if item.text.strip())


async def compare_documents(
    previous_document: List[TextInfo],
    current_document: List[TextInfo],
) -> DocumentComparisonResponse:
    """
    Compare two contract versions using LLM.

    LLM reads both documents and identifies every difference.
    """
    service_container = get_service_container()
    llm_model = service_container.azure_openai_model

    previous_text = _build_contract_text(previous_document)
    current_text = _build_contract_text(current_document)

    logger.info(f"Document comparison — " f"previous: {len(previous_document)} paragraphs, " f"current: {len(current_document)} paragraphs")

    prompt = Path(r"src\services\prompts\v1\version_compare_prompt.mustache").read_text(encoding="utf-8")

    context: Dict[str, Any] = {
        "is_verification": False,
        "document_a_text": previous_text,
        "document_b_text": current_text,
        "existing_changes": "",
    }

    response: DocumentComparisonResponse = await llm_model.generate(
        prompt=prompt,
        context=context,
        response_model=DocumentComparisonResponse,
    )

    logger.info(
        f"Comparison complete — {len(response.changes)} changes "
        f"({sum(1 for c in response.changes if c.significance == 'high')} high, "
        f"{sum(1 for c in response.changes if c.significance == 'medium')} medium, "
        f"{sum(1 for c in response.changes if c.significance == 'low')} low) | "
        f"overall risk: {response.overall_risk_impact}"
    )

    return response
