"""Missing clauses tool — identifies absent, incomplete, or ambiguous contract clauses."""

import logging

from src.dependencies import get_service_container
from src.schemas.missing_clauses import MissingClausesLLMResponse
from src.services.prompts.v1 import load_prompt

logger = logging.getLogger("AI_Contract.MissingClauses")


async def get_missing_clauses(contract_text: str) -> MissingClausesLLMResponse:
    """Analyze contract text and identify missing or deficient clauses.

    Detects the contract type automatically and audits against
    standard clauses expected for that type.
    """
    if not contract_text or not contract_text.strip():
        raise ValueError("Contract text is empty. Please provide the contract content.")

    container = get_service_container()
    logger.info(f"Analyzing {len(contract_text)} chars for missing clauses")

    prompt = load_prompt("missing_clauses_prompt")

    system_message = (
        "You are a senior legal contract analyst conducting a formal clause completeness audit. "
        "Follow EVERY instruction in the user prompt precisely. Key requirements:\n"
        "1. Report ALL missing clauses — do not stop early.\n"
        "2. NO DUPLICATE findings — merge overlapping clauses into one entry.\n"
        "3. EVERY explanation MUST name the actual parties and reference specific details.\n"
        "4. EVERY draft_clause MUST have at least 3 sub-sections (a), (b), (c) with substantive language.\n"
        "5. Return valid JSON matching the schema exactly."
    )

    return await container.azure_openai_model.generate(
        prompt=prompt,
        context={"contract_text": contract_text},
        response_model=MissingClausesLLMResponse,
        system_message=system_message,
    )
