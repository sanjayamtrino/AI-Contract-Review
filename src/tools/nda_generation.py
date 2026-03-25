from pathlib import Path
from typing import Optional

from src.api.context import get_session_id
from src.dependencies import get_service_container
from src.schemas.tool_schema import NDAGenerationRequest


async def generate_nda_headings(request: NDAGenerationRequest, session_id: Optional[str] = None) -> str:
    """Generate NDA headings."""

    container = get_service_container()
    llm_model = container.azure_openai_model

    # Get session_id from context if not provided
    if not session_id:
        session_id = get_session_id()

    session = None
    if session_id:
        try:
            session = container.session_manager.get_or_create_session(session_id)
        except Exception:
            session = None

    # Check if NDA generation already exists in session cache
    cache_key = "nda_generation"
    if session and cache_key in session.tool_results:
        return session.tool_results[cache_key]

    # step = _validate_step(request.step)
    prompt = Path(r"src\services\prompts\v1\nda_generation.mustache").read_text(encoding="utf-8")

    context = {
        "nda_description": request.nda_description,
    }

    generated_text: str = await llm_model.generate(prompt=prompt, context=context, response_model=None, mode="markdown")

    # Store the result in session cache if session exists
    if session:
        session.tool_results[cache_key] = generated_text

    return generated_text
    # return NDAGenerationResponse(generated_text=generated_text)


async def generate_heading_description(request: NDAGenerationRequest) -> str:
    """Generate content for a specific NDA heading based on type of document."""

    # container = get_service_container()
    # llm_model = container.azure_openai_model

    # # Get session_id from context if not provided
    # if not session_id:
    #     session_id = get_session_id()

    # session = None
    # if session_id:
    #     try:
    #         session = container.session_manager.get_or_create_session(session_id)
    #     except Exception:
    #         session = None

    # # Check if NDA generation already exists in session cache
    # cache_key = "nda_generation_heading"
    # if session and cache_key in session.tool_results:
    #     return session.tool_results[cache_key]

    # This function can be used to generate individual headings if needed
    prompt = Path(r"src\services\prompts\v1\nda_description_prompt.mustache").read_text(encoding="utf-8")

    context = {
        "nda_description": request.nda_description,
        "heading": request.headings,
    }

    generated_heading: str = await get_service_container().azure_openai_model.generate(prompt=prompt, context=context, response_model=None, mode="markdown")

    return generated_heading
