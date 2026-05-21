# from pathlib import Path

# from src.dependencies import get_service_container
# from src.schemas.describe_and_draft import DraftRequest, DraftResponse

# AGENT_NAME = "describe_and_draft"


# async def draft_document(request: DraftRequest, session_id: str) -> DraftResponse:
#     """Draft the document for the user query."""

#     container = get_service_container()
#     llm_model = container.azure_openai_model

#     session_data = container.session_manager.get_session(session_id=session_id)
#     if not session_data:
#         return {}

#     agent_results = session_data.tool_results.setdefault(AGENT_NAME, {})

#     prompt = Path(r"src\services\prompts\v1\describe_and_draft_prompt.mustache").read_text(encoding="utf-8")

#     # Build previous versions summary to inject into the prompt
#     previous_versions = {}
#     for clause_title, clause_data in agent_results.items():
#         previous_versions[clause_title] = {"summary": clause_data["summary"], "versions": clause_data["versions"]}

#     generated_content: DraftResponse = await llm_model.generate(
#         prompt=prompt,
#         context={"user_query": request.user_query, "previous_versions": previous_versions if previous_versions else ""},
#         response_model=DraftResponse,
#     )

#     for clause in generated_content.data:
#         title = clause.clause_title

#         # Initialize clause entry if it does not exist
#         if title not in agent_results:
#             agent_results[title] = {"summary": clause.summary, "versions": []}  # set only once

#         clause_entry = agent_results[title]

#         # Determine next version number
#         next_version = len(clause_entry["versions"]) + 1

#         clause_entry["versions"].append({"version": next_version, "content": clause.content})

#     return generated_content


from pathlib import Path
from typing import Optional

from src.dependencies import get_service_container
from src.schemas.describe_and_draft import (
    NDAContentGenerationRequest,
    NDAContentGenerationResponse,
    NDAGenerationHeadingRequest,
    NDAGenerationHeadingResponse,
)

AGENT_NAME = "describe_and_draft"


async def generate_nda_headings(request: NDAGenerationHeadingRequest, session_id: Optional[str] = None) -> NDAGenerationHeadingResponse:
    """Generate NDA headings."""

    container = get_service_container()
    llm_model = container.azure_openai_model

    session_data = container.session_manager.get_session(session_id)
    if not session_data:
        return NDAGenerationHeadingResponse(headings=[])

    agent_results = session_data.tool_results.setdefault(AGENT_NAME, {})

    # Return cached headings if they already exist
    if agent_results:
        return NDAGenerationHeadingResponse(headings=list(agent_results.keys()))

    # Load prompt
    prompt = Path(r"src\services\prompts\v1\nda_generation.mustache").read_text(encoding="utf-8")

    context = {
        "nda_description": request.nda_description,
    }

    # Generate headings from LLM
    generated_content: NDAGenerationHeadingResponse = await llm_model.generate(
        prompt=prompt,
        context=context,
        response_model=NDAGenerationHeadingResponse,
        mode="JSON",
    )

    # Cache generated headings
    for heading in generated_content.headings:
        agent_results[heading] = {}
        agent_results["user_input"] = request.nda_description

    return generated_content


async def generate_heading_description(request: NDAContentGenerationRequest, session_id: Optional[str] = None) -> NDAContentGenerationResponse:
    """Generate content for a specific NDA heading and store it in session."""

    container = get_service_container()
    llm_model = container.azure_openai_model

    # Get session
    session_data = container.session_manager.get_session(session_id)
    if not session_data:
        raise ValueError("Session not found")

    # Ensure agent storage exists
    agent_results = session_data.tool_results.setdefault(AGENT_NAME, {})

    # Ensure heading exists in session (user must have generated headings first)
    heading = request.heading  # assuming this is a single string

    if heading not in agent_results:
        raise ValueError(f"Heading '{heading}' not found in session.")

    if "content" in agent_results[heading]:
        return agent_results[heading]["content"]

    # Load prompt
    prompt = Path(r"src\services\prompts\v1\nda_description_prompt.mustache").read_text(encoding="utf-8")

    context = {
        "nda_description": agent_results["user_input"],
        "heading": heading,
    }

    # Generate content
    generated_heading: NDAContentGenerationResponse = await llm_model.generate(
        prompt=prompt,
        context=context,
        response_model=NDAContentGenerationResponse,
        mode="JSON",
    )

    # Store inside session under that heading
    agent_results[heading]["content"] = generated_heading.content

    return generated_heading
