import asyncio
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.dependencies import get_service_container
from src.services.llm.azure_openai_model import AzureOpenAIModel
from src.services.vector_store.manager import get_all_chunks

llm_service = AzureOpenAIModel()


class DummyResponse(BaseModel):

    summary: str = Field(..., description="Brief summary of the doc.")


async def get_summary(session_id: Optional[str] = None) -> Any:
    """Summary tool for the orchestrator agent or API."""

    # Prefer session-specific chunks when session_id provided
    if session_id:
        container = get_service_container()
        try:
            session = container.session_manager.get_session(session_id)
        except Exception:
            session = None

        if not session:
            raise ValueError(f"Session '{session_id}' not found or expired")

        results = session.chunk_store
    else:
        results = get_all_chunks()

    full_text = "\n\n".join((chunk.content for chunk in results.values() if getattr(chunk, "content", None)))

    prompt_template = Path(r"src\services\prompts\v1\summary_prompt_template.mustache").read_text()
    context = {"text": full_text}

    summary = await llm_service.generate(prompt=prompt_template, context=context, response_model=DummyResponse)

    return summary


async def get_location() -> str:
    """Location tool for retrieveing the current location."""

    return "Hyderabad, Telangana"


def get_key_information() -> str:
    """Returns key information."""

    return "Key Information regarding the docs."


async def main() -> Any:
    await get_summary()


if __name__ == "__main__":
    asyncio.run(main())
