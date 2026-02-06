import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.services.llm.azure_openai_model import AzureOpenAIModel
from src.services.vector_store.manager import get_all_chunks

llm_service = AzureOpenAIModel()


class DummyResponse(BaseModel):

    summary: str = Field(..., description="Brief summary of the doc.")


# async def get_summary() -> Any:
#     """Summary tool for the orchestrator agent."""

#     results = get_all_chunks()
#     full_text = "\n\n".join((chunk.content for chunk in results.values() if getattr(chunk, "content", None)))
#     print(full_text)

#     prompt_template = Path(r"src\services\prompts\v1\summary_prompt_template.mustache").read_text()
#     context = {"text": full_text}

#     summary = await llm_service.generate(prompt=prompt_template, context=context, response_model=DummyResponse)

#     return summary

def get_summary() -> str:
    """Summary tool for the orchestrator agent."""

    return "This is a dummy summary of the document."


def get_location() -> str:
    """Location tool for retrieveing the current location."""

    return "Hyderabad, Telangana"


def get_key_information() -> str:
    """Returns key information."""

    return "Key Information regarding the docs."


async def main() -> Any:
    await get_summary()


if __name__ == "__main__":
    asyncio.run(main())
