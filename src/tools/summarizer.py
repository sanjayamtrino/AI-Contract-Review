from typing import Any, Optional

from pydantic import BaseModel, Field

from src.dependencies import get_service_container
from src.services.prompts.v1 import load_prompt
from src.services.vector_store.manager import get_all_chunks


class SummaryResponse(BaseModel):

    summary: str = Field(
        ...,
        description=(
            "A comprehensive, structured summary of the legal document in markdown format. "
            "Must include: 1) Document Type & Purpose, 2) Parties Involved, "
            "3) Key Terms & Obligations (bullet points), 4) Critical Dates & Deadlines, "
            "5) Notable Risks or Concerns. Target 150-300 words."
        ),
    )


async def get_summary(session_id: Optional[str] = None) -> Any:
    """Summary tool for the orchestrator agent. Works with session-based ingestion."""

    container = get_service_container()

    if session_id:
        try:
            session = container.session_manager.get_session(session_id)
        except Exception:
            session = None

        if not session:
            raise ValueError(f"Session '{session_id}' not found or expired")

        results = session.chunk_store
    else:
        # fallback (dev / legacy)
        results = get_all_chunks()

    if not results:
        raise ValueError("No document ingested. Please ingest a document first.")

    full_text = "\n\n".join(chunk.content for chunk in results.values() if getattr(chunk, "content", None))

    prompt_template = load_prompt("summary_prompt_template")
    context = {"context": full_text}

    summary = await container.azure_openai_model.generate(prompt=prompt_template, context=context, response_model=SummaryResponse)

    return summary
