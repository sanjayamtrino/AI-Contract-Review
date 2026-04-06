"""Summary tool — generates a document summary from ingested chunks."""

from typing import Optional

from src.dependencies import get_service_container
from src.schemas.tool_schema import SummaryToolResponse
from src.services.prompts.v1 import load_prompt
from src.services.vector_store.manager import get_all_chunks


async def get_summary(session_id: Optional[str], response: str = "JSON") -> str:
    """Generate a summary of the ingested document.

    Uses session-specific chunks when available, falls back to global store.
    """
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
        results = get_all_chunks()

    if not results:
        raise ValueError("No document ingested. Please ingest a document first.")

    full_text = "\n\n".join(
        chunk.content for chunk in results.values()
        if getattr(chunk, "content", None)
    )

    prompt_template = load_prompt("summary_prompt_template")
    summary: SummaryToolResponse = await container.azure_openai_model.generate(
        prompt=prompt_template, context={"text": full_text}, response_model=SummaryToolResponse
    )

    return summary.summary if response == "JSON" else summary
