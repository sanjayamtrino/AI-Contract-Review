"""Key information tool — extracts structured contract details from ingested chunks."""

from typing import Optional

from pydantic import BaseModel

from src.dependencies import get_service_container
from src.schemas.tool_schema import KeyInformationToolResponse
from src.services.prompts.v1 import load_prompt
from src.services.vector_store.manager import get_all_chunks


async def get_key_information(session_id: Optional[str] = None, response_format: str = "JSON") -> str | BaseModel:
    """Extract structured key contract details (parties, dates, value, etc.).

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

    prompt_template = load_prompt("key_information_prompt")
    response: KeyInformationToolResponse = await container.azure_openai_model.generate(
        prompt=prompt_template, context={"contract_text": full_text}, response_model=KeyInformationToolResponse
    )

    return response.model_dump() if response_format == "JSON" else response
