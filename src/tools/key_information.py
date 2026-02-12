from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from src.dependencies import get_service_container
from src.schemas.tool_schema import KeyInformationToolResponse
from src.services.llm.azure_openai_model import AzureOpenAIModel
from src.services.vector_store.manager import get_all_chunks

_llm = AzureOpenAIModel()


async def get_key_information(session_id: Optional[str] = None, response_format: str = "JSON") -> str | BaseModel:
    """Extract structured key contract details from the currently ingested document."""

    # Prefer session-specific chunks if session_id is provided
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

    if not results:
        raise ValueError("No document ingested. Please ingest a document first.")

    full_text = "\n\n".join(chunk.content for chunk in results.values() if getattr(chunk, "content", None))

    prompt_path = Path(r"src\services\prompts\v1\key_information_prompt.mustache").read_text(encoding="utf-8")

    response: KeyInformationToolResponse = await _llm.generate(prompt=prompt_path, context={"contract_text": full_text}, response_model=KeyInformationToolResponse)

    if response_format == "JSON":
        return response.model_dump()
    else:
        return response
