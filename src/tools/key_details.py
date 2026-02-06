from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.services.llm.azure_openai_model import AzureOpenAIModel
from src.services.vector_store.manager import get_all_chunks


class KeyDetailsResponse(BaseModel):
    agreement_type: str | None = Field(None, description="Type of agreement (e.g., NDA, Service Agreement)")
    effective_date: str | None = Field(None, description="Contract effective date")
    expiration_date: str | None = Field(None, description="Contract expiration date")
    contract_value: str | None = Field(None, description="Total contract value")
    duration: str | None = Field(None, description="Contract duration")
    net_payment_terms: str | None = Field(None, description="Net payment terms")
    # termination_conditions: str | None = Field(None, description="Termination conditions")
    # governing_law: str | None = Field(None, description="Governing law jurisdiction")
    # confidentiality_clause: str | None = Field(None, description="Confidentiality clause details")


_llm = AzureOpenAIModel()


async def get_key_details() -> Any:
    """
    Extract key contractual fields from the ingested document.
    """

    chunks = get_all_chunks()
    if not chunks:
        return {"error": "No document ingested. Please ingest a document first."}

    full_text = "\n\n".join(chunk.content for chunk in chunks.values() if getattr(chunk, "content", None))

    prompt = Path("src/services/prompts/v1/key_details_prompt_template.mustache").read_text(encoding="utf-8")

    return await _llm.generate(
        prompt=prompt,
        context={"text": full_text},
        response_model=KeyDetailsResponse,
    )
