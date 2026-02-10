from pathlib import Path
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from src.dependencies import get_service_container
from src.services.llm.azure_openai_model import AzureOpenAIModel
from src.services.vector_store.manager import get_all_chunks

# Pydantic Schemas


class Party(BaseModel):
    name: Optional[str] = Field(
        None,
        description="Full legal name of the party as stated in the document",
    )
    role: str = Field(
        ...,
        description="Standardized role of the party (DISCLOSING_PARTY, CLIENT, etc.)",
    )
    role_description: Optional[str] = Field(
        None,
        description="Custom role description if role is OTHER",
    )
    address: Optional[str] = Field(
        None,
        description="Registered address if stated in the document",
    )


class EffectiveDate(BaseModel):
    value: Optional[str] = Field(
        None,
        description="Effective date in ISO 8601 format (YYYY-MM-DD)",
        example="2024-02-02",
    )
    raw_text: Optional[str] = Field(
        None,
        description="Exact verbatim text from the document describing the effective date",
    )
    is_conditional: Optional[bool] = Field(
        None,
        description="True if the effective date is conditional",
    )
    condition: Optional[str] = Field(
        None,
        description="Condition on which the effective date depends, if any",
    )


class ExpirationDate(BaseModel):
    value: Optional[str] = Field(
        None,
        description="Expiration date in ISO 8601 format (YYYY-MM-DD)",
        example="2027-02-01",
    )
    raw_text: Optional[str] = Field(
        None,
        description="Exact verbatim text from the document describing the expiration date",
    )
    is_auto_renewing: Optional[bool] = Field(
        None,
        description="True if the agreement auto-renews",
    )
    auto_renewal_terms: Optional[str] = Field(
        None,
        description="Auto-renewal terms if specified",
    )


class ContractValue(BaseModel):
    total_value: Optional[int] = Field(
        None,
        description="Total numeric contract value without currency symbol",
        example=25000000,
    )
    currency: Optional[str] = Field(
        None,
        description="ISO 4217 currency code (e.g., INR, USD)",
        example="INR",
    )
    raw_text: Optional[str] = Field(
        None,
        description="Exact verbatim text describing the contract value",
    )
    value_type: Optional[str] = Field(
        None,
        description="FIXED | VARIABLE | ESTIMATED | NOT_TO_EXCEED | PER_UNIT | RECURRING",
    )
    recurring_amount: Optional[int] = Field(
        None,
        description="Recurring payment amount if applicable",
    )
    recurring_frequency: Optional[str] = Field(
        None,
        description="WEEKLY | MONTHLY | QUARTERLY | SEMI_ANNUALLY | ANNUALLY",
    )


class Duration(BaseModel):
    value: Optional[int] = Field(
        None,
        description="Numeric duration value",
        example=3,
    )
    unit: Optional[str] = Field(
        None,
        description="DAYS | MONTHS | YEARS",
        example="YEARS",
    )
    raw_text: Optional[str] = Field(
        None,
        description="Exact verbatim text describing duration",
    )
    is_indefinite: Optional[bool] = Field(
        None,
        description="True if duration is indefinite",
    )
    derivation: Optional[str] = Field(
        None,
        description="EXPLICIT or COMPUTED",
    )


class NetTerm(BaseModel):
    days: Optional[int] = Field(
        None,
        description="Numeric net payment days (Net 45 → 45)",
        example=45,
    )
    raw_text: Optional[str] = Field(
        None,
        description="Exact verbatim text describing payment terms",
    )
    payment_method: Optional[str] = Field(
        None,
        description="Payment method if specified (ACH, wire transfer, etc.)",
    )
    late_penalty: Optional[str] = Field(
        None,
        description="Late payment penalty clause if present",
    )
    billing_frequency: Optional[str] = Field(
        None,
        description="ONE_TIME | MONTHLY | MILESTONE_BASED | etc.",
    )


class ExtractionMetadata(BaseModel):
    total_fields_extracted: int = Field(
        ...,
        description="Count of non-null top-level extracted fields",
    )
    confidence_notes: List[str] = Field(
        default_factory=list,
        description="Notes where extraction required interpretation or ambiguity existed",
    )
    document_type_detected: str = Field(
        ...,
        description="Detected legal document type (NDA, MSA, SOW, etc.)",
    )
    fields_not_found: List[str] = Field(
        default_factory=list,
        description="List of top-level fields that were entirely null",
    )


class KeyDetailsResponse(BaseModel):

    model_config = {
        "populate_by_name": True,
        "extra": "forbid",
    }

    extraction_reasoning: str = Field(
        ...,
        alias="_extraction_reasoning",
        description="Step-by-step reasoning of what was found and not found",
    )
    parties: List[Party] = Field(
        ...,
        description="List of parties involved in the agreement",
    )
    effective_date: EffectiveDate
    expiration_date: ExpirationDate
    contract_value: ContractValue
    duration: Duration
    net_term: NetTerm
    extraction_metadata: ExtractionMetadata


# LLM Execution Logic


_llm = AzureOpenAIModel()


async def get_key_details(session_id: Optional[str] = None) -> Any:
    """
    Extract structured key contract details from the currently ingested document.
    Works with session-based ingestion.
    """

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
        # fallback (dev / legacy)
        results = get_all_chunks()

    if not results:
        raise ValueError("No document ingested. Please ingest a document first.")

    full_text = "\n\n".join(chunk.content for chunk in results.values() if getattr(chunk, "content", None))

    prompt_path = Path("src/services/prompts/v1/key_details_prompt_template.mustache")
    prompt = prompt_path.read_text(encoding="utf-8")

    return await _llm.generate(
        prompt=prompt,
        context={"text": full_text},
        response_model=KeyDetailsResponse,
    )
