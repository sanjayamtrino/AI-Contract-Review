"""Schemas for the document comparison endpoint."""

from typing import List, Optional

from pydantic import BaseModel, Field


class CompareRequest(BaseModel):
    """Request body for the compare endpoint."""

    session_id: str
    document_id_a: str
    document_id_b: str


class ClauseComparisonLLMResponse(BaseModel):
    """Structured output schema for the per-pair LLM comparison call."""

    change_type: str = Field(description="One of: modified, reordered")
    modification_type: Optional[str] = Field(
        None,
        description="Sub-type: value, language, scope, structural, rewritten",
    )
    risk_level: str = Field(description="One of: high, medium, low")
    affected_party: Optional[str] = Field(
        None, description="Which party is affected by this change"
    )
    old_text: str = Field(description="Exact text from Document A that changed")
    new_text: str = Field(description="Exact text from Document B that changed")
    legal_implication: str = Field(
        description="Brief note on legal meaning of this change"
    )
    is_substantive: bool = Field(
        description="Whether this is a meaningful change vs cosmetic"
    )


class ChangeEntry(BaseModel):
    """A single change between two document clauses."""

    clause_name: str
    change_type: str  # added, removed, modified, reordered
    modification_type: Optional[str] = None
    risk_level: str  # high, medium, low
    affected_party: Optional[str] = None
    confidence: str = "high"  # high, medium, low
    text_from_doc_a: Optional[str] = None
    text_from_doc_b: Optional[str] = None
    legal_implication: Optional[str] = None
    is_substantive: bool = True


class SectionGroup(BaseModel):
    """A group of changes under the same section heading."""

    section_name: str
    changes: List[ChangeEntry]


class CompareSummary(BaseModel):
    """Summary statistics for the comparison."""

    total_changes: int
    added: int
    removed: int
    modified: int
    reordered: int
    overall_risk: str  # high if any high, else medium if any medium, else low
    high_risk_count: int
    llm_calls_made: int
    llm_calls_skipped: int = 0


class CompareResponse(BaseModel):
    """Top-level response for the compare endpoint."""

    success: bool
    error: Optional[str] = None
    message: Optional[str] = None
    summary: Optional[CompareSummary] = None
    sections: List[SectionGroup] = []
    document_id_a: Optional[str] = None
    document_id_b: Optional[str] = None
